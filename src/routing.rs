//! Per-model provider routing table (TCR-2).
//!
//! Pure, lock-free, network-free core (like [`crate::model`]): given a boot-time
//! [`crate::config::Config`], build a [`RoutingTable`] that answers "which upstream
//! provider(s), in what order, may serve THIS model" without touching a socket, a
//! lock or the account pool. `src/proxy.rs` walks the [`Route`] it returns; the
//! account-pool-specific bits (quota headers, per-account affinity, rate-limit
//! holds) stay entirely inside `src/manager/select.rs` — a [`Credential::Fleet`]
//! candidate is the ONLY one that ever crosses into that world, and it crosses by
//! falling through to the existing `manager.select()` path, unchanged.
//!
//! A misconfigured table must never take the proxy dark: [`RoutingTable::from_config`]
//! never panics and never returns `Err` — a typo'd candidate name, a tied priority,
//! a provider whose `models` allowlist can't actually serve a route naming it, or a
//! duplicate provider degrade to a WARNING (the candidate/provider is dropped, the
//! rest of the table stays usable). Only three conditions are severe enough that the
//! whole table degrades to fleet-only (see [`from_config`] doc): a provider name
//! collision, more than one provider declaring [`ProviderAuth::Fleet`], and an
//! unreadable or over-permissive credential file. Every message returned in the
//! `Vec<String>` is meant to be logged by the caller (`Manager::assemble`) —
//! `tracing::warn!` for the plain ones, `tracing::error!` for the ones prefixed
//! `"hard error: "`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

use axum::http::{HeaderName, HeaderValue};

use crate::config::{Config, ProviderAuth};

/// Where a [`Credential`]'s secret value came from — carried alongside the
/// resolved credential ONLY so [`RoutingTable::dump`] can report provenance
/// (env var name / file path) without ever touching the secret itself.
#[derive(Debug, Clone)]
pub enum CredentialSource {
    Fleet,
    Env(String),
    File(PathBuf),
}

/// A resolved, ready-to-inject credential for a third-party provider, or the
/// sentinel meaning "route this through the existing fleet/OAuth pool instead".
#[derive(Debug, Clone)]
pub enum Credential {
    /// Not a third-party credential at all — the caller must fall through to
    /// `manager.select()` / the existing pooled-account path.
    Fleet,
    /// Inject `value` under header `name` on every outbound request to this
    /// provider.
    Header {
        name: HeaderName,
        value: HeaderValue,
    },
}

/// One provider, fully resolved at boot: its credential loaded, its header name
/// validated, its model map ready to consult per-request.
#[derive(Debug, Clone)]
pub struct ResolvedProvider {
    pub name: String,
    pub base_url: String,
    pub credential: Credential,
    pub source: CredentialSource,
    /// Request model id -> this provider's model id. A model absent here is sent
    /// verbatim (no translation).
    pub model_map: HashMap<String, String>,
    /// Additive vendor headers, already validated into typed header parts.
    pub extra_headers: Vec<(HeaderName, HeaderValue)>,
    /// Advisory glob list from config, kept only for `dump()`.
    pub models: Vec<String>,
}

impl ResolvedProvider {
    /// Whether this provider IS the fleet (pooled-account) path.
    pub fn is_fleet(&self) -> bool {
        matches!(self.credential, Credential::Fleet)
    }

    /// The outbound model id for `requested`, translating through `model_map`
    /// when configured, else passing the request's id through unchanged.
    pub fn translate_model<'a>(&'a self, requested: &'a str) -> &'a str {
        self.model_map
            .get(requested)
            .map_or(requested, String::as_str)
    }
}

/// One compiled `model_routes[]` entry: a glob, its priority/declaration order
/// (for sorting), and the resolved candidate list (provider indices, unknown
/// names already dropped).
#[derive(Debug, Clone)]
struct CompiledRoute {
    model_glob: String,
    priority: i64,
    decl_order: usize,
    /// Indices into `RoutingTable::providers`, in the declared candidate order.
    candidates: Vec<usize>,
}

/// The boot-built routing table. Empty (`providers` and `routes` both empty) is
/// the fully inert state a config with no `providers`/`modelRoutes` keys produces
/// — every method on an inert table behaves as if TCR-2 did not exist.
#[derive(Debug, Clone)]
pub struct RoutingTable {
    providers: Vec<ResolvedProvider>,
    routes: Vec<CompiledRoute>,
}

/// A resolved candidate list for one request's model, in first-match-wins order.
/// Empty means "no rule matched — take today's first-party path", not an error.
pub struct Route<'a> {
    table: &'a RoutingTable,
    candidates: &'a [usize],
    matched: Option<(&'a str, i64)>,
}

impl<'a> Route<'a> {
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// The resolved provider at hop `i`, if any.
    pub fn get(&self, i: usize) -> Option<&'a ResolvedProvider> {
        self.candidates
            .get(i)
            .and_then(|&idx| self.table.providers.get(idx))
    }

    /// `(glob, priority)` of the rule this route matched, for logging.
    pub fn matched(&self) -> Option<(&'a str, i64)> {
        self.matched
    }

    /// Index of the (at most one — `from_config` hard-errors a table with more
    /// than one) `Fleet` candidate in this route, if any. TCR-7's
    /// fleet-exhaustion resume uses `fleet_hop() + 1` as the hop to resume
    /// [`dispatch_provider_route`] from once the pooled-account fleet is
    /// exhausted, so any candidates declared AFTER `Fleet` in the route get a
    /// chance to serve instead of the request dead-ending in a 429.
    pub fn fleet_hop(&self) -> Option<usize> {
        (0..self.candidates.len()).find(|&i| self.get(i).is_some_and(|p| p.is_fleet()))
    }
}

const EMPTY_CANDIDATES: &[usize] = &[];

impl RoutingTable {
    /// Fully inert table: no providers, no routes. `candidates_for` always
    /// returns an empty route, `dump()` is an empty document.
    fn empty() -> Self {
        Self {
            providers: Vec::new(),
            routes: Vec::new(),
        }
    }

    /// Build the routing table from `config`, never panicking and never
    /// returning `Err`. Returns the built table plus every diagnostic message —
    /// plain messages are boot-time WARNINGS (a typo'd config, degraded but
    /// usable table); messages prefixed `"hard error: "` mean the WHOLE table
    /// fell back to fleet-only (still safe to boot, TCR-2 is simply off).
    ///
    /// The three hard-error conditions (anything else degrades to a warning):
    ///   - two providers sharing one `name` (ambiguous candidate reference);
    ///   - more than one provider declaring `auth: {"kind":"fleet"}` (the doc
    ///     comment on [`ProviderAuth::Fleet`] says "exactly one provider may
    ///     declare this" — a second one makes "the fleet provider" ambiguous);
    ///   - a `File`-credential path that cannot be read, or whose permissions
    ///     are broader than owner-only (a credential file group/world-readable
    ///     is a security bug, not a routing bug — refuse to boot with it rather
    ///     than silently load a leaky secret).
    ///   - a `model_routes[]` entry exists (the operator IS doing custom
    ///     routing) but no provider anywhere declares `auth: fleet` — such a
    ///     table has no guaranteed escape hatch when every third-party
    ///     candidate is exhausted, which is exactly the shape that used to dead-
    ///     end a request instead of falling through to the existing pooled path.
    pub fn from_config(config: &Config) -> (Self, Vec<String>) {
        let mut warnings = Vec::new();

        if config.providers.is_empty() && config.model_routes.is_empty() {
            return (Self::empty(), warnings);
        }

        let mut providers: Vec<ResolvedProvider> = Vec::new();
        let mut name_to_idx: HashMap<String, usize> = HashMap::new();
        let mut fleet_count = 0usize;

        for p in &config.providers {
            if name_to_idx.contains_key(&p.name) {
                warnings.push(format!(
                    "hard error: duplicate provider name {:?} — routing disabled, falling back to fleet-only",
                    p.name
                ));
                return (Self::empty(), warnings);
            }
            let (credential, source) = match resolve_auth(&p.auth) {
                Ok(cs) => cs,
                Err(msg) => {
                    warnings.push(format!(
                        "hard error: {msg} — routing disabled, falling back to fleet-only"
                    ));
                    return (Self::empty(), warnings);
                }
            };
            if matches!(credential, Credential::Fleet) {
                fleet_count += 1;
            }
            let mut extra_headers = Vec::new();
            for (k, v) in &p.headers {
                let Some(val_str) = v.as_str() else {
                    warnings.push(format!(
                        "provider {:?}: header {k:?} value is not a string — dropped",
                        p.name
                    ));
                    continue;
                };
                match (
                    HeaderName::try_from(k.as_str()),
                    HeaderValue::from_str(val_str),
                ) {
                    (Ok(name), Ok(value)) => extra_headers.push((name, value)),
                    _ => warnings.push(format!(
                        "provider {:?}: header {k:?} is not a valid HTTP header — dropped",
                        p.name
                    )),
                }
            }
            let mut model_map = HashMap::new();
            for (k, v) in &p.model_map {
                if let Some(val_str) = v.as_str() {
                    model_map.insert(k.clone(), val_str.to_string());
                } else {
                    warnings.push(format!(
                        "provider {:?}: modelMap[{k:?}] value is not a string — dropped",
                        p.name
                    ));
                }
            }
            name_to_idx.insert(p.name.clone(), providers.len());
            providers.push(ResolvedProvider {
                name: p.name.clone(),
                base_url: p.base_url.trim_end_matches('/').to_string(),
                credential,
                source,
                model_map,
                extra_headers,
                models: p.models.clone(),
            });
        }

        if fleet_count > 1 {
            warnings.push(format!(
                "hard error: {fleet_count} providers declare auth kind \"fleet\" (exactly one may) — routing disabled, falling back to fleet-only"
            ));
            return (Self::empty(), warnings);
        }

        if fleet_count == 0 && !config.model_routes.is_empty() {
            warnings.push(
                "hard error: model_routes[] configured but no provider declares auth kind \"fleet\" — no escape hatch when every third-party candidate is exhausted; routing disabled, falling back to fleet-only".to_string(),
            );
            return (Self::empty(), warnings);
        }

        let mut routes: Vec<CompiledRoute> = Vec::new();
        for (decl_order, r) in config.model_routes.iter().enumerate() {
            let mut candidates = Vec::new();
            for name in &r.candidates {
                match name_to_idx.get(name) {
                    Some(&idx) => candidates.push(idx),
                    None => warnings.push(format!(
                        "modelRoutes[{decl_order}] ({:?}): unknown candidate provider {name:?} — dropped",
                        r.model
                    )),
                }
            }
            for &idx in &candidates {
                let p = &providers[idx];
                if !p.models.is_empty() && !p.models.iter().any(|g| glob_match(g, &r.model)) {
                    warnings.push(format!(
                        "modelRoutes[{decl_order}] ({:?}): candidate {:?} does not advertise a matching model in its `models` list — kept, but check the config",
                        r.model, p.name
                    ));
                }
            }
            if candidates.is_empty() {
                warnings.push(format!(
                    "modelRoutes[{decl_order}] ({:?}): no usable candidates left after dropping unknown names — route is dead",
                    r.model
                ));
            }
            routes.push(CompiledRoute {
                model_glob: r.model.clone(),
                priority: r.priority,
                decl_order,
                candidates,
            });
        }

        // Sort by (priority ascending, declaration order) — first-match-wins.
        routes.sort_by_key(|r| (r.priority, r.decl_order));
        // Tied priority between DIFFERENT globs is a warning: declaration order
        // silently decides it, and the operator may not have meant to rely on that.
        for w in routes.windows(2) {
            if w[0].priority == w[1].priority && w[0].model_glob != w[1].model_glob {
                warnings.push(format!(
                    "modelRoutes: {:?} and {:?} share priority {} — declaration order ({:?} before {:?}) decides which wins; set distinct priorities to make this explicit",
                    w[0].model_glob, w[1].model_glob, w[0].priority, w[0].model_glob, w[1].model_glob
                ));
            }
        }

        (Self { providers, routes }, warnings)
    }

    /// The candidates for `model`, first-match-wins by `(priority, declaration
    /// order)`. `None` (no parseable request model) or no matching rule both
    /// yield an empty route — the caller takes today's first-party path.
    pub fn candidates_for(&self, model: Option<&str>) -> Route<'_> {
        let Some(model) = model else {
            return Route {
                table: self,
                candidates: EMPTY_CANDIDATES,
                matched: None,
            };
        };
        for r in &self.routes {
            if glob_match(&r.model_glob, model) {
                return Route {
                    table: self,
                    candidates: &r.candidates,
                    matched: Some((r.model_glob.as_str(), r.priority)),
                };
            }
        }
        Route {
            table: self,
            candidates: EMPTY_CANDIDATES,
            matched: None,
        }
    }

    /// Serializable snapshot of the table — dumpable/inspectable without sending
    /// a request. NEVER carries a credential value, only its kind + provenance
    /// name (env var name / file path), so this is safe to print, log, or expose
    /// over `/_tcr/status`.
    pub fn dump(&self) -> RoutingDump {
        RoutingDump {
            providers: self
                .providers
                .iter()
                .map(|p| ProviderDump {
                    name: p.name.clone(),
                    base_url: p.base_url.clone(),
                    credential_kind: match &p.credential {
                        Credential::Fleet => "fleet",
                        Credential::Header { .. } => "header",
                    },
                    credential_source: match &p.source {
                        CredentialSource::Fleet => None,
                        CredentialSource::Env(var) => Some(format!("env:{var}")),
                        CredentialSource::File(path) => Some(format!("file:{}", path.display())),
                    },
                    models: p.models.clone(),
                    model_map: p.model_map.clone(),
                })
                .collect(),
            routes: self
                .routes
                .iter()
                .map(|r| RouteDump {
                    model: r.model_glob.clone(),
                    priority: r.priority,
                    candidates: r
                        .candidates
                        .iter()
                        .filter_map(|&i| self.providers.get(i).map(|p| p.name.clone()))
                        .collect(),
                })
                .collect(),
        }
    }
}

/// Resolve one provider's `auth` into a ready-to-use [`Credential`], or an error
/// message describing why it could not be (caller decides warn vs hard-error).
fn resolve_auth(auth: &ProviderAuth) -> Result<(Credential, CredentialSource), String> {
    match auth {
        ProviderAuth::Fleet => Ok((Credential::Fleet, CredentialSource::Fleet)),
        ProviderAuth::Env {
            var,
            header,
            prefix,
        } => {
            let Ok(secret) = std::env::var(var) else {
                return Err(format!("env var {var:?} is not set"));
            };
            let header_name = header.as_deref().unwrap_or("authorization");
            let prefix = prefix.as_deref().unwrap_or("Bearer ");
            let name = HeaderName::try_from(header_name).map_err(|_| {
                format!("header name {header_name:?} is not a valid HTTP header name")
            })?;
            let value = HeaderValue::from_str(&format!("{prefix}{secret}"))
                .map_err(|_| "credential value is not a valid HTTP header value".to_string())?;
            Ok((
                Credential::Header { name, value },
                CredentialSource::Env(var.clone()),
            ))
        }
        ProviderAuth::File {
            path,
            header,
            prefix,
        } => {
            let meta = std::fs::metadata(path)
                .map_err(|e| format!("credential file {path:?} could not be read: {e}"))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                let mode = meta.permissions().mode();
                if mode & 0o077 != 0 {
                    return Err(format!(
                        "credential file {path:?} is group/world-accessible (mode {mode:o}) — chmod 600 it"
                    ));
                }
            }
            let _ = &meta; // silence unused-on-non-unix
            let secret = std::fs::read_to_string(path)
                .map_err(|e| format!("credential file {path:?} could not be read: {e}"))?;
            let secret = secret.trim().to_string();
            let header_name = header.as_deref().unwrap_or("authorization");
            let prefix = prefix.as_deref().unwrap_or("Bearer ");
            let name = HeaderName::try_from(header_name).map_err(|_| {
                format!("header name {header_name:?} is not a valid HTTP header name")
            })?;
            let value = HeaderValue::from_str(&format!("{prefix}{secret}"))
                .map_err(|_| "credential value is not a valid HTTP header value".to_string())?;
            Ok((
                Credential::Header { name, value },
                CredentialSource::File(path.clone()),
            ))
        }
    }
}

/// Serializable routing-table snapshot. See [`RoutingTable::dump`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct RoutingDump {
    pub providers: Vec<ProviderDump>,
    pub routes: Vec<RouteDump>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProviderDump {
    pub name: String,
    pub base_url: String,
    pub credential_kind: &'static str,
    /// `env:VAR_NAME` or `file:/path` — provenance only, NEVER the secret.
    pub credential_source: Option<String>,
    pub models: Vec<String>,
    pub model_map: HashMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RouteDump {
    pub model: String,
    pub priority: i64,
    pub candidates: Vec<String>,
}

/// ASCII case-insensitive glob match, `*` only (no `?`, no character classes).
/// Deliberately hand-rolled rather than pulling in a `glob` crate for one
/// wildcard — every new dependency is a line item in `cargo audit`/`deny`, both
/// required CI checks.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<u8> = pattern.bytes().map(|b| b.to_ascii_lowercase()).collect();
    let txt: Vec<u8> = text.bytes().map(|b| b.to_ascii_lowercase()).collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut match_from = 0usize;
    while ti < txt.len() {
        if pi < pat.len() && pat[pi] == txt[ti] {
            pi += 1;
            ti += 1;
        } else if pi < pat.len() && pat[pi] == b'*' {
            star = Some(pi);
            match_from = ti;
            pi += 1;
        } else if let Some(si) = star {
            pi = si + 1;
            match_from += 1;
            ti = match_from;
        } else {
            return false;
        }
    }
    while pi < pat.len() && pat[pi] == b'*' {
        pi += 1;
    }
    pi == pat.len()
}

/// Rewrite the root `model` field of a JSON request body to `new_id`, leaving
/// everything else — including any `model` key nested inside message content —
/// untouched. Unlike [`crate::account_uuid::patch_account_uuid`] this is a full
/// parse/replace/re-serialize, not a byte-splice: a model-id rename is not
/// same-length in general (`claude-sonnet-4-6` -> `deepseek-chat`), so there is
/// no fixed-width slot to overwrite in place.
///
/// Fails safe: any parse surprise (not a JSON object, or re-serialization somehow
/// failing) returns the ORIGINAL body untouched via [`Cow::Borrowed`]. Returns
/// [`Cow::Borrowed`] too when the root `model` already equals `new_id` — no
/// allocation for a no-op translation.
pub fn patch_model<'a>(body: &'a [u8], new_id: &str) -> Cow<'a, [u8]> {
    let Ok(serde_json::Value::Object(mut map)) = serde_json::from_slice::<serde_json::Value>(body)
    else {
        return Cow::Borrowed(body);
    };
    if map.get("model").and_then(|v| v.as_str()) == Some(new_id) {
        return Cow::Borrowed(body);
    }
    map.insert(
        "model".to_string(),
        serde_json::Value::String(new_id.to_string()),
    );
    match serde_json::to_vec(&serde_json::Value::Object(map)) {
        Ok(v) => Cow::Owned(v),
        Err(_) => Cow::Borrowed(body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ModelRoute, Provider, ProviderAuth};
    use serde_json::{Map, Value};

    fn provider(name: &str, base_url: &str, auth: ProviderAuth, models: &[&str]) -> Provider {
        Provider {
            name: name.to_string(),
            base_url: base_url.to_string(),
            auth,
            models: models.iter().map(|s| s.to_string()).collect(),
            model_map: Map::new(),
            headers: Map::new(),
        }
    }

    fn route(model: &str, priority: i64, candidates: &[&str]) -> ModelRoute {
        ModelRoute {
            model: model.to_string(),
            priority,
            candidates: candidates.iter().map(|s| s.to_string()).collect(),
        }
    }

    // --- glob matching -------------------------------------------------

    #[test]
    fn glob_matches_prefix_and_suffix_wildcards() {
        assert!(glob_match("claude-sonnet-*", "claude-sonnet-4-6"));
        assert!(glob_match("*-sonnet-*", "claude-sonnet-4-6"));
        assert!(glob_match("*", "anything"));
        assert!(!glob_match("claude-opus-*", "claude-sonnet-4-6"));
    }

    #[test]
    fn glob_is_case_insensitive() {
        assert!(glob_match("CLAUDE-SONNET-*", "claude-sonnet-4-6"));
        assert!(glob_match("claude-sonnet-*", "CLAUDE-SONNET-4-6"));
    }

    #[test]
    fn glob_exact_match_no_wildcard() {
        assert!(glob_match("claude-opus-4-6", "claude-opus-4-6"));
        assert!(!glob_match("claude-opus-4-6", "claude-opus-4-7"));
    }

    // --- from_config / candidates_for -----------------------------------

    fn empty_config() -> Config {
        serde_json::from_str("{}").unwrap()
    }

    #[test]
    fn from_config_on_empty_config_is_fully_inert() {
        let (table, warnings) = RoutingTable::from_config(&empty_config());
        assert!(warnings.is_empty(), "no config, no warnings: {warnings:?}");
        assert!(table.candidates_for(Some("claude-sonnet-4-6")).is_empty());
        assert!(table.candidates_for(None).is_empty());
        let dump = table.dump();
        assert!(dump.providers.is_empty());
        assert!(dump.routes.is_empty());
    }

    #[test]
    fn candidates_for_none_model_is_empty_even_with_routes() {
        let mut config = empty_config();
        config.providers = vec![provider(
            "anthropic",
            "https://api.anthropic.com",
            ProviderAuth::Fleet,
            &[],
        )];
        config.model_routes = vec![route("*", 0, &["anthropic"])];
        let (table, _warnings) = RoutingTable::from_config(&config);
        assert!(table.candidates_for(None).is_empty());
    }

    #[test]
    fn candidates_for_unmatched_model_is_empty() {
        let mut config = empty_config();
        config.providers = vec![provider(
            "anthropic",
            "https://api.anthropic.com",
            ProviderAuth::Fleet,
            &[],
        )];
        config.model_routes = vec![route("claude-opus-*", 0, &["anthropic"])];
        let (table, _warnings) = RoutingTable::from_config(&config);
        assert!(table.candidates_for(Some("claude-sonnet-4-6")).is_empty());
    }

    #[test]
    fn priority_ordering_lower_wins() {
        let mut config = empty_config();
        config.providers = vec![
            provider("a", "https://a.example.com", ProviderAuth::Fleet, &[]),
            provider(
                "b",
                "https://b.example.com",
                ProviderAuth::Env {
                    var: "TCR2_TEST_PRIORITY_VAR".into(),
                    header: None,
                    prefix: None,
                },
                &[],
            ),
        ];
        std::env::set_var("TCR2_TEST_PRIORITY_VAR", "secret");
        config.model_routes = vec![
            route("claude-sonnet-*", 5, &["a"]),
            route("claude-sonnet-*", 0, &["b"]),
        ];
        let (table, _warnings) = RoutingTable::from_config(&config);
        let r = table.candidates_for(Some("claude-sonnet-4-6"));
        assert_eq!(r.matched().map(|(_, p)| p), Some(0));
        assert_eq!(r.get(0).map(|p| p.name.as_str()), Some("b"));
        std::env::remove_var("TCR2_TEST_PRIORITY_VAR");
    }

    #[test]
    fn tied_priority_falls_back_to_declaration_order_and_warns() {
        let mut config = empty_config();
        config.providers = vec![provider(
            "a",
            "https://a.example.com",
            ProviderAuth::Fleet,
            &[],
        )];
        // Two DIFFERENT globs at the same priority — declaration order decides.
        config.model_routes = vec![
            route("claude-sonnet-*", 0, &["a"]),
            route("claude-opus-*", 0, &["a"]),
        ];
        let (_table, warnings) = RoutingTable::from_config(&config);
        assert!(
            warnings.iter().any(|w| w.contains("share priority")),
            "expected a tie warning, got {warnings:?}"
        );
    }

    #[test]
    fn tied_priority_declaration_order_actually_wins_first_declared() {
        let mut config = empty_config();
        config.providers = vec![provider(
            "a",
            "https://a.example.com",
            ProviderAuth::Fleet,
            &[],
        )];
        config.model_routes = vec![
            route("claude-*", 0, &["a"]), // declared first
            route("claude-*", 0, &["a"]), // identical glob, declared second
        ];
        let (table, _warnings) = RoutingTable::from_config(&config);
        let r = table.candidates_for(Some("claude-sonnet-4-6"));
        // Both rules match the same model; the FIRST declared at the tied
        // priority must be the one selected (first-match-wins, not merged).
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn unknown_candidate_name_is_dropped_and_warned_table_stays_usable() {
        let mut config = empty_config();
        config.providers = vec![provider(
            "anthropic",
            "https://api.anthropic.com",
            ProviderAuth::Fleet,
            &[],
        )];
        config.model_routes = vec![route("claude-sonnet-*", 0, &["ghost", "anthropic"])];
        let (table, warnings) = RoutingTable::from_config(&config);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("ghost") && w.contains("unknown candidate")),
            "expected an unknown-candidate warning, got {warnings:?}"
        );
        let r = table.candidates_for(Some("claude-sonnet-4-6"));
        assert_eq!(
            r.len(),
            1,
            "the ghost candidate is dropped, anthropic remains"
        );
        assert_eq!(r.get(0).map(|p| p.name.as_str()), Some("anthropic"));
    }

    #[test]
    fn model_routes_without_a_fleet_provider_is_a_hard_error() {
        let mut config = empty_config();
        config.providers = vec![provider(
            "deepseek",
            "https://api.deepseek.com/anthropic",
            ProviderAuth::Env {
                var: "TCR2_TEST_NOFLEET_VAR".into(),
                header: None,
                prefix: None,
            },
            &[],
        )];
        std::env::set_var("TCR2_TEST_NOFLEET_VAR", "secret");
        config.model_routes = vec![route("claude-sonnet-*", 0, &["deepseek"])];
        let (table, warnings) = RoutingTable::from_config(&config);
        assert!(warnings.iter().any(|w| w.starts_with("hard error:")));
        assert!(table.candidates_for(Some("claude-sonnet-4-6")).is_empty());
        std::env::remove_var("TCR2_TEST_NOFLEET_VAR");
    }

    #[test]
    fn duplicate_provider_name_is_a_hard_error() {
        let mut config = empty_config();
        config.providers = vec![
            provider("dup", "https://a.example.com", ProviderAuth::Fleet, &[]),
            provider("dup", "https://b.example.com", ProviderAuth::Fleet, &[]),
        ];
        let (table, warnings) = RoutingTable::from_config(&config);
        assert!(warnings
            .iter()
            .any(|w| w.starts_with("hard error:") && w.contains("duplicate")));
        assert!(table.dump().providers.is_empty());
    }

    // --- dump() never leaks a credential value --------------------------

    #[test]
    fn dump_never_contains_credential_value_but_does_contain_provenance_name() {
        let mut config = empty_config();
        config.providers = vec![
            provider(
                "anthropic",
                "https://api.anthropic.com",
                ProviderAuth::Fleet,
                &[],
            ),
            provider(
                "deepseek",
                "https://api.deepseek.com/anthropic",
                ProviderAuth::Env {
                    var: "TCR2_TEST_DUMP_SECRET".into(),
                    header: None,
                    prefix: None,
                },
                &["claude-sonnet-*"],
            ),
        ];
        std::env::set_var("TCR2_TEST_DUMP_SECRET", "sk-super-secret-value-do-not-leak");
        config.model_routes = vec![route("claude-sonnet-*", 0, &["deepseek", "anthropic"])];
        let (table, _warnings) = RoutingTable::from_config(&config);
        let dump = table.dump();
        let json = serde_json::to_string(&dump).unwrap();

        // Positive control: an empty search proves nothing unless we also prove
        // the thing we expect to find really would show up.
        assert!(
            json.contains("TCR2_TEST_DUMP_SECRET"),
            "provenance (env var NAME) must be present: {json}"
        );
        assert!(
            !json.contains("sk-super-secret-value-do-not-leak"),
            "the actual secret value must never appear in the dump: {json}"
        );
        std::env::remove_var("TCR2_TEST_DUMP_SECRET");
    }

    // --- patch_model -----------------------------------------------------

    #[test]
    fn patch_model_rewrites_only_root_model() {
        let body = br#"{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":{"model":"nested-should-not-change"}}]}"#;
        let out = patch_model(body, "deepseek-chat");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "deepseek-chat");
        assert_eq!(
            v["messages"][0]["content"]["model"],
            "nested-should-not-change"
        );
    }

    #[test]
    fn patch_model_noop_when_already_equal_borrows() {
        let body = br#"{"model":"deepseek-chat","messages":[]}"#;
        let out = patch_model(body, "deepseek-chat");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn patch_model_on_non_json_passes_through() {
        let out = patch_model(b"not json", "deepseek-chat");
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(&*out, b"not json");
    }
}
