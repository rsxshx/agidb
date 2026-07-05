//! Server-side context: the open `Store` + an extractor (real or null).
//!
//! Wraps the layer-3 + layer-2 surfaces the tool handlers need behind a
//! single thread-safe API. The Store sits behind a `Mutex` because every
//! tool call may mutate it (observe, consolidate); the extractor is
//! immutable after construction.

use std::sync::Mutex;

use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{
    Episode, EpisodeId, ExtractContext, Extraction, Provenance, Query, Recall, TextExtractor,
};
use agidb_core::Result as AgidbResult;
use agidb_extract::{observe_text, ExtractorConfig, ObserveContext};

/// Either a real `Extractor` (loaded GLiNER + heuristic relations) or a
/// `NullExtractor` that returns empty extractions. We use the null
/// variant when the model cache is empty, so the MCP server still starts
/// and serves recall / consolidate / get_episode normally — observe
/// just stores text-only episodes until the user warms the cache.
///
/// The real variant is boxed: `Extractor` carries a GLiNER ONNX session
/// (~tens of KB on the stack from the wrapper types) while `Null` is a
/// unit variant, and clippy (correctly) complains about the size delta.
pub enum AgidbExtractor {
    Real(Box<agidb_extract::Extractor>),
    Null,
}

impl TextExtractor for AgidbExtractor {
    fn extract(&self, text: &str, ctx: &ExtractContext) -> AgidbResult<Extraction> {
        match self {
            Self::Real(e) => e.extract(text, ctx),
            Self::Null => Ok(Extraction {
                triples: Vec::new(),
                valid_time: None,
                raw_entities: Vec::new(),
            }),
        }
    }
}

pub struct AgidbContext {
    store: Mutex<Store>,
    extractor: AgidbExtractor,
}

impl AgidbContext {
    /// Open the store at `db_path` and try to load the extractor. If
    /// model artifacts are missing the extractor degrades to null so
    /// the server still starts.
    pub fn open(db_path: &str) -> AgidbResult<Self> {
        let store = Store::open(StoreConfig::at(db_path))?;
        let extractor = match agidb_extract::Extractor::new(ExtractorConfig::default()) {
            Ok(e) => {
                tracing::info!("loaded real Extractor");
                AgidbExtractor::Real(Box::new(e))
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "no Extractor loaded (model cache cold?); falling back to NullExtractor — \
                     observe will store text-only episodes"
                );
                AgidbExtractor::Null
            }
        };
        Ok(Self {
            store: Mutex::new(store),
            extractor,
        })
    }

    /// Test-only constructor: open a fresh store at `db_path` with the
    /// `NullExtractor` (no model load attempt).
    pub fn open_null(db_path: &str) -> AgidbResult<Self> {
        let store = Store::open(StoreConfig::at(db_path))?;
        Ok(Self {
            store: Mutex::new(store),
            extractor: AgidbExtractor::Null,
        })
    }

    pub fn observe_text(&self, text: &str, source: &str) -> AgidbResult<EpisodeId> {
        let mut store = self.store.lock().expect("store mutex poisoned");
        let ctx = ObserveContext {
            observation_time: chrono::Utc::now(),
            provenance: Provenance {
                source: source.to_string(),
                ..Provenance::default()
            },
        };
        observe_text(&mut store, &self.extractor, text, ctx)
    }

    pub fn recall(&self, query: &Query) -> AgidbResult<Recall> {
        let store = self.store.lock().expect("store mutex poisoned");
        store.recall(query)
    }

    pub fn consolidate(&self) -> AgidbResult<agidb_core::consolidate::ConsolidationReport> {
        let mut store = self.store.lock().expect("store mutex poisoned");
        store.consolidate()
    }

    pub fn get_episode(&self, id: u64) -> AgidbResult<Option<Episode>> {
        let store = self.store.lock().expect("store mutex poisoned");
        store.get_episode(EpisodeId::new(id))
    }

    /// Run a read-only closure against the store.
    pub fn with_store<T>(&self, f: impl FnOnce(&Store) -> AgidbResult<T>) -> AgidbResult<T> {
        let store = self.store.lock().expect("store mutex poisoned");
        f(&store)
    }

    /// Run a mutating closure against the store.
    pub fn with_store_mut<T>(
        &self,
        f: impl FnOnce(&mut Store) -> AgidbResult<T>,
    ) -> AgidbResult<T> {
        let mut store = self.store.lock().expect("store mutex poisoned");
        f(&mut store)
    }
}
