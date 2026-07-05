//! Floor 1 — the sensory buffer.
//!
//! A capacity-bounded ring of raw text frames with surprise-gated
//! promotion to episodic memory. Surprise is *recall-shaped novelty*:
//! `1 − max_similarity(gist(text), gists of the most recent
//! [`SURPRISE_REFERENCE_WINDOW`] episodes)` — deliberately not a
//! belief-based prediction (that is a documented follow-up). Duplicate
//! or near-duplicate signal scores near 0 and stays in the buffer;
//! novel signal scores near 0.5+ and is promoted as a text-only
//! episode with `provenance.source = "sensory"`.

use chrono::{DateTime, Utc};
use redb::{ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::episode::encode_gist_signature;
use crate::error::Result;
use crate::hdc::HV;
use crate::store::{decode, encode, Store};
use crate::types::{Episode, EpisodeId, Provenance, TimeRange};

/// Ring of raw frames — `frame_id → SensoryFrame`.
pub const SENSORY_FRAMES: TableDefinition<u64, Vec<u8>> = TableDefinition::new("sensory_frames");

/// Manifest key for the monotonic frame-id counter.
const KEY_NEXT_SENSORY_ID: &str = "next_sensory_id";

/// Maximum frames retained in the ring buffer.
pub const DEFAULT_SENSORY_CAPACITY: u64 = 1000;

/// Frames whose surprise is at or above this are promoted to episodic
/// memory. Random text against unrelated context scores ≈ 0.5;
/// verbatim repetition scores ≈ 0.0.
pub const SURPRISE_PROMOTION_THRESHOLD: f32 = 0.4;

/// How many recent episode gists the surprise score compares against.
const SURPRISE_REFERENCE_WINDOW: usize = 64;

/// One raw frame in the sensory ring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensoryFrame {
    pub id: u64,
    pub text: String,
    pub at: DateTime<Utc>,
    pub surprise: f32,
    /// Set when the frame crossed the promotion threshold.
    pub promoted: Option<EpisodeId>,
}

/// What `observe_sensory` did.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensoryObservation {
    pub frame_id: u64,
    pub surprise: f32,
    pub promoted: Option<EpisodeId>,
}

impl Store {
    /// Surprise of `text` against the most recent episodes' gists.
    /// 1.0 on an empty store (everything is novel); 0.0 for empty or
    /// whitespace-only text (nothing to remember).
    pub fn surprise_score(&self, text: &str) -> Result<f32> {
        let gist = encode_gist_signature(text);
        if gist == HV::zero() {
            return Ok(0.0);
        }
        let mut max_sim = f32::MIN;
        let mut seen = 0usize;
        for entry in self.scan_entries().iter().rev() {
            if entry.tombstoned {
                continue;
            }
            let Ok(hv) = self.signatures.read(entry.gist_offset) else {
                continue;
            };
            max_sim = max_sim.max(gist.similarity(&hv));
            seen += 1;
            if seen >= SURPRISE_REFERENCE_WINDOW {
                break;
            }
        }
        if seen == 0 {
            return Ok(1.0);
        }
        Ok((1.0 - max_sim).clamp(0.0, 1.0))
    }

    /// Record a raw frame; promote it to an episode when its surprise
    /// crosses [`SURPRISE_PROMOTION_THRESHOLD`]. The ring keeps the
    /// last [`DEFAULT_SENSORY_CAPACITY`] frames.
    pub fn observe_sensory(&mut self, text: &str) -> Result<SensoryObservation> {
        let at = Utc::now();
        let surprise = self.surprise_score(text)?;

        let promoted = if surprise >= SURPRISE_PROMOTION_THRESHOLD {
            let id = self.next_episode_id()?;
            let gist = encode_gist_signature(text);
            let episode = Episode {
                id,
                text: text.to_string(),
                signature_offset: 0,
                gist_offset: 0,
                triples: vec![],
                valid_time: TimeRange::point(at),
                t_tx_start: at,
                provenance: Provenance {
                    source: "sensory".into(),
                    ..Provenance::default()
                },
                confidence: 0.5,
                superseded_by: None,
            };
            Some(self.observe(episode, &gist)?)
        } else {
            None
        };

        let frame_id = self.next_sensory_id()?;
        let frame = SensoryFrame {
            id: frame_id,
            text: text.to_string(),
            at,
            surprise,
            promoted,
        };
        let tx = self.db.begin_write()?;
        {
            let mut table = tx.open_table(SENSORY_FRAMES)?;
            table.insert(frame_id, encode(&frame)?)?;
            if frame_id > DEFAULT_SENSORY_CAPACITY {
                table.remove(frame_id - DEFAULT_SENSORY_CAPACITY)?;
            }
        }
        tx.commit()?;

        Ok(SensoryObservation {
            frame_id,
            surprise,
            promoted,
        })
    }

    /// Up to `limit` frames, newest first.
    pub fn sensory_frames(&self, limit: usize) -> Result<Vec<SensoryFrame>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(SENSORY_FRAMES)?;
        let mut out = Vec::new();
        for entry in table.iter()?.rev() {
            if out.len() >= limit {
                break;
            }
            let (_, v) = entry?;
            out.push(decode(&v.value())?);
        }
        Ok(out)
    }

    fn next_sensory_id(&mut self) -> Result<u64> {
        let tx = self.db.begin_write()?;
        let id;
        {
            let mut manifest = tx.open_table(crate::store::MANIFEST)?;
            let raw = manifest.get(KEY_NEXT_SENSORY_ID)?.map(|v| v.value());
            let current: u64 = match raw {
                Some(bytes) => decode(&bytes)?,
                None => 1,
            };
            manifest.insert(KEY_NEXT_SENSORY_ID, encode(&(current + 1))?)?;
            id = current;
        }
        tx.commit()?;
        Ok(id)
    }
}
