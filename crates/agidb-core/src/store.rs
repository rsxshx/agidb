//! `store` — the redb-backed metadata layer.
//!
//! Layer 3 plumbing. Every Episode, Concept, and SemanticAtom row
//! lives here; HVs themselves live in [`crate::signatures`] and are
//! referenced by `signature_offset`.

use crate::error::{AgidbError, Result};
use crate::hdc::{D_BYTES, HV};
use crate::signatures::SignatureFile;
use crate::types::*;
use chrono::{DateTime, Duration, Utc};
use redb::{
    Database, MultimapTableDefinition, ReadableTable, ReadableTableMetadata, TableDefinition,
};
use roaring::RoaringBitmap;
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Table definitions — the redb schema.
//
// Values are bincode-serialized blobs so we can evolve the in-Rust types
// without rewriting the on-disk format on every change. The
// `format_version` in the manifest table gates breaking changes.
// ---------------------------------------------------------------------------

/// Primary table — every Episode by id.
pub const EPISODES: TableDefinition<u64, Vec<u8>> = TableDefinition::new("episodes");

/// Every Concept by id.
pub const CONCEPTS: TableDefinition<u64, Vec<u8>> = TableDefinition::new("concepts");

/// `entity_name → ConceptId.raw()` lookup. Includes canonical names
/// *and* aliases (layer-2 alias resolution writes both forms here).
pub const CONCEPT_BY_NAME: TableDefinition<&str, u64> = TableDefinition::new("concept_by_name");

/// `ConceptId → many EpisodeId`. Drives tier-A exact recall.
pub const CONCEPT_EPISODES: MultimapTableDefinition<u64, u64> =
    MultimapTableDefinition::new("concept_episodes");

/// Inverted index from an HV active-dim index to a roaring bitmap of
/// `EpisodeId` low 32 bits (sufficient for v0.1 single-node scale).
pub const INVERTED_INDEX: TableDefinition<u32, Vec<u8>> = TableDefinition::new("inverted_index");

/// Every SemanticAtom by id.
pub const SEMANTIC_ATOMS: TableDefinition<u64, Vec<u8>> = TableDefinition::new("semantic_atoms");

/// Append-only audit trail of every consolidation pass.
pub const CONSOLIDATION_LOG: TableDefinition<u64, Vec<u8>> =
    TableDefinition::new("consolidation_log");

/// Manifest values (`format_version`, monotonic counters, …).
pub const MANIFEST: TableDefinition<&str, Vec<u8>> = TableDefinition::new("manifest");

/// Manifest key for the format version u32.
const KEY_FORMAT_VERSION: &str = "format_version";

/// On-disk store format version, persisted in the manifest.
///
/// v2: `Episode` gained `gist_offset` (a second HV per episode in
/// `signatures.dat`). v1 stores fail to open with `FormatVersion`;
/// migrate by exporting JSONL from a v1 build and importing here
/// (`gist_offset` is `#[serde(default)]` so old exports load).
pub const STORE_FORMAT_VERSION: u32 = 2;

/// Manifest key for the next-concept-id counter (u64).
const KEY_NEXT_CONCEPT_ID: &str = "next_concept_id";

/// Manifest key for the next-episode-id counter (u64). Used by
/// `agidb-extract::observe_text` to mint ids monotonically across
/// reopens. Caller-supplied ids in `Store::observe` (used by the phase-2
/// tests) bypass this counter — collisions remain last-writer-wins
/// until the phase-4 sequence-counter contract.
const KEY_NEXT_EPISODE_ID: &str = "next_episode_id";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for opening a store. Defaults match the v0.1 targets
/// in [`crate::types`] / `docs/spec/tech-spec.md`.
#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub root: PathBuf,
    pub strict: bool,
    pub format_version: u32,
}

impl StoreConfig {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            strict: false,
            format_version: STORE_FORMAT_VERSION,
        }
    }
}

/// One log entry per consolidation pass; written into `CONSOLIDATION_LOG`.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationLogEntry {
    pub at: DateTime<Utc>,
    pub episodes_scanned: u32,
    pub atoms_created: u32,
    pub contradictions: u32,
    pub atoms_decayed: u32,
    pub bytes_reclaimed: u64,
}

/// A snapshot of store-wide counts. Returned by [`Store::stats`].
#[derive(Clone, Debug, serde::Serialize)]
pub struct Stats {
    pub episodes: u64,
    pub concepts: u64,
    pub semantic_atoms: u64,
    pub goals: u64,
    pub beliefs: u64,
    pub consolidation_passes: u64,
    /// Number of HVs stored in `signatures.dat` (each is 1024 bytes).
    pub signatures: u64,
}

/// One row of the in-memory scan directory — the compact per-episode
/// record the tier-B/C/D scans sweep instead of deserializing full
/// `Episode` rows out of redb. ~64 bytes per episode; rebuilt from the
/// `EPISODES` + `TOMBSTONES` tables at open and kept in sync by every
/// mutation path (`observe`, `supersede`, tombstone write/restore).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanEntry {
    pub id: u64,
    pub sig_offset: u64,
    pub gist_offset: u64,
    /// Cached popcount of the HV at `sig_offset` — used by the tier-B
    /// phi scoring so the scan does one popcount per pair, not two.
    pub sig_popcount: u32,
    pub valid_start: DateTime<Utc>,
    pub valid_end: Option<DateTime<Utc>>,
    pub tombstoned: bool,
}

impl ScanEntry {
    /// Bi-temporal filter — mirrors `TimeRange::contains`.
    pub fn valid_at(&self, t: DateTime<Utc>) -> bool {
        t >= self.valid_start && self.valid_end.map(|e| t <= e).unwrap_or(true)
    }
}

/// Owning handle to a agidb store — the redb database + the mmap'd
/// signatures file held together.
pub struct Store {
    pub db: Database,
    pub signatures: SignatureFile,
    pub config: StoreConfig,
    /// In-memory scan directory over every episode (see [`ScanEntry`]).
    scan_dir: Vec<ScanEntry>,
    /// `episode_id → index into scan_dir` for O(1) updates.
    scan_pos: std::collections::HashMap<u64, usize>,
}

impl Store {
    /// Open or create the store at `config.root`. Idempotent — opening
    /// an existing store verifies the manifest's format version.
    pub fn open(config: StoreConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.root)?;
        let db_path = config.root.join("meta.redb");
        let sig_path = config.root.join("signatures.dat");

        let db = Database::create(&db_path)?;
        let signatures = SignatureFile::open(&sig_path)?;

        // Initialize / verify manifest + create every table so later
        // read-only transactions don't trip the "table does not exist"
        // error on an empty store.
        {
            let tx = db.begin_write()?;
            {
                let mut manifest = tx.open_table(MANIFEST)?;
                let stored_version = manifest.get(KEY_FORMAT_VERSION)?.map(|v| v.value());
                match stored_version {
                    Some(bytes) => {
                        let stored: u32 = decode(&bytes)?;
                        if stored != config.format_version {
                            return Err(AgidbError::FormatVersion {
                                got: stored,
                                expected: config.format_version,
                            });
                        }
                    }
                    None => {
                        manifest.insert(KEY_FORMAT_VERSION, encode(&config.format_version)?)?;
                    }
                }
                let has_counter = manifest.get(KEY_NEXT_CONCEPT_ID)?.is_some();
                if !has_counter {
                    manifest.insert(KEY_NEXT_CONCEPT_ID, encode(&1u64)?)?;
                }
                // Touch every table so it exists. redb materializes
                // a table on the first open_table inside a write tx.
                let _ = tx.open_table(EPISODES)?;
                let _ = tx.open_table(CONCEPTS)?;
                let _ = tx.open_table(CONCEPT_BY_NAME)?;
                let _ = tx.open_multimap_table(CONCEPT_EPISODES)?;
                let _ = tx.open_table(INVERTED_INDEX)?;
                let _ = tx.open_table(SEMANTIC_ATOMS)?;
                let _ = tx.open_table(CONSOLIDATION_LOG)?;
                // Phase 9 — cognitive primitives (additive; v1 stores
                // simply have empty goals/beliefs tables).
                let _ = tx.open_table(crate::goal::GOALS)?;
                let _ = tx.open_table(crate::belief::BELIEFS)?;
                let _ = tx.open_table(crate::belief::BELIEF_REVISIONS)?;
                // Phase 10 — self-model audit log + self-vector history.
                let _ = tx.open_table(crate::learning_log::LEARNING_EVENTS)?;
                let _ = tx.open_table(crate::self_model::SELF_VECTOR_HISTORY)?;
                // Phase 11 — unlearn tombstones.
                let _ = tx.open_table(crate::unlearn::TOMBSTONES)?;
            }
            tx.commit()?;
        }

        let mut store = Self {
            db,
            signatures,
            config,
            scan_dir: Vec::new(),
            scan_pos: std::collections::HashMap::new(),
        };
        store.rebuild_scan_dir()?;
        Ok(store)
    }

    /// Rebuild the in-memory scan directory from the `EPISODES` and
    /// `TOMBSTONES` tables. One full-table decode, paid once at open.
    fn rebuild_scan_dir(&mut self) -> Result<()> {
        let tx = self.db.begin_read()?;
        let tombstoned: BTreeSet<u64> = {
            let table = tx.open_table(crate::unlearn::TOMBSTONES)?;
            let mut set = BTreeSet::new();
            for entry in table.iter()? {
                let (k, _) = entry?;
                let (kind, id) = k.value();
                if kind == crate::unlearn::TOMBSTONE_EPISODE {
                    set.insert(id);
                }
            }
            set
        };
        let table = tx.open_table(EPISODES)?;
        self.scan_dir.clear();
        self.scan_pos.clear();
        let mut entries: Vec<ScanEntry> = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry?;
            let ep: Episode = decode(&v.value())?;
            let sig_popcount = self
                .signatures
                .read(ep.signature_offset)
                .map(|hv| hv.popcount())
                .unwrap_or(0);
            entries.push(ScanEntry {
                id: ep.id.raw(),
                sig_offset: ep.signature_offset,
                gist_offset: ep.gist_offset,
                sig_popcount,
                valid_start: ep.valid_time.start,
                valid_end: ep.valid_time.end,
                tombstoned: tombstoned.contains(&ep.id.raw()),
            });
        }
        for entry in entries {
            self.scan_push(entry);
        }
        Ok(())
    }

    /// Insert-or-replace a scan-directory entry.
    fn scan_push(&mut self, entry: ScanEntry) {
        match self.scan_pos.get(&entry.id) {
            Some(&pos) => self.scan_dir[pos] = entry,
            None => {
                self.scan_pos.insert(entry.id, self.scan_dir.len());
                self.scan_dir.push(entry);
            }
        }
    }

    /// The scan directory, in insertion order. Read by the recall tiers.
    pub(crate) fn scan_entries(&self) -> &[ScanEntry] {
        &self.scan_dir
    }

    /// Scan-directory entry for one episode, if present.
    pub(crate) fn scan_entry(&self, id: u64) -> Option<&ScanEntry> {
        self.scan_pos.get(&id).map(|&pos| &self.scan_dir[pos])
    }

    /// Flip the tombstone flag on one episode's scan entry. Called by
    /// the unlearn/restore paths right after they mutate `TOMBSTONES`.
    pub(crate) fn scan_set_tombstoned(&mut self, id: u64, tombstoned: bool) {
        if let Some(&pos) = self.scan_pos.get(&id) {
            self.scan_dir[pos].tombstoned = tombstoned;
        }
    }

    /// Open or create + initialize the self-vector. Convenience wrapper
    /// so callers don't need to remember the init step.
    pub fn open_initialized(config: StoreConfig) -> Result<Self> {
        let mut store = Self::open(config)?;
        store.init_self_vector()?;
        Ok(store)
    }

    /// Persist an Episode + its HV signature in one transactional unit.
    /// Updates the concept index, the concept-by-name lookup, the
    /// concept→episodes multimap, and the inverted index in the same
    /// commit.
    ///
    /// The caller's `episode.id` is used as-is — phase 2 trusts the
    /// caller to supply unique ids. Collisions overwrite (last-writer-
    /// wins) until a phase-4 sequence-counter lands.
    pub fn observe(&mut self, mut episode: Episode, signature: &HV) -> Result<EpisodeId> {
        // 1. Append the signatures outside the redb tx — the mmap and
        //    redb have independent commit cycles, but the offsets are
        //    only "live" once the redb row that references them commits,
        //    and a crash before commit leaves at most junk HVs at the
        //    tail of signatures.dat (no dangling reference).
        //
        //    Two HVs per episode: the caller's structured signature and
        //    the gist HV of the raw text. Persisting the gist here is
        //    what lets the tier-C/D scan sweep the mmap instead of
        //    re-encoding every episode's text per query.
        let offset = self.signatures.append(signature)?;
        episode.signature_offset = offset;
        let gist = crate::episode::encode_gist_signature(&episode.text);
        episode.gist_offset = if &gist == signature {
            // Extraction-less episodes pass the gist *as* the signature;
            // don't store the same HV twice.
            offset
        } else {
            self.signatures.append(&gist)?
        };
        let episode_id = episode.id;
        let active_dims: Vec<u32> = signature.active_dims().collect();

        // 2. One redb transaction for the row + every index update.
        let tx = self.db.begin_write()?;
        {
            let mut episodes = tx.open_table(EPISODES)?;
            let mut concepts = tx.open_table(CONCEPTS)?;
            let mut concept_by_name = tx.open_table(CONCEPT_BY_NAME)?;
            let mut concept_episodes = tx.open_multimap_table(CONCEPT_EPISODES)?;
            let mut inverted = tx.open_table(INVERTED_INDEX)?;
            let mut manifest = tx.open_table(MANIFEST)?;

            episodes.insert(episode_id.raw(), encode(&episode)?)?;

            // For each subject and object in each triple, ensure the
            // corresponding Concept exists and link it to this episode.
            let mut seen: BTreeSet<String> = BTreeSet::new();
            for tr in &episode.triples {
                for entity_name in [&tr.subject, &tr.object] {
                    if !seen.insert(entity_name.clone()) {
                        continue;
                    }
                    // Materialize the looked-up value (or None) before
                    // any mutating call on the same table — redb's
                    // AccessGuard borrows the table immutably and would
                    // collide with the next insert otherwise.
                    let existing = concept_by_name
                        .get(entity_name.as_str())?
                        .map(|v| v.value());
                    let concept_id = match existing {
                        Some(raw) => ConceptId::new(raw),
                        None => {
                            let new_id = next_concept_id(&mut manifest)?;
                            let concept = Concept {
                                id: new_id,
                                canonical_name: entity_name.clone(),
                                aliases: vec![],
                                entity_type: "unknown".into(),
                            };
                            concepts.insert(new_id.raw(), encode(&concept)?)?;
                            concept_by_name.insert(entity_name.as_str(), new_id.raw())?;
                            new_id
                        }
                    };
                    concept_episodes.insert(concept_id.raw(), episode_id.raw())?;
                }
            }

            // Inverted index: each active dim of the HV gains a
            // pointer to this episode. Roaring bitmaps keep the index
            // compact even with millions of episodes.
            for dim in active_dims {
                let existing = inverted.get(dim)?.map(|v| v.value());
                let mut bitmap = match existing {
                    Some(bytes) => RoaringBitmap::deserialize_from(bytes.as_slice())
                        .map_err(|e| AgidbError::Internal(format!("roaring decode: {e}")))?,
                    None => RoaringBitmap::new(),
                };
                bitmap.insert(episode_id.raw() as u32);
                let mut bytes = Vec::with_capacity(bitmap.serialized_size());
                bitmap
                    .serialize_into(&mut bytes)
                    .map_err(|e| AgidbError::Internal(format!("roaring encode: {e}")))?;
                inverted.insert(dim, bytes)?;
            }
        }
        tx.commit()?;
        self.signatures.flush()?;

        // Keep the in-memory scan directory in sync with the row that
        // just committed.
        self.scan_push(ScanEntry {
            id: episode_id.raw(),
            sig_offset: episode.signature_offset,
            gist_offset: episode.gist_offset,
            sig_popcount: signature.popcount(),
            valid_start: episode.valid_time.start,
            valid_end: episode.valid_time.end,
            tombstoned: false,
        });

        // Phase 10 — emit a learning event (after the tx commits so
        // record_event's own write tx doesn't deadlock).
        let _ = self.record_event(crate::learning_log::LearningEvent::EpisodeStored {
            id: episode_id,
            at: Utc::now(),
        });

        Ok(episode_id)
    }

    /// Fetch a Concept's canonical name by id. Used by
    /// `agidb-extract::observe_text` to translate alias-resolved
    /// `ConceptId`s back into the canonical names that `Triple` stores
    /// — important when fuzzy match merges a typo'd mention into an
    /// existing canonical concept ("Bandar" → existing "Bandra").
    pub fn concept_canonical_name(&self, id: ConceptId) -> Result<Option<String>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(CONCEPTS)?;
        match table.get(id.raw())? {
            Some(v) => {
                let concept: Concept = decode(&v.value())?;
                Ok(Some(concept.canonical_name))
            }
            None => Ok(None),
        }
    }

    /// Fetch an Episode by id.
    pub fn get_episode(&self, id: EpisodeId) -> Result<Option<Episode>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(EPISODES)?;
        match table.get(id.raw())? {
            Some(v) => Ok(Some(decode::<Episode>(&v.value())?)),
            None => Ok(None),
        }
    }

    /// Look up a ConceptId by canonical name or alias.
    pub fn concept_id_for(&self, name: &str) -> Result<Option<ConceptId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(CONCEPT_BY_NAME)?;
        match table.get(name)? {
            Some(v) => Ok(Some(ConceptId::new(v.value()))),
            None => Ok(None),
        }
    }

    /// Tier-A exact recall. Returns every Episode that references
    /// `concept`, optionally filtered to the bi-temporal slice valid
    /// at `as_of`. Order is unspecified (callers sort if they need a
    /// specific ordering).
    pub fn recall_exact(
        &self,
        concept: ConceptId,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<Episode>> {
        let tx = self.db.begin_read()?;
        let concept_episodes = tx.open_multimap_table(CONCEPT_EPISODES)?;
        let episodes = tx.open_table(EPISODES)?;

        let mut results = Vec::new();
        for raw in concept_episodes.get(concept.raw())? {
            let raw_id = raw?.value();
            if let Some(v) = episodes.get(raw_id)? {
                let ep: Episode = decode(&v.value())?;
                if let Some(t) = as_of {
                    if !ep.valid_time.contains(t) {
                        continue;
                    }
                }
                results.push(ep);
            }
        }
        Ok(results)
    }

    /// Mark `older` as superseded by `newer`. Closes the old
    /// `valid_time` interval at `newer.valid_time.start - 1ms` and
    /// writes the `superseded_by` link in one transaction.
    pub fn supersede(&mut self, older: EpisodeId, newer: EpisodeId) -> Result<()> {
        let closed_end;
        let tx = self.db.begin_write()?;
        {
            let mut episodes = tx.open_table(EPISODES)?;

            // Read both in scope of the same write tx so we can't
            // observe a stale `newer.valid_time.start`.
            let newer_bytes = episodes
                .get(newer.raw())?
                .ok_or(AgidbError::UnknownEpisode(newer.raw()))?
                .value();
            let newer_ep: Episode = decode(&newer_bytes)?;

            let older_bytes = episodes
                .get(older.raw())?
                .ok_or(AgidbError::UnknownEpisode(older.raw()))?
                .value();
            let mut older_ep: Episode = decode(&older_bytes)?;

            older_ep.superseded_by = Some(newer);
            closed_end = newer_ep.valid_time.start - Duration::milliseconds(1);
            older_ep.valid_time.end = Some(closed_end);

            episodes.insert(older.raw(), encode(&older_ep)?)?;
        }
        tx.commit()?;
        if let Some(&pos) = self.scan_pos.get(&older.raw()) {
            self.scan_dir[pos].valid_end = Some(closed_end);
        }
        Ok(())
    }

    /// Flush in-memory state to disk. redb commits are already durable;
    /// this just flushes the signatures mmap.
    pub fn flush(&self) -> Result<()> {
        self.signatures.flush()
    }

    /// Row counts for every table plus the on-disk signature file size.
    /// Cheap (one read transaction, four `len()` calls) and safe to call
    /// at any time.
    pub fn stats(&self) -> Result<Stats> {
        let tx = self.db.begin_read()?;
        let episodes = tx.open_table(EPISODES)?.len()?;
        let concepts = tx.open_table(CONCEPTS)?.len()?;
        let semantic_atoms = tx.open_table(SEMANTIC_ATOMS)?.len()?;
        let goals = tx.open_table(crate::goal::GOALS)?.len()?;
        let beliefs = tx.open_table(crate::belief::BELIEFS)?.len()?;
        let consolidation_passes = tx.open_table(CONSOLIDATION_LOG)?.len()?;
        Ok(Stats {
            episodes,
            concepts,
            semantic_atoms,
            goals,
            beliefs,
            consolidation_passes,
            signatures: self.signatures.len(),
        })
    }

    /// Return up to `limit` episodes in id (ascending) order. Used by the
    /// CLI `list` command for a quick "what's in the store" view.
    pub fn list_episodes(&self, limit: usize) -> Result<Vec<Episode>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(EPISODES)?;
        let mut out = Vec::with_capacity(limit.min(256));
        for entry in table.iter()? {
            if out.len() >= limit {
                break;
            }
            let (_, v) = entry?;
            out.push(decode(&v.value())?);
        }
        Ok(out)
    }

    /// Dump every Episode (with its HV) as JSON lines. Round-trips
    /// through `import_jsonl` into a fresh store.
    pub fn export_jsonl(&self, mut writer: impl Write) -> Result<()> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(EPISODES)?;
        for entry in table.iter()? {
            let (_, v) = entry?;
            let episode: Episode = decode(&v.value())?;
            let hv = self.signatures.read(episode.signature_offset)?;
            let record = ExportRecord {
                episode,
                hv: hv.0.to_vec(),
            };
            let line = serde_json::to_string(&record)
                .map_err(|e| AgidbError::Internal(format!("json encode: {e}")))?;
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Import JSON lines produced by `export_jsonl`. Returns the count
    /// of episodes imported.
    pub fn import_jsonl(&mut self, reader: impl std::io::Read) -> Result<u32> {
        let reader = BufReader::new(reader);
        let mut count = 0u32;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: ExportRecord = serde_json::from_str(&line)
                .map_err(|e| AgidbError::Internal(format!("json decode: {e}")))?;
            if record.hv.len() != D_BYTES {
                return Err(AgidbError::Internal(format!(
                    "import: expected {} hv bytes, got {}",
                    D_BYTES,
                    record.hv.len()
                )));
            }
            let mut bytes = [0u8; D_BYTES];
            bytes.copy_from_slice(&record.hv);
            let hv = HV(bytes);
            self.observe(record.episode, &hv)?;
            count += 1;
        }
        Ok(count)
    }

    /// Mint a fresh `EpisodeId`. Monotonic and persisted across reopens
    /// via the `next_episode_id` manifest counter. Used by
    /// `agidb-extract::observe_text` so callers don't have to manage ids
    /// themselves.
    pub fn next_episode_id(&mut self) -> Result<EpisodeId> {
        let tx = self.db.begin_write()?;
        let id;
        {
            let mut manifest = tx.open_table(MANIFEST)?;
            let raw = manifest.get(KEY_NEXT_EPISODE_ID)?.map(|v| v.value());
            let current: u64 = match raw {
                Some(bytes) => decode(&bytes)?,
                None => 1,
            };
            manifest.insert(KEY_NEXT_EPISODE_ID, encode(&(current + 1))?)?;
            id = EpisodeId::new(current);
        }
        tx.commit()?;
        Ok(id)
    }

    /// Case-insensitive lookup against `CONCEPT_BY_NAME`. O(N); fine for
    /// the v0.1 concept-count regime. Returns the first row whose
    /// lowercased canonical name matches `lowercased`.
    pub fn concept_id_for_ci(&self, lowercased: &str) -> Result<Option<ConceptId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(CONCEPT_BY_NAME)?;
        for row in table.iter()? {
            let (k, v) = row?;
            if k.value().to_lowercase() == lowercased {
                return Ok(Some(ConceptId::new(v.value())));
            }
        }
        Ok(None)
    }

    /// Return every ConceptId whose lowercased canonical name is within
    /// Levenshtein distance `max_dist` of `lowercased`. Skips the exact
    /// match — use [`concept_id_for_ci`] for that. O(N).
    pub fn fuzzy_concept_candidates(
        &self,
        lowercased: &str,
        max_dist: usize,
    ) -> Result<Vec<ConceptId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(CONCEPT_BY_NAME)?;
        let mut hits = Vec::new();
        for row in table.iter()? {
            let (k, v) = row?;
            let folded = k.value().to_lowercase();
            if folded == lowercased {
                continue;
            }
            if levenshtein(&folded, lowercased) <= max_dist {
                hits.push(ConceptId::new(v.value()));
            }
        }
        Ok(hits)
    }

    /// Idempotent on canonical name: if a Concept with this name already
    /// exists, return its `ConceptId` unchanged. Otherwise mint a new id,
    /// persist a Concept row with `entity_type` set, and return it.
    ///
    /// Used by `agidb-extract`'s alias resolver to pre-create concepts
    /// with the NER-derived `entity_type` before `Store::observe` would
    /// otherwise auto-create them with `entity_type = "unknown"`.
    pub fn create_concept(&mut self, canonical_name: &str, entity_type: &str) -> Result<ConceptId> {
        if let Some(existing) = self.concept_id_for(canonical_name)? {
            return Ok(existing);
        }
        let tx = self.db.begin_write()?;
        let id;
        {
            let mut concepts = tx.open_table(CONCEPTS)?;
            let mut by_name = tx.open_table(CONCEPT_BY_NAME)?;
            let mut manifest = tx.open_table(MANIFEST)?;
            id = next_concept_id(&mut manifest)?;
            let concept = Concept {
                id,
                canonical_name: canonical_name.to_string(),
                aliases: Vec::new(),
                entity_type: entity_type.to_string(),
            };
            concepts.insert(id.raw(), encode(&concept)?)?;
            by_name.insert(canonical_name, id.raw())?;
        }
        tx.commit()?;
        Ok(id)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct ExportRecord {
    episode: Episode,
    hv: Vec<u8>,
}

pub(crate) fn encode<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serialize(value).map_err(|e| AgidbError::Internal(format!("bincode encode: {e}")))
}

pub(crate) fn decode<T: for<'de> serde::Deserialize<'de>>(bytes: &[u8]) -> Result<T> {
    bincode::deserialize(bytes).map_err(|e| AgidbError::Internal(format!("bincode decode: {e}")))
}

/// Iterative Levenshtein distance (two-row variant). Used by
/// [`Store::fuzzy_concept_candidates`]; duplicated in
/// `agidb-extract::aliases` only to keep agidb-core extraction-blind in
/// the rare case a future caller needs the helper standalone.
fn levenshtein(a: &str, b: &str) -> usize {
    let (m, n) = (a.chars().count(), b.chars().count());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Read-modify-write the monotonic concept-id counter inside the
/// caller's open write transaction.
fn next_concept_id(manifest: &mut redb::Table<&str, Vec<u8>>) -> Result<ConceptId> {
    let raw = manifest.get(KEY_NEXT_CONCEPT_ID)?.map(|v| v.value());
    let current: u64 = match raw {
        Some(bytes) => decode(&bytes)?,
        None => 1,
    };
    manifest.insert(KEY_NEXT_CONCEPT_ID, encode(&(current + 1))?)?;
    Ok(ConceptId::new(current))
}
