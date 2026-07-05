//! The three systems under test. Each ingests the same corpus and
//! answers each query with a ranked id list (top 10).

use std::path::Path;
use std::time::Instant;

use agidb_core::episode::encode_episode_signature;
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{Episode, EpisodeId, Provenance, Query, TimeRange, Triple};
use anyhow::Result;
use rusqlite::Connection;

use crate::corpus::{BenchQuery, Doc};

pub const K: usize = 10;

pub trait System {
    fn name(&self) -> &'static str;
    fn ingest(&mut self, docs: &[Doc]) -> Result<()>;
    fn query(&mut self, q: &BenchQuery) -> Result<Vec<u64>>;
    fn disk_bytes(&self) -> u64;
}

// --- agidb -----------------------------------------------------------------

pub struct AgidbSystem {
    store: Store,
    root: std::path::PathBuf,
}

impl AgidbSystem {
    pub fn open(root: &Path) -> Result<Self> {
        Ok(Self {
            store: Store::open(StoreConfig::at(root))?,
            root: root.to_path_buf(),
        })
    }
}

impl System for AgidbSystem {
    fn name(&self) -> &'static str {
        "agidb"
    }

    fn ingest(&mut self, docs: &[Doc]) -> Result<()> {
        for d in docs {
            let ep_id = EpisodeId::new(d.id);
            let triples = vec![Triple {
                subject: d.person.clone(),
                predicate: d.predicate.to_string(),
                object: d.place.clone(),
                confidence: 0.9,
                episode_id: ep_id,
            }];
            let sig = encode_episode_signature(&triples, Some(d.valid_start));
            let ep = Episode {
                id: ep_id,
                text: d.text.clone(),
                signature_offset: 0,
                gist_offset: 0,
                triples,
                valid_time: TimeRange::point(d.valid_start),
                t_tx_start: d.valid_start,
                provenance: Provenance {
                    source: "bench".into(),
                    ..Provenance::default()
                },
                confidence: 0.9,
                superseded_by: None,
            };
            self.store.observe(ep, &sig)?;
        }
        for d in docs {
            if let Some(newer) = d.superseded_by {
                self.store
                    .supersede(EpisodeId::new(d.id), EpisodeId::new(newer))?;
            }
        }
        Ok(())
    }

    fn query(&mut self, q: &BenchQuery) -> Result<Vec<u64>> {
        let mut query = Query::cue(q.cue.clone()).with_k(K);
        if let Some(t) = q.as_of {
            query = query.with_as_of(t);
        }
        let r = self.store.recall(&query)?;
        Ok(r.matches.iter().map(|m| m.episode_id.raw()).collect())
    }

    fn disk_bytes(&self) -> u64 {
        dir_bytes(&self.root)
    }
}

// --- SQLite FTS5 (BM25) ------------------------------------------------------

pub struct Fts5System {
    conn: Connection,
    path: std::path::PathBuf,
}

impl Fts5System {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE docs (
                 id INTEGER PRIMARY KEY,
                 text TEXT NOT NULL,
                 valid_start INTEGER NOT NULL,
                 valid_end INTEGER
             );
             CREATE VIRTUAL TABLE fts USING fts5(text, content='docs', content_rowid='id');",
        )?;
        Ok(Self {
            conn,
            path: path.to_path_buf(),
        })
    }
}

impl System for Fts5System {
    fn name(&self) -> &'static str {
        "sqlite-fts5"
    }

    fn ingest(&mut self, docs: &[Doc]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for d in docs {
            tx.execute(
                "INSERT INTO docs (id, text, valid_start, valid_end) VALUES (?1, ?2, ?3, NULL)",
                rusqlite::params![d.id as i64, d.text, d.valid_start.timestamp()],
            )?;
            tx.execute(
                "INSERT INTO fts (rowid, text) VALUES (?1, ?2)",
                rusqlite::params![d.id as i64, d.text],
            )?;
        }
        for d in docs {
            if let Some(newer) = d.superseded_by {
                tx.execute(
                    "UPDATE docs SET valid_end =
                       (SELECT valid_start - 1 FROM docs WHERE id = ?1)
                     WHERE id = ?2",
                    rusqlite::params![newer as i64, d.id as i64],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn query(&mut self, q: &BenchQuery) -> Result<Vec<u64>> {
        let tokens: Vec<String> = q
            .cue
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| format!("\"{}\"", t.replace('"', "")))
            .collect();
        if tokens.is_empty() {
            return Ok(vec![]);
        }
        let match_expr = tokens.join(" OR ");
        if let Some(t) = q.as_of {
            let mut stmt = self.conn.prepare_cached(
                "SELECT f.rowid FROM fts f JOIN docs d ON d.id = f.rowid
                 WHERE fts MATCH ?1
                   AND d.valid_start <= ?2
                   AND (d.valid_end IS NULL OR d.valid_end >= ?2)
                 ORDER BY bm25(fts) LIMIT ?3",
            )?;
            let ids = stmt
                .query_map(
                    rusqlite::params![match_expr, t.timestamp(), K as i64],
                    |r| r.get::<_, i64>(0),
                )?
                .collect::<std::result::Result<Vec<i64>, _>>()?;
            Ok(ids.into_iter().map(|i| i as u64).collect())
        } else {
            let mut stmt = self.conn.prepare_cached(
                "SELECT rowid FROM fts WHERE fts MATCH ?1 ORDER BY bm25(fts) LIMIT ?2",
            )?;
            let ids = stmt
                .query_map(rusqlite::params![match_expr, K as i64], |r| {
                    r.get::<_, i64>(0)
                })?
                .collect::<std::result::Result<Vec<i64>, _>>()?;
            Ok(ids.into_iter().map(|i| i as u64).collect())
        }
    }

    fn disk_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }
}

// --- naive scan (the no-index floor) -----------------------------------------

pub struct ScanSystem {
    rows: Vec<(u64, String, i64, Option<i64>)>,
}

impl ScanSystem {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }
}

impl System for ScanSystem {
    fn name(&self) -> &'static str {
        "naive-scan"
    }

    fn ingest(&mut self, docs: &[Doc]) -> Result<()> {
        for d in docs {
            self.rows
                .push((d.id, d.text.to_lowercase(), d.valid_start.timestamp(), None));
        }
        let ends: Vec<(u64, i64)> = docs
            .iter()
            .filter_map(|d| {
                d.superseded_by.map(|newer| {
                    let end = docs
                        .iter()
                        .find(|x| x.id == newer)
                        .map(|x| x.valid_start.timestamp() - 1)
                        .unwrap_or(i64::MAX);
                    (d.id, end)
                })
            })
            .collect();
        for (id, end) in ends {
            if let Some(row) = self.rows.iter_mut().find(|r| r.0 == id) {
                row.3 = Some(end);
            }
        }
        Ok(())
    }

    fn query(&mut self, q: &BenchQuery) -> Result<Vec<u64>> {
        let tokens: Vec<String> = q
            .cue
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect();
        let mut scored: Vec<(usize, u64)> = self
            .rows
            .iter()
            .filter(|(_, _, start, end)| match q.as_of {
                Some(t) => {
                    let ts = t.timestamp();
                    *start <= ts && end.map(|e| e >= ts).unwrap_or(true)
                }
                None => true,
            })
            .map(|(id, text, _, _)| {
                let score = tokens.iter().filter(|t| text.contains(*t)).count();
                (score, *id)
            })
            .filter(|(s, _)| *s > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        Ok(scored.into_iter().take(K).map(|(_, id)| id).collect())
    }

    fn disk_bytes(&self) -> u64 {
        self.rows
            .iter()
            .map(|(_, t, _, _)| t.len() as u64 + 24)
            .sum()
    }
}

// --- shared -------------------------------------------------------------------

pub fn dir_bytes(root: &Path) -> u64 {
    fn walk(p: &Path, acc: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(p) {
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, acc);
                } else if let Ok(m) = e.metadata() {
                    *acc += m.len();
                }
            }
        }
    }
    let mut acc = 0;
    walk(root, &mut acc);
    acc
}

pub fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let out = f();
    (out, t0.elapsed().as_secs_f64() * 1000.0)
}
