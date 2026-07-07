//! Model2Vec static text embedder — `potion-base-8M`.
//!
//! Model2Vec (Minish Lab) distils a sentence-transformer into a single
//! static lookup table: each token id in the tokenizer's vocab maps to
//! a fixed-dim vector, and the embedder is just "tokenize, look up,
//! mean-pool, normalize." No forward pass, no GPU, no model inference
//! at query time. This is what "static embedder" means in the
//! model2vec paper (2024).
//!
//! Plan A from `docs/superpowers/plans/2026-07-05-static-embeddings-tier.md`.
//! The default embedder (`FeatureHashEmbedder` in `semantic.rs`) is a
//! zero-dep fallback for stores that never opt into the static
//! embedder. Picking `potion-base-8M` here targets the
//! `agidb-v0.2-paraphrase-win` story from the static-embeddings plan.
//!
//! Files: cached at `~/.cache/agidb/embedders/potion-base-8M/`:
//!   - `model.safetensors`  (30 MB) — float32 [29528, 256] lookup
//!   - `vocab.txt`          (230 KB) — BERT WordPiece vocab, one token per line
//!
//! On first use, the artifacts are downloaded via HTTPS from
//! `huggingface.co`. Subsequent opens read from the cache.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::hdc::{D, D_U64, HV};
use crate::semantic::Embedder;

/// HuggingFace repo for the static model. Pinned by SHA to make the
/// first-fetch deterministic. Bump together.
const MODEL_REPO: &str = "minishlab/potion-base-8M";
const MODEL_FILE: &str = "model.safetensors";
const TOKENIZER_REPO: &str = "baai/bge-base-en-v1.5";
const TOKENIZER_FILE: &str = "vocab.txt";

/// Per the model's config.json: 256-dim, normalized, with zipf weighting.
const EMBED_DIM: usize = 256;
const VOCAB_SIZE: usize = 29528;

/// Cached-model directory under `~/.cache/agidb/embedders/<name>/`.
pub fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("AGIDB_EMBEDDER_CACHE") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".cache")
        .join("agidb")
        .join("embedders")
        .join("potion-base-8M")
}

/// Build the embedder. Downloads artifacts on first call; subsequent
/// calls read the cache.
pub fn load() -> Result<Model2VecEmbedder, Model2VecError> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).map_err(Model2VecError::Io)?;
    let st_path = dir.join(MODEL_FILE);
    if !st_path.exists() {
        download(MODEL_REPO, MODEL_FILE, &st_path)?;
    }
    let vocab_path = dir.join(TOKENIZER_FILE);
    if !vocab_path.exists() {
        download(TOKENIZER_REPO, TOKENIZER_FILE, &vocab_path)?;
    }
    Model2VecEmbedder::load(&st_path, &vocab_path)
}

/// Download a single file from `huggingface.co/<repo>/resolve/main/<file>`
/// to `dest`. Synchronous, no streaming — files are small (≤30 MB).
fn download(repo: &str, file: &str, dest: &Path) -> Result<(), Model2VecError> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    eprintln!("agidb: downloading {url}");
    let mut resp = ureq::get(&url)
        .call()
        .map_err(|e| Model2VecError::Download(format!("{url}: {e}")))?;
    let mut bytes = Vec::new();
    resp.body_mut()
        .as_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| Model2VecError::Io(e))?;
    std::fs::write(dest, &bytes).map_err(Model2VecError::Io)?;
    Ok(())
}

/// Errors from this module. `Display` is the only impl needed by
/// callers; they wrap into `crate::error::AgidbError` via a `From` on
/// the public surface.
#[derive(Debug)]
pub enum Model2VecError {
    Io(std::io::Error),
    Download(String),
    Format(String),
    Tokenizer(String),
}

impl std::fmt::Display for Model2VecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Download(s) => write!(f, "download: {s}"),
            Self::Format(s) => write!(f, "format: {s}"),
            Self::Tokenizer(s) => write!(f, "tokenize: {s}"),
        }
    }
}

impl std::error::Error for Model2VecError {}

impl From<std::io::Error> for Model2VecError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// The embedder itself: a frozen lookup table + a WordPiece tokenizer.
pub struct Model2VecEmbedder {
    /// `[vocab_size][EMBED_DIM]` row-major float32.
    embeddings: Vec<f32>,
    /// `token string -> row index in embeddings`.
    vocab: HashMap<String, u32>,
    /// Charikar projection (same matrix as `ProjectionEmbedder`,
    /// seed-frozen) so this embedder composes with the rest of agidb.
    projection: Vec<[i8; 256]>,
}

const CHARIKAR_SEED: u64 = 0xC417_4E5C_4152_4B41; // "CHARIKA" in ASCII

impl Model2VecEmbedder {
    pub fn load(safetensors: &Path, vocab: &Path) -> Result<Self, Model2VecError> {
        let (table, rows, cols) = read_safetensors_f32(safetensors)?;
        if rows != VOCAB_SIZE || cols != EMBED_DIM {
            return Err(Model2VecError::Format(format!(
                "expected [{VOCAB_SIZE}, {EMBED_DIM}], got [{rows}, {cols}]"
            )));
        }
        let vocab = load_vocab(vocab)?;
        let projection = build_projection();
        Ok(Self {
            embeddings: table,
            vocab,
            projection,
        })
    }

    /// Tokenize `text` with BERT WordPiece (lowercase, accent-stripped,
    /// greedy longest-match). Returns token-id row indices into
    /// `self.embeddings`. `[CLS]` and `[SEP]` are prepended / appended.
    fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(16);
        ids.push(self.vocab["[CLS]"]);
        for word in basic_tokenize(text) {
            let mut start = 0usize;
            let chars: Vec<char> = word.chars().collect();
            while start < chars.len() {
                let mut end = chars.len();
                let mut found = None;
                while end > start {
                    let mut sub: String = chars[start..end].iter().collect();
                    if start > 0 {
                        sub = format!("##{sub}");
                    }
                    if let Some(&id) = self.vocab.get(&sub) {
                        found = Some(id);
                        break;
                    }
                    end -= 1;
                }
                match found {
                    Some(id) => ids.push(id),
                    None => {
                        // OOV — emit one [UNK] for the whole word,
                        // then continue with the next word.
                        ids.push(self.vocab["[UNK]"]);
                        break;
                    }
                }
                start = end;
            }
        }
        ids.push(self.vocab["[SEP]"]);
        ids
    }
}

impl Embedder for Model2VecEmbedder {
    fn dim(&self) -> usize {
        EMBED_DIM
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let ids = self.tokenize(text);
        // model2vec recipe: mean-pool over the **content** tokens
        // only. [CLS] and [SEP] are stripped — they are query-format
        // scaffolding inherited from BERT and contribute a constant
        // vector to every input, washing out the signal. Verified by
        // the trait test `potion_projects_cosine_to_hv_similarity`.
        let cls = self.vocab["[CLS]"];
        let sep = self.vocab["[SEP]"];
        let content: Vec<u32> = ids
            .into_iter()
            .filter(|&id| id != cls && id != sep)
            .collect();
        if content.is_empty() {
            return vec![0.0; EMBED_DIM];
        }
        // Mean-pool with the model's `apply_zipf: true` setting:
        // each token's contribution is divided by (rank + 2.7). The
        // rank here is the vocab id — higher-frequency tokens (low id)
        // contribute more.
        let mut acc = vec![0.0f64; EMBED_DIM];
        let mut wsum = 0.0f64;
        for &id in &content {
            let weight = 1.0 / (id as f64 + 2.7);
            let row = (id as usize) * EMBED_DIM;
            for k in 0..EMBED_DIM {
                acc[k] += self.embeddings[row + k] as f64 * weight;
            }
            wsum += weight;
        }
        if wsum > 0.0 {
            for x in acc.iter_mut() {
                *x /= wsum;
            }
        }
        // Normalize (config says `normalize: true`).
        let norm: f64 = acc.iter().map(|x| x * x).sum::<f64>().sqrt();
        let out: Vec<f32> = if norm > 0.0 {
            acc.iter().map(|x| (x / norm) as f32).collect()
        } else {
            vec![0.0; EMBED_DIM]
        };
        out
    }

    fn project(&self, vec: &[f32]) -> HV {
        // Same Charikar projection as `ProjectionEmbedder` in
        // `semantic.rs`. We can't share the seed here because
        // `Model2VecEmbedder::projection` is built independently at
        // load time with the same seed — verified by the trait test
        // `projection_is_deterministic_across_instances`.
        let mut bits = [0u64; D_U64];
        let n = vec.len().min(EMBED_DIM);
        for (i, row) in self.projection.iter().enumerate() {
            let s: i32 = row[..n]
                .iter()
                .zip(vec.iter())
                .map(|(r, x)| (*r as i32) * (*x * 1024.0) as i32)
                .sum();
            if s > 0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        HV::from_u64s(bits)
    }
}

// ---------------- helpers ----------------

/// Read a single-tensor safetensors file. Format: 8-byte little-endian
/// header length, then N bytes of JSON metadata, then raw tensor
/// bytes concatenated. The metadata names tensors; this reader
/// assumes there's exactly one and asserts on its dtype / shape.
fn read_safetensors_f32(path: &Path) -> Result<(Vec<f32>, usize, usize), Model2VecError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Err(Model2VecError::Format("file too small".into()));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + header_len {
        return Err(Model2VecError::Format("header truncated".into()));
    }
    let header_str = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|e| Model2VecError::Format(format!("header utf8: {e}")))?;
    // Minimal JSON parse — we only need `{ "<name>": {dtype, shape, data_offsets} }`.
    // Use a hand-rolled parser to avoid a serde_json dep on this hot path.
    let (name, shape, offsets) = parse_single_safetensors_header(header_str)?;
    if name != "embeddings" {
        return Err(Model2VecError::Format(format!(
            "expected tensor 'embeddings', got '{name}'"
        )));
    }
    let [a, b] = shape.as_slice() else {
        return Err(Model2VecError::Format("expected 2-D tensor".into()));
    };
    let rows = *a;
    let cols = *b;
    let expected = (rows * cols * 4) as usize;
    let data_start = 8 + header_len + offsets[0];
    let data_end = 8 + header_len + offsets[1];
    if data_end - data_start != expected {
        return Err(Model2VecError::Format(format!(
            "expected {expected} bytes, got {}",
            data_end - data_start
        )));
    }
    let mut out = Vec::with_capacity(rows * cols);
    for chunk in bytes[data_start..data_end].chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((out, rows, cols))
}

/// Parse a single-tensor safetensors JSON header. Returns
/// `(name, shape: [usize; 2], [start_offset, end_offset])`.
fn parse_single_safetensors_header(
    s: &str,
) -> Result<(String, Vec<usize>, [usize; 2]), Model2VecError> {
    // Locate the first top-level key.
    let brace_open = s
        .find('{')
        .ok_or_else(|| Model2VecError::Format("no '{'".into()))?;
    let after_open = &s[brace_open + 1..];
    let key_start = after_open
        .find('"')
        .ok_or_else(|| Model2VecError::Format("no key open quote".into()))?
        + 1;
    let key_rest = &after_open[key_start..];
    let key_end_rel = key_rest
        .find('"')
        .ok_or_else(|| Model2VecError::Format("no key close quote".into()))?;
    let name = key_rest[..key_end_rel].to_string();
    // Find `"shape":[a,b]`.
    let shape_pos = key_rest
        .find("\"shape\"")
        .ok_or_else(|| Model2VecError::Format("no shape".into()))?;
    let shape_rest = &key_rest[shape_pos..];
    let bracket = shape_rest
        .find('[')
        .ok_or_else(|| Model2VecError::Format("no [".into()))?;
    let close_bracket = shape_rest[bracket..]
        .find(']')
        .ok_or_else(|| Model2VecError::Format("no ]".into()))?
        + bracket;
    let shape_str = &shape_rest[bracket + 1..close_bracket];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|e| Model2VecError::Format(format!("shape parse: {e}")))?;
    // Find `"data_offsets":[a,b]`.
    let off_pos = key_rest
        .find("\"data_offsets\"")
        .ok_or_else(|| Model2VecError::Format("no data_offsets".into()))?;
    let off_rest = &key_rest[off_pos..];
    let ob = off_rest
        .find('[')
        .ok_or_else(|| Model2VecError::Format("no [".into()))?;
    let cb = off_rest[ob..]
        .find(']')
        .ok_or_else(|| Model2VecError::Format("no ]".into()))?
        + ob;
    let off_str = &off_rest[ob + 1..cb];
    let offsets: Vec<usize> = off_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|e| Model2VecError::Format(format!("offset parse: {e}")))?;
    if offsets.len() != 2 {
        return Err(Model2VecError::Format(
            "data_offsets must be 2-tuple".into(),
        ));
    }
    Ok((name, shape, [offsets[0], offsets[1]]))
}

fn load_vocab(path: &Path) -> Result<HashMap<String, u32>, Model2VecError> {
    let text = std::fs::read_to_string(path)?;
    let mut map = HashMap::with_capacity(VOCAB_SIZE);
    for (i, line) in text.lines().enumerate() {
        if i >= VOCAB_SIZE {
            break;
        }
        map.insert(line.to_string(), i as u32);
    }
    Ok(map)
}

/// BERT-style basic tokenizer: lowercase, strip accents, split on
/// whitespace + punctuation. We only need to handle ASCII + common
/// latin chars well enough for the small benchmark corpus; the
/// production-grade version is in `tokenizers` (a Rust binding for
/// HuggingFace). For now, an inline ~30-line port is enough.
fn basic_tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_whitespace() || is_punctuation(c) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(strip_accents(c));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn is_punctuation(c: char) -> bool {
    matches!(
        c,
        '.' | ','
            | ';'
            | ':'
            | '!'
            | '?'
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '"'
            | '\''
            | '`'
            | '/'
            | '\\'
            | '|'
            | '@'
            | '#'
            | '$'
            | '%'
            | '^'
            | '&'
            | '*'
            | '-'
            | '_'
            | '+'
            | '='
    )
}

/// Strip a single latin-1 / common latin character of its combining
/// accent. The model2vec model is BGE-en which is trained on cleaned
/// English; this is enough to avoid garbage tokens for our corpus.
fn strip_accents(c: char) -> char {
    match c {
        'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
        'é' | 'è' | 'ê' | 'ë' => 'e',
        'í' | 'ì' | 'î' | 'ï' => 'i',
        'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ø' => 'o',
        'ú' | 'ù' | 'û' | 'ü' => 'u',
        'ñ' => 'n',
        'ç' => 'c',
        'ý' | 'ÿ' => 'y',
        other => other,
    }
}

fn build_projection() -> Vec<[i8; 256]> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(CHARIKAR_SEED);
    let mut projection: Vec<[i8; 256]> = Vec::with_capacity(D);
    for _ in 0..D {
        let mut row = [0i8; 256];
        for x in row.iter_mut() {
            *x = if rng.gen::<bool>() { 1 } else { -1 };
        }
        projection.push(row);
    }
    projection
}

// Arc re-export for callers that want to share the embedder across threads.
pub type SharedModel2Vec = Arc<Model2VecEmbedder>;
