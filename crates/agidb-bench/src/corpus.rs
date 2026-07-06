//! Deterministic synthetic corpus + query set. Seeded splitmix64 —
//! same seed, same corpus, byte for byte, forever.

use chrono::{DateTime, Duration, TimeZone, Utc};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn pick<'a, T>(&mut self, s: &'a [T]) -> &'a T {
        &s[self.below(s.len())]
    }
}

pub const PEOPLE: [&str; 40] = [
    "Sarah", "Marco", "Priya", "Ankit", "Dev", "Alice", "Bob", "Carol", "Dan", "Eve", "Farid",
    "Grace", "Hana", "Ivan", "Julia", "Kenji", "Lena", "Miguel", "Nadia", "Omar", "Pilar", "Quinn",
    "Ravi", "Sofia", "Tarun", "Uma", "Viktor", "Wei", "Ximena", "Yuki", "Zainab", "Arjun",
    "Bianca", "Chetan", "Daria", "Emil", "Fatima", "Gustav", "Helga", "Iris",
];

pub const PLACES: [&str; 40] = [
    "Bawri",
    "Trishna",
    "Olive",
    "Pali",
    "Mahesh",
    "Soam",
    "Britannia",
    "Gajalee",
    "Dakshin",
    "Yauatcha",
    "Masque",
    "Bombay Canteen",
    "Kissa",
    "Subko",
    "Blue Tokai",
    "Araku",
    "Naru",
    "Izumi",
    "Gymkhana",
    "Dishoom",
    "Hoppers",
    "Brat",
    "Noma",
    "Ikoyi",
    "Septime",
    "Attica",
    "Quintonil",
    "Maido",
    "Narisawa",
    "Odette",
    "Alchemist",
    "Hisa",
    "Franceschetta",
    "Etxebarri",
    "Diverxo",
    "Steirereck",
    "Frantzen",
    "Disfrutar",
    "Trivet",
    "Ariana",
];

/// (canonical predicate, surface verbs)
pub const PREDICATES: [(&str, &[&str]); 4] = [
    ("recommends", &["recommended", "suggested", "pitched"]),
    ("likes", &["likes", "loves", "enjoys"]),
    ("works_at", &["works at", "is employed by"]),
    ("located_in", &["is located in", "is based in"]),
];

#[derive(Clone, Debug)]
pub struct Doc {
    pub id: u64,
    pub text: String,
    pub person: String,
    pub place: String,
    pub predicate: &'static str,
    pub valid_start: DateTime<Utc>,
    /// Set on the older half of a supersession pair.
    pub superseded_by: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum QueryClass {
    Exact,
    SingleEntity,
    Noisy,
    Temporal,
    /// Paraphrase of an exact-class cue — same relevant set, cue has
    /// zero token overlap with the stored sentences. Tests the
    /// semantic-tier fallback (Charikar-projected static-text
    /// embedding); FTS5 / naive-scan have no equivalent.
    Paraphrase,
}

/// Paraphrase templates keyed on the canonical predicate. Each
/// template is a semantic restatement of a stored factual sentence
/// ("{person} {recommended} {place}") that shares no token overlap
/// with the source. Designed for a static-text embedder.
const PARAPHRASE_TEMPLATES: &[(&str, &[&str])] = &[
    (
        "recommends",
        &[
            "good {p} place suggestion",
            "any recommendation for {p}",
            "where should we go for {p}",
            "what are some {p} suggestions",
        ],
    ),
    (
        "likes",
        &[
            "things {p} is into",
            "any {p} favorites",
            "what does {p} enjoy",
        ],
    ),
    (
        "works_at",
        &["where does {p} work", "{p}'s employer", "who employs {p}"],
    ),
    (
        "located_in",
        &["where is {p} based", "{p}'s home city", "{p}'s location"],
    ),
];

#[derive(Clone, Debug)]
pub struct BenchQuery {
    pub class: QueryClass,
    pub cue: String,
    pub as_of: Option<DateTime<Utc>>,
    /// Any of these ids in the top-k counts as a hit.
    pub relevant: Vec<u64>,
}

pub fn epoch() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
}

pub fn build_corpus(n: usize, rng: &mut Rng) -> Vec<Doc> {
    let mut docs = Vec::with_capacity(n);
    for id in 1..=(n as u64) {
        let person = rng.pick(&PEOPLE).to_string();
        let place = rng.pick(&PLACES).to_string();
        let (canonical, surfaces) = rng.pick(&PREDICATES);
        let surface = rng.pick(surfaces);
        let day = rng.below(300) as i64;
        let valid_start = epoch() + Duration::days(day);
        docs.push(Doc {
            id,
            text: format!("{person} {surface} {place} on day {day}"),
            person,
            place,
            predicate: canonical,
            valid_start,
            superseded_by: None,
        });
    }
    // Supersession pairs: for every 100th doc, append a newer doc that
    // supersedes it (same person+place, different verb, +30 days).
    let n_pairs = n / 100;
    for k in 0..n_pairs {
        let old_idx = k * 100;
        let old = docs[old_idx].clone();
        let new_id = (docs.len() + 1) as u64;
        let (canonical, surfaces) = PREDICATES[(k + 1) % PREDICATES.len()];
        let surface = surfaces[0];
        let valid_start = old.valid_start + Duration::days(30);
        docs.push(Doc {
            id: new_id,
            text: format!("{} {surface} {} after reconsidering", old.person, old.place),
            person: old.person.clone(),
            place: old.place.clone(),
            predicate: canonical,
            valid_start,
            superseded_by: None,
        });
        docs[old_idx].superseded_by = Some(new_id);
    }
    docs
}

fn typo_drop(s: &str, rng: &mut Rng) -> String {
    if s.len() < 3 {
        return s.to_string();
    }
    let pos = 1 + rng.below(s.len() - 2);
    let mut out = String::with_capacity(s.len());
    for (i, c) in s.chars().enumerate() {
        if i != pos {
            out.push(c);
        }
    }
    out
}

pub fn build_queries(docs: &[Doc], per_class: usize, rng: &mut Rng) -> Vec<BenchQuery> {
    let mut queries = Vec::new();
    let plain: Vec<&Doc> = docs.iter().filter(|d| d.superseded_by.is_none()).collect();

    let relevant_both = |person: &str, place: &str| -> Vec<u64> {
        docs.iter()
            .filter(|d| d.person == person && d.place == place)
            .map(|d| d.id)
            .collect()
    };

    for _ in 0..per_class {
        let d = *rng.pick(&plain);
        queries.push(BenchQuery {
            class: QueryClass::Exact,
            cue: format!("what did {} think of {}", d.person, d.place),
            as_of: None,
            relevant: relevant_both(&d.person, &d.place),
        });
    }
    for _ in 0..per_class {
        let d = *rng.pick(&plain);
        queries.push(BenchQuery {
            class: QueryClass::SingleEntity,
            cue: format!("did {} recommend anything", d.person),
            as_of: None,
            relevant: docs
                .iter()
                .filter(|x| x.person == d.person)
                .map(|x| x.id)
                .collect(),
        });
    }
    for _ in 0..per_class {
        let d = *rng.pick(&plain);
        let person = typo_drop(&d.person, rng);
        let place = typo_drop(&d.place, rng);
        queries.push(BenchQuery {
            class: QueryClass::Noisy,
            cue: format!("what did {person} think of {place}"),
            as_of: None,
            relevant: relevant_both(&d.person, &d.place),
        });
    }
    // Temporal: query each supersession pair before the new fact.
    let pairs: Vec<&Doc> = docs.iter().filter(|d| d.superseded_by.is_some()).collect();
    for i in 0..per_class {
        let old = pairs[i % pairs.len()];
        queries.push(BenchQuery {
            class: QueryClass::Temporal,
            cue: format!("what did {} think of {}", old.person, old.place),
            as_of: Some(old.valid_start + Duration::days(1)),
            relevant: vec![old.id],
        });
    }
    // Paraphrase: same relevant set as Exact, but the cue is a
    // paraphrase drawn from a templated list keyed on the canonical
    // predicate. The paraphrase has zero token overlap with the
    // stored sentences; only a semantic-tier system recovers them.
    let exact_relevant = |person: &str, place: &str| -> Vec<u64> {
        docs.iter()
            .filter(|d| d.person == person && d.place == place)
            .map(|d| d.id)
            .collect()
    };
    let mut par_rng = Rng::new(rng.next()); // deterministic sub-seed
    for _ in 0..per_class {
        let d = *rng.pick(&plain);
        let templates = PARAPHRASE_TEMPLATES
            .iter()
            .find(|(p, _)| *p == d.predicate)
            .map(|(_, t)| *t)
            .unwrap_or(&["what about {p}'s take on {q}"]);
        let template = par_rng.pick(templates);
        let cue = template.replace("{p}", &d.place).replace("{q}", &d.person);
        queries.push(BenchQuery {
            class: QueryClass::Paraphrase,
            cue,
            as_of: None,
            relevant: exact_relevant(&d.person, &d.place),
        });
    }
    queries
}
