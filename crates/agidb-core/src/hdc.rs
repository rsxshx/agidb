//! Layer 1 — the HDC kernel.
//!
//! Implements:
//! - [`HV`] — a binary hypervector of [`D`] bits / [`D_BYTES`] bytes
//! - [`HV::bind`], [`HV::bundle`], [`HV::hamming`] — the algebraic primitives
//! - AVX-512 + NEON + portable POPCOUNT paths for [`HV::hamming`]
//!
//! See `docs/architecture/layer-1-recall.md` for the math and
//! `docs/phases/phase-1-hdc-kernel.md` for the exit criterion
//! (8192-bit hamming scan over 100k signatures in <5ms on M2).

use std::hash::{Hash, Hasher};

/// Dimensionality of every hypervector, in bits.
pub const D: usize = 8192;

/// Dimensionality of every hypervector, in bytes (1024).
pub const D_BYTES: usize = D / 8;

/// Dimensionality in `u64` words (128). Used by the hot scan loops.
const D_U64: usize = D_BYTES / 8;

/// A binary hypervector. Fixed size: 8192 bits / 1024 bytes, aligned
/// to a 64-byte cache line so AVX-512 / NEON loads can use aligned
/// instructions without a runtime check.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct HV(pub [u8; D_BYTES]);

// Manual serde impls — serde's built-in array support tops out at 512
// elements, but D_BYTES = 1024. Serialize as a Vec<u8> for portability.
impl serde::Serialize for HV {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.0.to_vec(), s)
    }
}

impl<'de> serde::Deserialize<'de> for HV {
    fn deserialize<Dser: serde::Deserializer<'de>>(
        d: Dser,
    ) -> std::result::Result<Self, Dser::Error> {
        let v: Vec<u8> = serde::Deserialize::deserialize(d)?;
        if v.len() != D_BYTES {
            return Err(serde::de::Error::custom(format!(
                "expected {} hv bytes, got {}",
                D_BYTES,
                v.len()
            )));
        }
        let mut bytes = [0u8; D_BYTES];
        bytes.copy_from_slice(&v);
        Ok(HV(bytes))
    }
}

impl HV {
    /// All-zero hypervector. Useful as a bundling accumulator.
    pub const fn zero() -> Self {
        HV([0u8; D_BYTES])
    }

    /// Deterministic hypervector derived from a name. Same name always
    /// produces the same `HV`; different names produce uncorrelated
    /// vectors (in expectation).
    ///
    /// Implementation: xorshift64 expansion of the std DefaultHasher of
    /// the name. Adequate for phase 1; phase 2+ may switch to blake3 or
    /// xxhash for stronger uniformity guarantees.
    pub fn from_name(name: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        let mut seed = hasher.finish().max(1); // xorshift breaks on zero seed
        let mut bytes = [0u8; D_BYTES];
        for chunk in bytes.chunks_mut(8) {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let next = seed.to_le_bytes();
            chunk.copy_from_slice(&next[..chunk.len()]);
        }
        HV(bytes)
    }

    /// XOR-binding. Binding is its own inverse: `a.bind(&b).bind(&b) == a`.
    /// Binding is commutative: `a.bind(&b) == b.bind(&a)`.
    pub fn bind(&self, other: &HV) -> HV {
        let mut out = [0u8; D_BYTES];
        // Operate on u64 chunks — LLVM auto-vectorizes this on every target
        // that supports SSE2 / NEON, so a hand-written SIMD path adds no
        // measurable speedup at the bind site.
        let a = self.as_u64s();
        let b = other.as_u64s();
        let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u64, D_U64) };
        for ((d, &ai), &bi) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *d = ai ^ bi;
        }
        HV(out)
    }

    /// Per-bit majority bundling. `bundle([a, b, c])` is the hypervector
    /// where each bit is `1` iff strictly more than half of the input HVs
    /// have that bit set. Bundle membership: any member's similarity to
    /// the bundle is significantly above 0.5 for small N.
    ///
    /// **Tie-breaking on even N:** strict majority — a 2-of-4 split
    /// resolves to 0, not 1. Locked in by the property tests so future
    /// changes don't drift silently.
    ///
    /// Singleton: `bundle([a]) == a`. Empty input: returns `HV::zero()`.
    pub fn bundle(hvs: &[HV]) -> HV {
        match hvs.len() {
            0 => return HV::zero(),
            1 => return hvs[0],
            _ => {}
        }
        let n = hvs.len();

        // Tally each of the D=8192 bit positions.
        let mut tallies = [0u32; D];
        for hv in hvs {
            for (byte_idx, &b) in hv.0.iter().enumerate() {
                let base = byte_idx * 8;
                // Unrolled per-bit increment — the inner loop body is
                // small enough that LLVM keeps it in registers.
                for bit in 0..8u32 {
                    tallies[base + bit as usize] += ((b >> bit) & 1) as u32;
                }
            }
        }

        // Output bit is 1 iff tally * 2 > n (strict majority; equivalent
        // to tally > n/2 but avoids integer-division truncation).
        let mut out = [0u8; D_BYTES];
        let n_u32 = n as u32;
        for (byte_idx, byte_out) in out.iter_mut().enumerate() {
            let mut byte = 0u8;
            let base = byte_idx * 8;
            for bit in 0..8u32 {
                if tallies[base + bit as usize] * 2 > n_u32 {
                    byte |= 1 << bit;
                }
            }
            *byte_out = byte;
        }
        HV(out)
    }

    /// Hamming distance — the number of bit positions where two HVs
    /// differ. Range: `0..=D`.
    ///
    /// Dispatches to the fastest available POPCOUNT implementation:
    /// - AVX-512 `vpopcntdq` when runtime-detected on x86_64
    /// - NEON `vcntq_u8` on aarch64 (always present in the base ISA)
    /// - Portable `u64::count_ones` over 128 chunks otherwise
    #[inline]
    pub fn hamming(&self, other: &HV) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
                // SAFETY: feature presence verified above.
                return unsafe { hamming_avx512(self, other) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is part of the aarch64 base ISA — always present.
            return unsafe { hamming_neon(self, other) };
        }
        #[allow(unreachable_code)]
        {
            hamming_portable(self, other)
        }
    }

    /// Cosine-like similarity score in `[0.0, 1.0]`, derived from
    /// hamming. Defined in terms of `hamming` so phase 1 only needs to
    /// implement the kernel once.
    pub fn similarity(&self, other: &HV) -> f32 {
        1.0 - (self.hamming(other) as f32 / D as f32)
    }

    /// Number of set bits. Cached by the store's scan directory so the
    /// phi scoring path doesn't recount per query.
    pub fn popcount(&self) -> u32 {
        self.as_u64s().iter().map(|x| x.count_ones()).sum()
    }

    /// Indices of every set bit, in ascending order. Used by the
    /// inverted-index update path in phase 2.
    pub fn active_dims(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().enumerate().flat_map(|(byte_idx, &b)| {
            (0..8u32)
                .filter(move |bit| (b >> bit) & 1 == 1)
                .map(move |bit| byte_idx as u32 * 8 + bit)
        })
    }

    /// Reinterpret the underlying byte buffer as a `&[u64; D_U64]`.
    ///
    /// `HV` is `#[repr(C, align(64))]` and `D_BYTES` is a multiple of 8,
    /// so the cast is always sound. Pulled out for the hot kernels that
    /// want word-sized access.
    #[inline]
    fn as_u64s(&self) -> &[u64] {
        // SAFETY: 64-byte aligned, length 1024 = 128 * 8, no aliasing.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u64, D_U64) }
    }
}

impl PartialEq for HV {
    fn eq(&self, other: &Self) -> bool {
        self.0[..] == other.0[..]
    }
}

impl Eq for HV {}

impl std::fmt::Debug for HV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Show the first 4 bytes as hex; full 1024 bytes are too noisy.
        write!(f, "HV(")?;
        for byte in &self.0[..4] {
            write!(f, "{:02x}", byte)?;
        }
        write!(f, "..)")
    }
}

// ---------------------------------------------------------------------------
// hamming() backends — portable + AVX-512 + NEON
// ---------------------------------------------------------------------------

/// Portable hamming: 128 × (XOR + count_ones) over u64 words.
///
/// Compiles to a tight loop using the hardware POPCNT instruction on any
/// x86_64 chip since Nehalem (2008) and any aarch64 chip via the FEAT_CSSC
/// extension or a NEON lowering. Hits ~50ns per scan on a Zen 4 / M2 even
/// without explicit SIMD because LLVM auto-vectorizes count_ones with AVX2
/// / NEON `cnt` instructions.
#[inline]
fn hamming_portable(a: &HV, b: &HV) -> u32 {
    a.as_u64s()
        .iter()
        .zip(b.as_u64s().iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// AVX-512 hamming using `vpopcntdq` (per-u64 popcount in a single op).
///
/// Processes 64 bytes per iteration → 16 iterations for an 8192-bit HV.
/// Each iteration is one aligned load × 2, one `xor`, one `vpopcntdq`,
/// one `add` — about 5 µops, < 5 ns per scan on Zen 4 / Sapphire Rapids.
///
/// SAFETY: caller must verify `avx512f` and `avx512vpopcntdq` are present
/// (the public `hamming` does so before dispatching here).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn hamming_avx512(a: &HV, b: &HV) -> u32 {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_si512();
    // 1024 bytes / 64 bytes per __m512i = 16 chunks, all aligned to 64.
    let ap = a.0.as_ptr() as *const __m512i;
    let bp = b.0.as_ptr() as *const __m512i;
    for i in 0..16 {
        let av = _mm512_load_si512(ap.add(i));
        let bv = _mm512_load_si512(bp.add(i));
        let xor = _mm512_xor_si512(av, bv);
        let pc = _mm512_popcnt_epi64(xor);
        acc = _mm512_add_epi64(acc, pc);
    }
    _mm512_reduce_add_epi64(acc) as u32
}

/// NEON hamming using `vcntq_u8` (per-byte popcount) + pairwise long add.
///
/// Processes 16 bytes per iteration → 64 iterations for an 8192-bit HV.
/// Accumulates into u16 lanes via `vpaddlq_u8` to avoid u8 overflow
/// (each u8 lane could reach 8 × 64 = 512 across the scan).
///
/// SAFETY: NEON is mandatory on aarch64; no runtime check needed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hamming_neon(a: &HV, b: &HV) -> u32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_u16(0); // 8 × u16
    let ap = a.0.as_ptr();
    let bp = b.0.as_ptr();
    for i in 0..64 {
        let av = vld1q_u8(ap.add(i * 16));
        let bv = vld1q_u8(bp.add(i * 16));
        let xor = veorq_u8(av, bv);
        let pc = vcntq_u8(xor); // 16 × u8 popcounts, each 0..=8
                                // Widen + pair-sum into u16 lanes to avoid overflow.
        let pc16 = vpaddlq_u8(pc); // 8 × u16 from pair-summed u8s
        acc = vaddq_u16(acc, pc16);
    }
    // Horizontal sum across the 8 u16 lanes.
    vaddvq_u16(acc) as u32
}

/// Phi coefficient (Pearson correlation over binary vectors) from
/// pre-computed counts. Density-corrected: independent vectors score
/// ≈ 0 regardless of how sparse either side is, which raw hamming
/// similarity does not guarantee (majority-bundling an even number of
/// HVs yields AND-like ~25%-dense vectors that inflate raw agreement).
///
/// `n` is the dimensionality (use `D as f64`), `pa`/`pb` the popcounts
/// of the two vectors, `hamming` their hamming distance. Returns 0.0
/// for degenerate inputs (all-zero / all-one vectors).
pub fn phi_from_counts(n: f64, pa: f64, pb: f64, hamming: f64) -> f32 {
    if pa <= 0.0 || pa >= n || pb <= 0.0 || pb >= n {
        return 0.0;
    }
    let n11 = (pa + pb - hamming) / 2.0;
    let num = n * n11 - pa * pb;
    let den = (pa * pb * (n - pa) * (n - pb)).sqrt();
    (num / den) as f32
}
