//! Shared helpers for the gallery examples: a tiny deterministic RNG so every
//! example is fully reproducible without an external dependency, plus the
//! output directory for rendered plots.
//!
//! Each `bin/*.rs` pulls this in with
//! `#[path = "../common.rs"] mod common;`.

#![allow(dead_code)]

use std::path::PathBuf;

/// A minimal SplitMix64 / xorshift-style PRNG. Deterministic and dependency
/// free — good enough to synthesize illustrative noise for the gallery.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. The same seed always yields the same stream.
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Next raw 64-bit value (SplitMix64).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        // Take the top 53 bits for a full-precision double in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal deviate via the Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A Poisson draw with mean `lambda` (Knuth's algorithm; fine for small means).
    pub fn poisson(&mut self, lambda: f64) -> f64 {
        let l = (-lambda).exp();
        let mut k = 0.0;
        let mut p = 1.0;
        loop {
            p *= self.uniform();
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }

    /// A Bernoulli draw with success probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> f64 {
        if self.uniform() < p {
            1.0
        } else {
            0.0
        }
    }

    /// An Exponential draw with rate `lambda` (mean `1/lambda`).
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        -self.uniform().max(1e-12).ln() / lambda
    }
}

/// Absolute path to `docs/book/src/examples/img`, where every example writes
/// its plot. Resolved relative to this crate's manifest so it works no matter
/// what the current working directory is.
pub fn img_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/solow-gallery
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent() // crates/
        .and_then(|p| p.parent()) // repo root
        .map(|root| root.join("docs/book/src/examples/img"))
        .expect("repo layout: crates/solow-gallery -> repo root")
}

/// Join `img_dir()` with `name`, creating the directory if needed.
pub fn img_path(name: &str) -> PathBuf {
    let dir = img_dir();
    let _ = std::fs::create_dir_all(&dir);
    dir.join(name)
}
