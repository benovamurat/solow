//! Block bootstrap for time-series data (Künsch 1989, Politis-Romano 1994).
//!
//! * [`moving_block_bootstrap_indices`] — Künsch's moving-block scheme.
//! * [`circular_block_bootstrap_indices`] — Politis-Romano's circular
//!   variant that avoids the boundary bias.
//! * [`stationary_bootstrap_indices`] — Politis-Romano stationary
//!   bootstrap with geometric block lengths.

use solow_core::{Error, Result};

/// Moving-block bootstrap indices — a resample of length `n` built
/// from `⌈n / block_len⌉` overlapping blocks of length `block_len`.
pub fn moving_block_bootstrap_indices(
    n: usize,
    block_len: usize,
    seed: u64,
) -> Result<Vec<usize>> {
    if n == 0 || block_len == 0 || block_len > n {
        return Err(Error::Value(
            "moving_block_bootstrap_indices: 0 < block_len ≤ n required".into(),
        ));
    }
    let mut state = seed.wrapping_add(0xC0DE_F00D);
    let n_blocks = (n + block_len - 1) / block_len;
    let max_start = n - block_len;
    let mut out = Vec::with_capacity(n_blocks * block_len);
    for _ in 0..n_blocks {
        let start = uniform_index(&mut state, max_start + 1);
        for i in 0..block_len {
            out.push(start + i);
        }
    }
    out.truncate(n);
    Ok(out)
}

/// Circular block bootstrap — wraps blocks around the end of the sample.
pub fn circular_block_bootstrap_indices(
    n: usize,
    block_len: usize,
    seed: u64,
) -> Result<Vec<usize>> {
    if n == 0 || block_len == 0 {
        return Err(Error::Value(
            "circular_block_bootstrap_indices: n, block_len must be > 0".into(),
        ));
    }
    let mut state = seed.wrapping_add(0xF00D_C0DE);
    let n_blocks = (n + block_len - 1) / block_len;
    let mut out = Vec::with_capacity(n_blocks * block_len);
    for _ in 0..n_blocks {
        let start = uniform_index(&mut state, n);
        for i in 0..block_len {
            out.push((start + i) % n);
        }
    }
    out.truncate(n);
    Ok(out)
}

/// Stationary bootstrap — Politis-Romano (1994) — block lengths drawn
/// from `Geom(p)` where `p = 1 / expected_block_len`.
pub fn stationary_bootstrap_indices(
    n: usize,
    expected_block_len: f64,
    seed: u64,
) -> Result<Vec<usize>> {
    if n == 0 {
        return Err(Error::Value("stationary_bootstrap_indices: n must be > 0".into()));
    }
    if expected_block_len < 1.0 {
        return Err(Error::Value(
            "stationary_bootstrap_indices: expected_block_len must be ≥ 1".into(),
        ));
    }
    let p = 1.0 / expected_block_len;
    let mut state = seed.wrapping_add(0xBEEF_D00D);
    let mut out = Vec::with_capacity(n);
    let mut cur = uniform_index(&mut state, n);
    out.push(cur);
    while out.len() < n {
        let u = uniform01(&mut state);
        if u < p {
            cur = uniform_index(&mut state, n);
        } else {
            cur = (cur + 1) % n;
        }
        out.push(cur);
    }
    Ok(out)
}

fn uniform_index(state: &mut u64, n: usize) -> usize {
    let nu = n as u64;
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let max = u64::MAX - (u64::MAX % nu);
    if *state < max {
        (*state % nu) as usize
    } else {
        (state.wrapping_mul(3) % nu) as usize
    }
}

fn uniform01(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let r = *state >> 11;
    (r as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_block_bootstrap_returns_valid_indices() {
        let idx = moving_block_bootstrap_indices(100, 10, 42).unwrap();
        assert_eq!(idx.len(), 100);
        for &i in &idx {
            assert!(i < 100);
        }
    }

    #[test]
    fn circular_block_bootstrap_returns_valid_indices() {
        let idx = circular_block_bootstrap_indices(50, 8, 7).unwrap();
        assert_eq!(idx.len(), 50);
        for &i in &idx {
            assert!(i < 50);
        }
    }

    #[test]
    fn stationary_bootstrap_returns_valid_indices() {
        let idx = stationary_bootstrap_indices(40, 5.0, 23).unwrap();
        assert_eq!(idx.len(), 40);
        for &i in &idx {
            assert!(i < 40);
        }
    }
}
