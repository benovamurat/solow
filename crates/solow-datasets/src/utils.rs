//! Sample-weight and resampling helpers — utils.class_weight
//! + utils.resample.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use solow_core::{Error, Result};

use crate::Lcg;

/// Compute per-sample weights so that every class contributes the same
/// total weight to the training set (the reference `compute_sample_weight('balanced')`).
pub fn compute_sample_weight(y: &[i64]) -> Result<Array1<f64>> {
    if y.is_empty() {
        return Err(Error::Value("compute_sample_weight: empty label vector".into()));
    }
    let mut counts: std::collections::BTreeMap<i64, usize> = Default::default();
    for &yi in y {
        *counts.entry(yi).or_insert(0) += 1;
    }
    let n_classes = counts.len() as f64;
    let n = y.len() as f64;
    let mut out = Array1::<f64>::zeros(y.len());
    for (i, &yi) in y.iter().enumerate() {
        let c = counts[&yi] as f64;
        out[i] = n / (n_classes * c);
    }
    Ok(out)
}

/// Compute per-class weights so that every class has weight
/// `n_samples / (n_classes * n_samples_in_class)`.
pub fn compute_class_weight(y: &[i64]) -> Result<Vec<(i64, f64)>> {
    if y.is_empty() {
        return Err(Error::Value("compute_class_weight: empty label vector".into()));
    }
    let mut counts: std::collections::BTreeMap<i64, usize> = Default::default();
    for &yi in y {
        *counts.entry(yi).or_insert(0) += 1;
    }
    let n_classes = counts.len() as f64;
    let n = y.len() as f64;
    Ok(counts
        .iter()
        .map(|(&c, &k)| (c, n / (n_classes * k as f64)))
        .collect())
}

/// Deterministic resample without replacement.
pub fn resample_indices_no_replace(n: usize, n_samples: usize, seed: u64) -> Result<Vec<usize>> {
    if n_samples > n {
        return Err(Error::Value(
            "resample_indices_no_replace: n_samples > n; use `_with_replace` instead".into(),
        ));
    }
    let mut rng = Lcg::new(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..n_samples {
        let j = i + rng.uniform_index(n - i);
        idx.swap(i, j);
    }
    idx.truncate(n_samples);
    Ok(idx)
}

/// Deterministic resample with replacement.
pub fn resample_indices_with_replace(n: usize, n_samples: usize, seed: u64) -> Result<Vec<usize>> {
    if n == 0 {
        return Err(Error::Value("resample_indices_with_replace: n = 0".into()));
    }
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        out.push(rng.uniform_index(n));
    }
    Ok(out)
}

/// Sub-select rows of `x` and `y` at the given indices.
pub fn take_rows<T: Copy + Default>(
    x: ArrayView2<'_, T>,
    y: ArrayView1<'_, T>,
    idx: &[usize],
) -> (Array2<T>, Array1<T>) {
    let d = x.ncols();
    let mut xs = Array2::<T>::default((idx.len(), d));
    let mut ys = Array1::<T>::default(idx.len());
    for (r, &i) in idx.iter().enumerate() {
        for j in 0..d {
            xs[[r, j]] = x[[i, j]];
        }
        ys[r] = y[i];
    }
    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_weight_balances_class_totals() {
        // Class 0 has 4 samples, class 1 has 2. Their per-class totals should match.
        let y = vec![0_i64, 0, 0, 0, 1, 1];
        let w = compute_sample_weight(&y).unwrap();
        let sum_class_0: f64 = (0..6).filter(|&i| y[i] == 0).map(|i| w[i]).sum();
        let sum_class_1: f64 = (0..6).filter(|&i| y[i] == 1).map(|i| w[i]).sum();
        assert!((sum_class_0 - sum_class_1).abs() < 1e-12);
    }

    #[test]
    fn resample_without_replace_returns_unique_indices() {
        let idx = resample_indices_no_replace(20, 10, 42).unwrap();
        let mut set: std::collections::BTreeSet<usize> = Default::default();
        for i in &idx {
            assert!(*i < 20);
            set.insert(*i);
        }
        assert_eq!(set.len(), 10);
    }
}
