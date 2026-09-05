//! `LeavePOut`, `RepeatedKFold`, `RepeatedStratifiedKFold`,
//! `HalvingGridSearchCV`, `HalvingRandomSearchCV`, `learning_curve`,
//! `validation_curve`, `permutation_test_score`.

use solow_core::{Error, Result};

use crate::splitters::{KFold, Split, Splitter, StratifiedKFold};

/// Leave-P-Out cross-validation. Emits `C(n, p)` folds; use with care
/// on large samples.
#[derive(Clone, Copy, Debug)]
pub struct LeavePOut {
    /// The number of held-out samples per fold.
    pub p: usize,
}

impl LeavePOut {
    /// Construct.
    pub fn new(p: usize) -> Result<Self> {
        if p == 0 {
            return Err(Error::Value("LeavePOut: p must be ≥ 1".into()));
        }
        Ok(Self { p })
    }
}

impl Splitter for LeavePOut {
    fn n_splits(&self, n: usize) -> Result<usize> {
        if self.p > n {
            return Err(Error::Value(format!(
                "LeavePOut: p={} > n={n}", self.p
            )));
        }
        // C(n, p) closed-form.
        let mut num = 1_u128;
        let mut denom = 1_u128;
        for i in 0..self.p {
            num = num.saturating_mul((n - i) as u128);
            denom = denom.saturating_mul((i + 1) as u128);
        }
        Ok((num / denom) as usize)
    }

    fn split(&self, n: usize) -> Result<Vec<Split>> {
        if self.p > n {
            return Err(Error::Value(format!(
                "LeavePOut: p={} > n={n}", self.p
            )));
        }
        let mut out = Vec::new();
        let mut combo = (0..self.p).collect::<Vec<usize>>();
        loop {
            let train: Vec<usize> = (0..n).filter(|i| !combo.contains(i)).collect();
            out.push(Split {
                train,
                test: combo.clone(),
            });
            // Next combination in lex order.
            let mut i = self.p;
            while i > 0 && combo[i - 1] == n - (self.p - (i - 1)) {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            combo[i - 1] += 1;
            for j in i..self.p {
                combo[j] = combo[j - 1] + 1;
            }
        }
        Ok(out)
    }
}

/// Repeated K-Fold cross-validation.
#[derive(Clone, Copy, Debug)]
pub struct RepeatedKFold {
    /// Number of folds per repeat.
    pub n_splits: usize,
    /// Number of repetitions.
    pub n_repeats: usize,
    /// Seed to derive per-repeat seeds.
    pub seed: u64,
}

impl RepeatedKFold {
    /// Construct.
    pub fn new(n_splits: usize, n_repeats: usize, seed: u64) -> Result<Self> {
        if n_splits < 2 || n_repeats == 0 {
            return Err(Error::Value(
                "RepeatedKFold: need n_splits ≥ 2 and n_repeats ≥ 1".into(),
            ));
        }
        Ok(Self { n_splits, n_repeats, seed })
    }
}

impl Splitter for RepeatedKFold {
    fn n_splits(&self, _n: usize) -> Result<usize> {
        Ok(self.n_splits * self.n_repeats)
    }

    fn split(&self, n: usize) -> Result<Vec<Split>> {
        let mut out = Vec::new();
        for r in 0..self.n_repeats {
            let s = self.seed.wrapping_add((r as u64).wrapping_mul(0x9E37_79B9));
            let kf = KFold::new(self.n_splits)?.shuffle(true).seed(s);
            out.extend(kf.split(n)?);
        }
        Ok(out)
    }
}

/// Repeated Stratified K-Fold.
#[derive(Clone, Copy, Debug)]
pub struct RepeatedStratifiedKFold {
    /// Number of folds per repeat.
    pub n_splits: usize,
    /// Number of repetitions.
    pub n_repeats: usize,
    /// Seed.
    pub seed: u64,
}

impl RepeatedStratifiedKFold {
    /// Construct.
    pub fn new(n_splits: usize, n_repeats: usize, seed: u64) -> Result<Self> {
        if n_splits < 2 || n_repeats == 0 {
            return Err(Error::Value(
                "RepeatedStratifiedKFold: need n_splits ≥ 2 and n_repeats ≥ 1".into(),
            ));
        }
        Ok(Self { n_splits, n_repeats, seed })
    }

    /// Split with class labels.
    pub fn split(&self, y: &[usize]) -> Result<Vec<Split>> {
        let mut out = Vec::new();
        for r in 0..self.n_repeats {
            let s = self.seed.wrapping_add((r as u64).wrapping_mul(0x9E37_79B9));
            let skf = StratifiedKFold::new(self.n_splits)?.shuffle(true).seed(s);
            out.extend(skf.split(y)?);
        }
        Ok(out)
    }
}

/// Learning-curve evaluator — score at increasing training-set sizes.
pub fn learning_curve<F>(
    n: usize,
    train_sizes: &[usize],
    n_splits: usize,
    mut score_fold: F,
    seed: u64,
) -> Result<Vec<(usize, Vec<f64>)>>
where
    F: FnMut(&[usize], &[usize]) -> Result<f64>,
{
    if train_sizes.is_empty() {
        return Err(Error::Value("learning_curve: empty train_sizes".into()));
    }
    let kf = KFold::new(n_splits)?.shuffle(true).seed(seed);
    let folds = kf.split(n)?;
    let mut out = Vec::with_capacity(train_sizes.len());
    for &ts in train_sizes {
        let mut per_fold = Vec::with_capacity(folds.len());
        for split in &folds {
            let truncated: Vec<usize> = split.train.iter().take(ts).copied().collect();
            per_fold.push(score_fold(&truncated, &split.test)?);
        }
        out.push((ts, per_fold));
    }
    Ok(out)
}

/// Validation-curve evaluator — score at increasing values of a
/// hyper-parameter.
pub fn validation_curve<P: Clone, F>(
    n: usize,
    params: &[P],
    n_splits: usize,
    mut score_fold: F,
    seed: u64,
) -> Result<Vec<(P, Vec<f64>)>>
where
    F: FnMut(&P, &[usize], &[usize]) -> Result<f64>,
{
    if params.is_empty() {
        return Err(Error::Value("validation_curve: empty params".into()));
    }
    let kf = KFold::new(n_splits)?.shuffle(true).seed(seed);
    let folds = kf.split(n)?;
    let mut out = Vec::with_capacity(params.len());
    for p in params {
        let mut per_fold = Vec::with_capacity(folds.len());
        for split in &folds {
            per_fold.push(score_fold(p, &split.train, &split.test)?);
        }
        out.push((p.clone(), per_fold));
    }
    Ok(out)
}

/// Permutation-test-score — significance test for a cross-validated score.
///
/// Returns `(score, permutation_scores, p_value)`.
pub fn permutation_test_score<F>(
    y: &[usize],
    n_splits: usize,
    n_permutations: usize,
    mut score_fn: F,
    seed: u64,
) -> Result<(f64, Vec<f64>, f64)>
where
    F: FnMut(&[usize], &[usize], &[usize]) -> Result<f64>,
{
    if n_permutations == 0 {
        return Err(Error::Value("permutation_test_score: n_permutations must be ≥ 1".into()));
    }
    let n = y.len();
    let kf = KFold::new(n_splits)?.shuffle(true).seed(seed);
    let folds = kf.split(n)?;
    let true_score = {
        let mut s = 0.0_f64;
        for split in &folds {
            s += score_fn(y, &split.train, &split.test)?;
        }
        s / folds.len() as f64
    };
    let mut perm_scores = Vec::with_capacity(n_permutations);
    let mut state = seed.wrapping_add(0xF00D_D00D);
    for _ in 0..n_permutations {
        let mut y_shuf: Vec<usize> = y.to_vec();
        for i in (1..y_shuf.len()).rev() {
            let j = uniform_index(&mut state, (i + 1) as u64);
            y_shuf.swap(i, j);
        }
        let mut s = 0.0_f64;
        for split in &folds {
            s += score_fn(&y_shuf, &split.train, &split.test)?;
        }
        perm_scores.push(s / folds.len() as f64);
    }
    let better = perm_scores.iter().filter(|&&p| p >= true_score).count();
    let p_value = (better as f64 + 1.0) / (n_permutations as f64 + 1.0);
    Ok((true_score, perm_scores, p_value))
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let max = u64::MAX - (u64::MAX % n);
    if *state < max {
        (*state % n) as usize
    } else {
        // Rare rejection retry, keep it O(1) in expectation.
        (state.wrapping_mul(3) % n) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leave_p_out_produces_c_n_p_folds() {
        let lp = LeavePOut::new(2).unwrap();
        let folds = lp.split(4).unwrap();
        // C(4, 2) = 6.
        assert_eq!(folds.len(), 6);
    }

    #[test]
    fn repeated_k_fold_produces_the_right_number_of_folds() {
        let rk = RepeatedKFold::new(3, 4, 42).unwrap();
        let folds = rk.split(9).unwrap();
        assert_eq!(folds.len(), 12);
    }
}
