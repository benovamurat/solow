//! Cross-validation splitters.
//!
//! Every splitter validates its inputs up front and produces a `Vec<Split>`
//! that the caller can iterate, index, or split across threads.
//!
//! The splitters are deterministic: with a fixed shuffle seed, two runs on
//! the same input produce identical folds.

use solow_core::{Error, Result};

/// One train / test fold. Indices are sample positions (`0..n`) that a caller
/// hands to `ndarray::select` or a slice-based indexing helper.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    /// Row indices to fit on.
    pub train: Vec<usize>,
    /// Row indices to score on.
    pub test: Vec<usize>,
}

/// Every splitter produces a validated list of folds.
pub trait Splitter {
    /// The number of folds this splitter will yield on `n` observations.
    fn n_splits(&self, n: usize) -> Result<usize>;

    /// Materialise every fold, ready to index into.
    fn split(&self, n: usize) -> Result<Vec<Split>>;
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (MMIX 64-bit LCG)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    // Donald Knuth's MMIX LCG — full-period 2^64, well-mixed high bits.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Deterministic Fisher-Yates shuffle in place, seeded by `seed`.
fn shuffle_indices_in_place(idx: &mut [usize], seed: u64) {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..idx.len()).rev() {
        // Uniform integer in [0, i] with rejection sampling.
        let range = (i as u64) + 1;
        let max = u64::MAX - (u64::MAX % range);
        let mut r = lcg_next(&mut state);
        while r >= max {
            r = lcg_next(&mut state);
        }
        let j = (r % range) as usize;
        idx.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// KFold
// ---------------------------------------------------------------------------

/// K-fold cross-validator with `n_splits` equal-sized folds.
///
/// When the sample count `n` is not divisible by `k`, the first `n mod k`
/// folds are one observation larger, matching the reference.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    seed: u64,
}

impl KFold {
    /// A `k`-fold splitter with `k = n_splits`; `n_splits ≥ 2` is required.
    pub fn new(n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(Error::Value(format!(
                "KFold requires n_splits ≥ 2, got {n_splits}"
            )));
        }
        Ok(Self {
            n_splits,
            shuffle: false,
            seed: 0,
        })
    }

    /// Enable or disable shuffling before splitting. Off by default.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Seed for the shuffle PRNG. Only takes effect when [`shuffle`](Self::shuffle)
    /// is `true`.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Splitter for KFold {
    fn n_splits(&self, n: usize) -> Result<usize> {
        if n < self.n_splits {
            return Err(Error::Value(format!(
                "KFold: n = {n} must be ≥ n_splits = {}",
                self.n_splits
            )));
        }
        Ok(self.n_splits)
    }

    fn split(&self, n: usize) -> Result<Vec<Split>> {
        let k = self.n_splits(n)?;
        let mut indices: Vec<usize> = (0..n).collect();
        if self.shuffle {
            shuffle_indices_in_place(&mut indices, self.seed);
        }
        let base = n / k;
        let extra = n % k;
        let mut folds: Vec<Vec<usize>> = Vec::with_capacity(k);
        let mut start = 0usize;
        for i in 0..k {
            let size = base + if i < extra { 1 } else { 0 };
            folds.push(indices[start..start + size].to_vec());
            start += size;
        }
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let test = folds[i].clone();
            let mut train: Vec<usize> = Vec::with_capacity(n - test.len());
            for (j, fold) in folds.iter().enumerate() {
                if j != i {
                    train.extend_from_slice(fold);
                }
            }
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// StratifiedKFold
// ---------------------------------------------------------------------------

/// K-fold cross-validator that preserves the per-class label distribution in
/// every fold.
///
/// The label vector is `y: &[usize]` — a classification target expressed as
/// integer class indices. Every fold receives approximately the same
/// proportion of each class as the full sample. This is the natural
/// resampler for imbalanced-class problems, and reduces to plain [`KFold`]
/// when every class is equally frequent.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    seed: u64,
}

impl StratifiedKFold {
    /// A stratified `k`-fold splitter; `n_splits ≥ 2` is required.
    pub fn new(n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(Error::Value(format!(
                "StratifiedKFold requires n_splits ≥ 2, got {n_splits}"
            )));
        }
        Ok(Self {
            n_splits,
            shuffle: false,
            seed: 0,
        })
    }

    /// Enable or disable within-class shuffling before splitting.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Seed for the shuffle PRNG.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Materialise the folds for a labelled sample.
    pub fn split(&self, y: &[usize]) -> Result<Vec<Split>> {
        let n = y.len();
        if n < self.n_splits {
            return Err(Error::Value(format!(
                "StratifiedKFold: n = {n} must be ≥ n_splits = {}",
                self.n_splits
            )));
        }
        // Group sample indices by class, preserving observation order.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort();
        classes.dedup();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); classes.len()];
        let class_index = |c: usize| classes.binary_search(&c).unwrap();
        for (i, &c) in y.iter().enumerate() {
            buckets[class_index(c)].push(i);
        }
        // Every class must have at least `n_splits` observations, matching
        // the reference stratification requirement.
        for (ci, bucket) in buckets.iter().enumerate() {
            if bucket.len() < self.n_splits {
                return Err(Error::Value(format!(
                    "StratifiedKFold: class {} has {} samples, needs ≥ n_splits = {}",
                    classes[ci],
                    bucket.len(),
                    self.n_splits
                )));
            }
        }
        if self.shuffle {
            for bucket in buckets.iter_mut() {
                shuffle_indices_in_place(bucket, self.seed);
            }
        }
        // Deal each class's indices round-robin across the folds so counts
        // stay balanced. This matches the reference algorithm.
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];
        for bucket in &buckets {
            for (i, &idx) in bucket.iter().enumerate() {
                folds[i % self.n_splits].push(idx);
            }
        }
        // Sort each fold to give a stable, comparable order.
        for fold in folds.iter_mut() {
            fold.sort();
        }
        let mut out = Vec::with_capacity(self.n_splits);
        for i in 0..self.n_splits {
            let test = folds[i].clone();
            let mut train: Vec<usize> = Vec::with_capacity(n - test.len());
            for (j, fold) in folds.iter().enumerate() {
                if j != i {
                    train.extend_from_slice(fold);
                }
            }
            train.sort();
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TimeSeriesSplit
// ---------------------------------------------------------------------------

/// Expanding-window walk-forward validation for time series.
///
/// Fold `i ∈ [0, n_splits)` trains on the first `n₀ + i·h` observations and
/// tests on the next `h`, where `h = test_size` (defaults to `n / (n_splits +
/// 1)`, matching the reference) and `n₀` is chosen so that every test window
/// fits. The training set never contains an index later than any test index,
/// so no future information leaks into the fit.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct TimeSeriesSplit {
    n_splits: usize,
    test_size: Option<usize>,
    gap: usize,
    max_train_size: Option<usize>,
}

impl TimeSeriesSplit {
    /// A time-series splitter with `n_splits ≥ 2` walk-forward folds.
    pub fn new(n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(Error::Value(format!(
                "TimeSeriesSplit requires n_splits ≥ 2, got {n_splits}"
            )));
        }
        Ok(Self {
            n_splits,
            test_size: None,
            gap: 0,
            max_train_size: None,
        })
    }

    /// Explicit test-window size. Defaults to `n / (n_splits + 1)` when unset.
    pub fn test_size(mut self, test_size: usize) -> Self {
        self.test_size = Some(test_size);
        self
    }

    /// Number of samples to skip between the end of the training window and
    /// the start of the test window. Defaults to `0`.
    pub fn gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Cap the training window to at most `max_train_size` samples — a
    /// rolling-window (instead of expanding) validation.
    pub fn max_train_size(mut self, max_train_size: usize) -> Self {
        self.max_train_size = Some(max_train_size);
        self
    }
}

impl Splitter for TimeSeriesSplit {
    fn n_splits(&self, _n: usize) -> Result<usize> {
        Ok(self.n_splits)
    }

    fn split(&self, n: usize) -> Result<Vec<Split>> {
        let test_size = self.test_size.unwrap_or(n / (self.n_splits + 1));
        if test_size == 0 {
            return Err(Error::Value(format!(
                "TimeSeriesSplit: computed test_size = 0 for n = {n}, n_splits = {}",
                self.n_splits
            )));
        }
        // Total samples required: at least one training row plus n_splits test
        // windows plus n_splits gaps.
        let required = 1 + self.n_splits * (test_size + self.gap);
        if n < required {
            return Err(Error::Value(format!(
                "TimeSeriesSplit: n = {n} is too small; needs ≥ {required} for {} splits at test_size {test_size} (gap {})",
                self.n_splits, self.gap
            )));
        }
        let mut out = Vec::with_capacity(self.n_splits);
        for i in 0..self.n_splits {
            let test_start = n - (self.n_splits - i) * test_size;
            let test_end = test_start + test_size;
            let train_end = test_start - self.gap;
            if train_end == 0 {
                return Err(Error::Value(
                    "TimeSeriesSplit: gap consumed all training data at some fold".into(),
                ));
            }
            let train_start = match self.max_train_size {
                Some(m) if train_end > m => train_end - m,
                _ => 0,
            };
            let train: Vec<usize> = (train_start..train_end).collect();
            let test: Vec<usize> = (test_start..test_end).collect();
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// LeaveOneOut
// ---------------------------------------------------------------------------

/// Leave-one-out cross-validation.
///
/// Yields `n` folds where each fold's test set is a single observation.
/// Equivalent to [`KFold::new(n)`] without shuffling; provided as a
/// standalone type for API clarity.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default)]
pub struct LeaveOneOut;

impl LeaveOneOut {
    /// Construct a leave-one-out splitter.
    pub fn new() -> Self {
        LeaveOneOut
    }
}

impl Splitter for LeaveOneOut {
    fn n_splits(&self, n: usize) -> Result<usize> {
        if n < 2 {
            return Err(Error::Value(format!("LeaveOneOut requires n ≥ 2, got {n}")));
        }
        Ok(n)
    }

    fn split(&self, n: usize) -> Result<Vec<Split>> {
        let _ = self.n_splits(n)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut train: Vec<usize> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if j != i {
                    train.push(j);
                }
            }
            out.push(Split {
                train,
                test: vec![i],
            });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// ShuffleSplit
// ---------------------------------------------------------------------------

/// Repeated random train / test splits at a chosen test fraction.
///
/// Unlike [`KFold`], `ShuffleSplit` does not partition the sample: successive
/// folds can share test observations, and the union of the test sets need
/// not cover every sample. The PRNG stream is fully deterministic given
/// `seed`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct ShuffleSplit {
    n_splits: usize,
    test_fraction: f64,
    seed: u64,
}

impl ShuffleSplit {
    /// `n_splits` random splits with the given `test_fraction ∈ (0, 1)`.
    pub fn new(n_splits: usize, test_fraction: f64) -> Result<Self> {
        if n_splits == 0 {
            return Err(Error::Value("ShuffleSplit requires n_splits ≥ 1".into()));
        }
        if !(0.0 < test_fraction && test_fraction < 1.0) {
            return Err(Error::Value(format!(
                "ShuffleSplit: test_fraction must be in (0, 1), got {test_fraction}"
            )));
        }
        Ok(Self {
            n_splits,
            test_fraction,
            seed: 0,
        })
    }

    /// PRNG seed for reproducible splits.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Splitter for ShuffleSplit {
    fn n_splits(&self, _n: usize) -> Result<usize> {
        Ok(self.n_splits)
    }

    fn split(&self, n: usize) -> Result<Vec<Split>> {
        if n < 2 {
            return Err(Error::Value(format!("ShuffleSplit: n = {n} must be ≥ 2")));
        }
        let test_size = ((n as f64) * self.test_fraction).round() as usize;
        if test_size == 0 || test_size >= n {
            return Err(Error::Value(format!(
                "ShuffleSplit: test_fraction = {} gives test_size = {test_size} on n = {n}",
                self.test_fraction
            )));
        }
        let mut out = Vec::with_capacity(self.n_splits);
        for r in 0..self.n_splits {
            let mut idx: Vec<usize> = (0..n).collect();
            shuffle_indices_in_place(
                &mut idx,
                self.seed
                    .wrapping_add(0xA0761D6478BD642F_u64.wrapping_mul((r as u64).wrapping_add(1))),
            );
            let (test_part, train_part) = idx.split_at(test_size);
            let mut test = test_part.to_vec();
            let mut train = train_part.to_vec();
            test.sort();
            train.sort();
            out.push(Split { train, test });
        }
        Ok(out)
    }
}
