//! Group-aware and leakage-safe cross-validation splitters.
//!
//! Three splitters live here that the plain [`crate::KFold`] family can't
//! express because they need extra information beyond the row count:
//!
//! * [`GroupKFold`] — every group is assigned to exactly one fold. A row's
//!   group is passed in a `groups` slice; typical uses are patients whose
//!   several visits must not be split across train/test, or households
//!   whose members must stay together.
//! * [`StratifiedGroupKFold`] — the group-aware analogue of
//!   [`crate::StratifiedKFold`]: groups are assigned to folds so that both
//!   (a) no group is split, and (b) the per-fold class prevalence is as
//!   close as possible to the overall prevalence.
//! * [`PurgedKFold`] — walk-forward K-fold with a **purge** band around
//!   each test fold and an optional **embargo** trailing the test fold,
//!   the standard remedy for label leakage in financial time-series
//!   (López de Prado, *Advances in Financial Machine Learning*, 2018,
//!   §7.4). Rows in the purge/embargo bands are dropped from the train
//!   set — they are neither trained on nor tested.
//!
//! Unlike the row-count-only splitters these do not implement the shared
//! [`crate::Splitter`] trait, because their `split` method needs an extra
//! argument (either `groups` or the raw row count together with the purge
//! parameters). Their split lists are the same [`crate::Split`] shape and
//! plug into every scorer/bootstrap helper the same way.

use crate::Split;
use solow_core::{Error, Result};

// ---------------------------------------------------------------------------
// GroupKFold
// ---------------------------------------------------------------------------

/// Group-aware K-fold cross-validation.
///
/// A `groups` slice with one entry per row assigns rows to opaque group
/// ids (any `usize`). Rows sharing a group are always kept together —
/// they go to the same fold. Folds are formed by a greedy longest-first
/// allocation of groups to the least-loaded fold, so the fold sizes stay
/// close to `n / n_splits` even when the group-size distribution is
/// heavily skewed. This is the same algorithm the reference
/// `GroupKFold` uses.
///
/// `n_splits` must be `≥ 2` and `≤ unique(groups)`.
#[derive(Copy, Clone, Debug)]
pub struct GroupKFold {
    n_splits: usize,
}

impl GroupKFold {
    /// Create a new `GroupKFold`. Fails if `n_splits < 2`.
    pub fn new(n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(Error::Value(format!(
                "GroupKFold: n_splits must be ≥ 2 (got {n_splits})"
            )));
        }
        Ok(Self { n_splits })
    }

    /// Number of folds this splitter emits.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Produce train/test index folds respecting `groups`.
    pub fn split(&self, groups: &[usize]) -> Result<Vec<Split>> {
        if groups.is_empty() {
            return Err(Error::Value(
                "GroupKFold::split: groups must be non-empty".into(),
            ));
        }
        // Enumerate unique groups in first-appearance order for determinism.
        let mut seen = std::collections::HashMap::<usize, usize>::new();
        let mut order: Vec<usize> = Vec::new();
        for &g in groups {
            if !seen.contains_key(&g) {
                seen.insert(g, order.len());
                order.push(g);
            }
        }
        let n_groups = order.len();
        if n_groups < self.n_splits {
            return Err(Error::Value(format!(
                "GroupKFold::split: got {n_groups} unique groups but n_splits = {}",
                self.n_splits
            )));
        }
        // Count rows per group, then sort groups by size descending; tie-break by first-appearance.
        let mut sizes = vec![0usize; n_groups];
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
        for (row_idx, &g) in groups.iter().enumerate() {
            let gi = seen[&g];
            sizes[gi] += 1;
            rows[gi].push(row_idx);
        }
        let mut order_by_size: Vec<usize> = (0..n_groups).collect();
        order_by_size.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));

        // Greedy: assign each group to the currently least-loaded fold.
        let mut fold_of_group = vec![0usize; n_groups];
        let mut fold_load = vec![0usize; self.n_splits];
        for &gi in &order_by_size {
            // Pick the fold with the smallest load; tie-break on the lowest fold index for determinism.
            let (fold, _) = fold_load
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(&b.0)))
                .unwrap();
            fold_of_group[gi] = fold;
            fold_load[fold] += sizes[gi];
        }
        // Emit one Split per fold.
        let n = groups.len();
        let mut out = Vec::with_capacity(self.n_splits);
        for k in 0..self.n_splits {
            let mut test = Vec::new();
            let mut train = Vec::with_capacity(n);
            for gi in 0..n_groups {
                if fold_of_group[gi] == k {
                    test.extend_from_slice(&rows[gi]);
                } else {
                    train.extend_from_slice(&rows[gi]);
                }
            }
            test.sort();
            train.sort();
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// StratifiedGroupKFold
// ---------------------------------------------------------------------------

/// Group-aware, class-stratified K-fold.
///
/// Assigns whole groups to folds so no group is split, while keeping the
/// per-fold class prevalence as close as possible to the global class
/// prevalence. Uses the greedy assignment the reference ships as
/// `StratifiedGroupKFold`: groups are ordered by decreasing "weight"
/// (their dominant-class count with a size tie-break) and each group is
/// placed into the fold that minimises the resulting sum-of-squared
/// deviations of per-class counts from the target counts.
#[derive(Copy, Clone, Debug)]
pub struct StratifiedGroupKFold {
    n_splits: usize,
}

impl StratifiedGroupKFold {
    /// Create a new `StratifiedGroupKFold`.
    pub fn new(n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(Error::Value(format!(
                "StratifiedGroupKFold: n_splits must be ≥ 2 (got {n_splits})"
            )));
        }
        Ok(Self { n_splits })
    }

    /// Number of folds.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Produce train/test folds respecting `groups` and stratifying by `y`.
    pub fn split(&self, y: &[usize], groups: &[usize]) -> Result<Vec<Split>> {
        if y.len() != groups.len() {
            return Err(Error::Shape(format!(
                "StratifiedGroupKFold::split: y has {} entries but groups has {}",
                y.len(),
                groups.len()
            )));
        }
        if y.is_empty() {
            return Err(Error::Value(
                "StratifiedGroupKFold::split: inputs must be non-empty".into(),
            ));
        }
        let n_classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        if n_classes == 0 {
            return Err(Error::Value(
                "StratifiedGroupKFold::split: y must contain at least one class label".into(),
            ));
        }
        // Enumerate unique groups in first-appearance order.
        let mut seen = std::collections::HashMap::<usize, usize>::new();
        let mut group_order: Vec<usize> = Vec::new();
        for &g in groups {
            if !seen.contains_key(&g) {
                seen.insert(g, group_order.len());
                group_order.push(g);
            }
        }
        let n_groups = group_order.len();
        if n_groups < self.n_splits {
            return Err(Error::Value(format!(
                "StratifiedGroupKFold::split: got {n_groups} unique groups but n_splits = {}",
                self.n_splits
            )));
        }
        // Per-group class counts and row lists.
        let mut group_counts: Vec<Vec<usize>> = vec![vec![0; n_classes]; n_groups];
        let mut group_rows: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
        for (row_idx, (&yi, &g)) in y.iter().zip(groups.iter()).enumerate() {
            let gi = seen[&g];
            group_counts[gi][yi] += 1;
            group_rows[gi].push(row_idx);
        }
        // Class totals.
        let mut total_per_class = vec![0usize; n_classes];
        for gc in &group_counts {
            for c in 0..n_classes {
                total_per_class[c] += gc[c];
            }
        }
        let target_per_fold: Vec<f64> = total_per_class
            .iter()
            .map(|&t| t as f64 / self.n_splits as f64)
            .collect();

        // Sort groups by decreasing total size (tie-break: first appearance).
        let mut group_sizes: Vec<usize> = group_counts.iter().map(|gc| gc.iter().sum()).collect();
        let mut order: Vec<usize> = (0..n_groups).collect();
        order.sort_by(|&a, &b| group_sizes[b].cmp(&group_sizes[a]).then(a.cmp(&b)));

        // Greedy assignment with a leading round-robin seed pass: the first
        // `n_splits` largest groups are dropped one-per-fold to guarantee no
        // fold is left empty (mirrors the reference StratifiedGroupKFold
        // behaviour on small samples). The remaining groups are then placed
        // into the fold that minimises the class-count sum-of-squared
        // deviations from the target.
        let mut fold_counts: Vec<Vec<f64>> = vec![vec![0.0; n_classes]; self.n_splits];
        let mut fold_of_group = vec![0usize; n_groups];
        for (seed_idx, &gi) in order.iter().take(self.n_splits).enumerate() {
            fold_of_group[gi] = seed_idx;
            for c in 0..n_classes {
                fold_counts[seed_idx][c] += group_counts[gi][c] as f64;
            }
        }
        for &gi in order.iter().skip(self.n_splits) {
            let mut best_fold = 0usize;
            let mut best_score = f64::INFINITY;
            for k in 0..self.n_splits {
                let mut score = 0.0_f64;
                for c in 0..n_classes {
                    let proposed = fold_counts[k][c] + group_counts[gi][c] as f64;
                    let dev = proposed - target_per_fold[c];
                    score += dev * dev;
                }
                if score < best_score || (score == best_score && k < best_fold) {
                    best_score = score;
                    best_fold = k;
                }
            }
            fold_of_group[gi] = best_fold;
            for c in 0..n_classes {
                fold_counts[best_fold][c] += group_counts[gi][c] as f64;
            }
        }
        // Suppress an unused-mut warning on `group_sizes`.
        let _ = &mut group_sizes;
        // Emit folds.
        let n = y.len();
        let mut out = Vec::with_capacity(self.n_splits);
        for k in 0..self.n_splits {
            let mut test = Vec::new();
            let mut train = Vec::with_capacity(n);
            for gi in 0..n_groups {
                if fold_of_group[gi] == k {
                    test.extend_from_slice(&group_rows[gi]);
                } else {
                    train.extend_from_slice(&group_rows[gi]);
                }
            }
            test.sort();
            train.sort();
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// PurgedKFold
// ---------------------------------------------------------------------------

/// K-fold time-series splitter with a purge band and an optional embargo.
///
/// The row order is treated as a temporal ordering: fold `k` uses the
/// contiguous block of test rows in `[k · n / K, (k + 1) · n / K)`. From
/// the train set, this splitter drops every row within `purge` steps of
/// the test block (on **both** sides — the past and future), and then
/// additionally drops the `embargo` rows immediately following the test
/// block. This is the López de Prado (2018) purged K-fold with an
/// embargo, the standard leakage-safe splitter for financial back-tests.
///
/// The purge width defends against the *label horizon*: if a row's label
/// depends on the next `h` observations, a nearby train row can leak
/// information into the test residual. The embargo defends against the
/// *feature horizon*: rolling features computed on the test block can
/// bleed into the immediately following train rows.
#[derive(Copy, Clone, Debug)]
pub struct PurgedKFold {
    n_splits: usize,
    purge: usize,
    embargo: usize,
}

impl PurgedKFold {
    /// Create a new `PurgedKFold` with `n_splits` folds, a symmetric
    /// `purge` band around every test block, and an `embargo` band
    /// trailing every test block.
    pub fn new(n_splits: usize, purge: usize, embargo: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(Error::Value(format!(
                "PurgedKFold: n_splits must be ≥ 2 (got {n_splits})"
            )));
        }
        Ok(Self {
            n_splits,
            purge,
            embargo,
        })
    }

    /// Number of folds.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Produce train/test folds on `n` chronologically-ordered rows.
    pub fn split(&self, n: usize) -> Result<Vec<Split>> {
        if n < self.n_splits {
            return Err(Error::Value(format!(
                "PurgedKFold::split: n = {n} must be ≥ n_splits = {}",
                self.n_splits
            )));
        }
        let mut out = Vec::with_capacity(self.n_splits);
        for k in 0..self.n_splits {
            let start = k * n / self.n_splits;
            let end = (k + 1) * n / self.n_splits;
            let purge_start = start.saturating_sub(self.purge);
            let purge_end = (end + self.purge).min(n);
            let embargo_end = (end + self.purge + self.embargo).min(n);
            let test: Vec<usize> = (start..end).collect();
            let mut train = Vec::with_capacity(n);
            for i in 0..purge_start {
                train.push(i);
            }
            // Skip [purge_start, purge_end) around the test block, plus the
            // trailing embargo up to `embargo_end`.
            for i in embargo_end.max(purge_end)..n {
                train.push(i);
            }
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CombinatorialPurgedKFold (López de Prado 2018)
// ---------------------------------------------------------------------------

/// Combinatorial Purged K-Fold cross-validation with an optional embargo.
///
/// A single walk-forward K-fold gives you `K` back-test paths; CPCV
/// instead partitions the row range into `n_splits` contiguous blocks,
/// then enumerates every subset of `n_test` blocks as a test set —
/// producing `C(n_splits, n_test)` folds along with a much richer set
/// of *distinct back-test paths* (each block appears in
/// `C(n_splits - 1, n_test - 1)` folds, so the average path count is
/// dramatically higher than for plain K-fold). Purge and embargo bands
/// work as in [`PurgedKFold`].
///
/// See López de Prado (2018), *Advances in Financial Machine Learning*,
/// §12.4 for a full derivation. CPCV is the standard leakage-safe
/// splitter when a low-variance estimate of a strategy's Sharpe or drawdown
/// is required.
#[derive(Copy, Clone, Debug)]
pub struct CombinatorialPurgedKFold {
    n_splits: usize,
    n_test: usize,
    purge: usize,
    embargo: usize,
}

impl CombinatorialPurgedKFold {
    /// New CPCV splitter. `n_test` must be `< n_splits`; typical values
    /// are `n_test = 2` for a smooth path count and `n_splits ∈ [8, 12]`.
    pub fn new(n_splits: usize, n_test: usize, purge: usize, embargo: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(Error::Value(format!(
                "CombinatorialPurgedKFold: n_splits must be ≥ 2 (got {n_splits})"
            )));
        }
        if n_test == 0 || n_test >= n_splits {
            return Err(Error::Value(format!(
                "CombinatorialPurgedKFold: n_test must satisfy 1 ≤ n_test < n_splits (got {n_test} / {n_splits})"
            )));
        }
        Ok(Self {
            n_splits,
            n_test,
            purge,
            embargo,
        })
    }

    /// Number of folds this splitter emits: `C(n_splits, n_test)`.
    pub fn n_folds(&self) -> usize {
        binomial(self.n_splits, self.n_test)
    }

    /// Enumerate every fold. Test blocks are contiguous, chosen from the
    /// `C(n_splits, n_test)` block subsets; training indices exclude
    /// every purge- and embargo-bordered row.
    pub fn split(&self, n: usize) -> Result<Vec<Split>> {
        if n < self.n_splits {
            return Err(Error::Value(format!(
                "CombinatorialPurgedKFold::split: n = {n} must be ≥ n_splits = {}",
                self.n_splits
            )));
        }
        let mut block_edges = Vec::with_capacity(self.n_splits + 1);
        for k in 0..=self.n_splits {
            block_edges.push(k * n / self.n_splits);
        }
        let mut subsets: Vec<Vec<usize>> = Vec::with_capacity(self.n_folds());
        let mut current = vec![0usize; self.n_test];
        fill_subsets(&mut subsets, &mut current, 0, 0, self.n_splits, self.n_test);

        let mut out = Vec::with_capacity(subsets.len());
        for subset in &subsets {
            let mut test: Vec<usize> = Vec::new();
            for &b in subset {
                test.extend(block_edges[b]..block_edges[b + 1]);
            }
            let mut mask = vec![false; n];
            for &b in subset {
                let (start, end) = (block_edges[b], block_edges[b + 1]);
                let purge_start = start.saturating_sub(self.purge);
                let purge_end = (end + self.purge).min(n);
                let embargo_end = (end + self.purge + self.embargo).min(n);
                for i in purge_start..embargo_end.max(purge_end) {
                    mask[i] = true;
                }
            }
            for &t in &test {
                mask[t] = true;
            }
            let train: Vec<usize> = (0..n).filter(|&i| !mask[i]).collect();
            test.sort();
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

fn fill_subsets(
    out: &mut Vec<Vec<usize>>,
    current: &mut Vec<usize>,
    slot: usize,
    start: usize,
    n_splits: usize,
    n_test: usize,
) {
    if slot == n_test {
        out.push(current.clone());
        return;
    }
    let remaining_slots = n_test - slot;
    for i in start..=(n_splits - remaining_slots) {
        current[slot] = i;
        fill_subsets(out, current, slot + 1, i + 1, n_splits, n_test);
    }
}
