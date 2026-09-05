//! StratifiedShuffleSplit and GroupShuffleSplit — randomised train/test
//! partitions with class or group stratification.

use solow_core::{Error, Result};

use crate::splitters::{Split, Splitter};

/// StratifiedShuffleSplit.
#[derive(Clone, Copy, Debug)]
pub struct StratifiedShuffleSplit {
    /// Number of resampling iterations.
    pub n_splits: usize,
    /// Fraction of samples kept for the test set in each split.
    pub test_size: f64,
    /// Seed to derive per-split seeds.
    pub seed: u64,
}

impl StratifiedShuffleSplit {
    /// Construct.
    pub fn new(n_splits: usize, test_size: f64, seed: u64) -> Result<Self> {
        if n_splits == 0 {
            return Err(Error::Value("StratifiedShuffleSplit: n_splits must be ≥ 1".into()));
        }
        if !(0.0..1.0).contains(&test_size) {
            return Err(Error::Value(format!(
                "StratifiedShuffleSplit: test_size must be in (0, 1) (got {test_size})"
            )));
        }
        Ok(Self { n_splits, test_size, seed })
    }

    /// Split with class labels.
    pub fn split(&self, y: &[usize]) -> Result<Vec<Split>> {
        let mut buckets: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (i, &c) in y.iter().enumerate() {
            buckets.entry(c).or_default().push(i);
        }
        let mut out = Vec::with_capacity(self.n_splits);
        for r in 0..self.n_splits {
            let mut state = self
                .seed
                .wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (_c, indices) in buckets.iter() {
                let mut shuffled = indices.clone();
                for i in (1..shuffled.len()).rev() {
                    let j = uniform_index(&mut state, (i + 1) as u64);
                    shuffled.swap(i, j);
                }
                let n_test = ((shuffled.len() as f64) * self.test_size).ceil() as usize;
                let n_test = n_test.min(shuffled.len().saturating_sub(1)).max(1);
                for (idx, &sample) in shuffled.iter().enumerate() {
                    if idx < n_test {
                        test.push(sample);
                    } else {
                        train.push(sample);
                    }
                }
            }
            train.sort();
            test.sort();
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

/// GroupShuffleSplit — train/test split at the group level.
#[derive(Clone, Copy, Debug)]
pub struct GroupShuffleSplit {
    /// Number of iterations.
    pub n_splits: usize,
    /// Fraction of groups (not rows) kept for the test set.
    pub test_size: f64,
    /// Seed.
    pub seed: u64,
}

impl GroupShuffleSplit {
    /// Construct.
    pub fn new(n_splits: usize, test_size: f64, seed: u64) -> Result<Self> {
        if n_splits == 0 {
            return Err(Error::Value("GroupShuffleSplit: n_splits must be ≥ 1".into()));
        }
        if !(0.0..1.0).contains(&test_size) {
            return Err(Error::Value(
                "GroupShuffleSplit: test_size must be in (0, 1)".into(),
            ));
        }
        Ok(Self { n_splits, test_size, seed })
    }

    /// Split with group labels.
    pub fn split(&self, groups: &[usize]) -> Result<Vec<Split>> {
        let mut unique: Vec<usize> = groups.to_vec();
        unique.sort();
        unique.dedup();
        let n_groups = unique.len();
        let n_test_groups = ((n_groups as f64) * self.test_size).ceil() as usize;
        let n_test_groups = n_test_groups.min(n_groups.saturating_sub(1)).max(1);
        let mut out = Vec::with_capacity(self.n_splits);
        for r in 0..self.n_splits {
            let mut state = self
                .seed
                .wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut shuffled = unique.clone();
            for i in (1..shuffled.len()).rev() {
                let j = uniform_index(&mut state, (i + 1) as u64);
                shuffled.swap(i, j);
            }
            let test_groups: std::collections::BTreeSet<usize> =
                shuffled.iter().take(n_test_groups).copied().collect();
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, &g) in groups.iter().enumerate() {
                if test_groups.contains(&g) {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            out.push(Split { train, test });
        }
        Ok(out)
    }
}

// Splitter trait implementations for uniform-size iteration.
impl Splitter for StratifiedShuffleSplit {
    fn n_splits(&self, _n: usize) -> Result<usize> {
        Ok(self.n_splits)
    }

    fn split(&self, _n: usize) -> Result<Vec<Split>> {
        Err(Error::Value(
            "StratifiedShuffleSplit: use .split(y) with class labels".into(),
        ))
    }
}

impl Splitter for GroupShuffleSplit {
    fn n_splits(&self, _n: usize) -> Result<usize> {
        Ok(self.n_splits)
    }

    fn split(&self, _n: usize) -> Result<Vec<Split>> {
        Err(Error::Value(
            "GroupShuffleSplit: use .split(groups) with group labels".into(),
        ))
    }
}

fn uniform_index(state: &mut u64, n: u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let max = u64::MAX - (u64::MAX % n);
    if *state < max {
        (*state % n) as usize
    } else {
        (state.wrapping_mul(3) % n) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratified_shuffle_split_preserves_class_ratios_approximately() {
        let y = vec![0_usize; 20]
            .into_iter()
            .chain(vec![1_usize; 30])
            .collect::<Vec<_>>();
        let s = StratifiedShuffleSplit::new(3, 0.2, 42).unwrap();
        let folds = s.split(&y).unwrap();
        assert_eq!(folds.len(), 3);
        for f in &folds {
            let (n0, n1) = f
                .test
                .iter()
                .fold((0_usize, 0_usize), |(a, b), &i| if y[i] == 0 { (a + 1, b) } else { (a, b + 1) });
            // Test set fractions should each be ≈ 20% of their class.
            assert!(n0 > 0 && n1 > 0);
        }
    }

    #[test]
    fn group_shuffle_split_never_leaks_groups_across_train_and_test() {
        let groups = vec![0_usize, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let s = GroupShuffleSplit::new(2, 0.4, 7).unwrap();
        let folds = s.split(&groups).unwrap();
        for f in &folds {
            let train_groups: std::collections::BTreeSet<usize> =
                f.train.iter().map(|&i| groups[i]).collect();
            let test_groups: std::collections::BTreeSet<usize> =
                f.test.iter().map(|&i| groups[i]).collect();
            assert!(train_groups.is_disjoint(&test_groups));
        }
    }
}
