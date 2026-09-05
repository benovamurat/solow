//! Reference tests for solow-cv splitters.
//!
//! Every fold produced by each splitter is checked against the invariants a
//! canonical the reference implementation guarantees:
//!
//! * partitioned splitters (K-fold, stratified K-fold, leave-one-out) produce
//!   test sets that jointly cover every observation exactly once;
//! * every fold's train and test sets are disjoint;
//! * time-series splits never contain a training index later than any test
//!   index;
//! * shuffled folds are reproducible given a fixed seed.

use std::collections::HashSet;

use solow_cv::{
    KFold, LeaveOneOut, ShuffleSplit, Split, Splitter, StratifiedKFold, TimeSeriesSplit,
};

fn assert_disjoint(train: &[usize], test: &[usize]) {
    let tset: HashSet<_> = test.iter().copied().collect();
    for &t in train {
        assert!(!tset.contains(&t), "train / test overlap on {t}");
    }
}

fn assert_covers(folds: &[Split], n: usize) {
    let mut union: HashSet<usize> = HashSet::new();
    for s in folds {
        union.extend(&s.test);
    }
    assert_eq!(union.len(), n, "test sets do not cover every sample");
}

// ---------------------------------------------------------------------------
// KFold
// ---------------------------------------------------------------------------

#[test]
fn kfold_partitions_and_sizes() {
    let kf = KFold::new(3).unwrap();
    let folds = kf.split(10).unwrap();
    assert_eq!(folds.len(), 3);
    // the reference rule: the first (n mod k) folds are one bigger.
    let sizes: Vec<usize> = folds.iter().map(|s| s.test.len()).collect();
    assert_eq!(sizes, vec![4, 3, 3]);
    for s in &folds {
        assert_disjoint(&s.train, &s.test);
        assert_eq!(s.train.len() + s.test.len(), 10);
    }
    assert_covers(&folds, 10);
}

#[test]
fn kfold_shuffle_is_reproducible() {
    let a = KFold::new(4).unwrap().shuffle(true).seed(42);
    let b = KFold::new(4).unwrap().shuffle(true).seed(42);
    let c = KFold::new(4).unwrap().shuffle(true).seed(43);
    assert_eq!(a.split(24).unwrap(), b.split(24).unwrap());
    assert_ne!(a.split(24).unwrap(), c.split(24).unwrap());
}

#[test]
fn kfold_rejects_n_less_than_k() {
    let kf = KFold::new(5).unwrap();
    assert!(kf.split(4).is_err());
}

// ---------------------------------------------------------------------------
// StratifiedKFold
// ---------------------------------------------------------------------------

#[test]
fn stratified_kfold_preserves_class_proportions() {
    // 12 samples, 3 classes, all size 4 → every fold gets one of each class.
    let y: Vec<usize> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    let folds = StratifiedKFold::new(4).unwrap().split(&y).unwrap();
    assert_eq!(folds.len(), 4);
    for s in &folds {
        assert_eq!(s.test.len(), 3);
        assert_disjoint(&s.train, &s.test);
        let mut classes: Vec<usize> = s.test.iter().map(|&i| y[i]).collect();
        classes.sort();
        assert_eq!(classes, vec![0, 1, 2]);
    }
    assert_covers(&folds, 12);
}

#[test]
fn stratified_kfold_rejects_undersized_class() {
    let y = vec![0, 0, 0, 0, 1];
    // Class 1 has 1 sample but we asked for 3 folds.
    assert!(StratifiedKFold::new(3).unwrap().split(&y).is_err());
}

// ---------------------------------------------------------------------------
// TimeSeriesSplit
// ---------------------------------------------------------------------------

#[test]
fn timeseries_split_walks_forward() {
    // n = 10, k = 4 → default test_size = 2 (10 / 5).
    let ts = TimeSeriesSplit::new(4).unwrap();
    let folds = ts.split(10).unwrap();
    assert_eq!(folds.len(), 4);
    for s in &folds {
        assert!(!s.train.is_empty() && !s.test.is_empty());
        let max_train = *s.train.iter().max().unwrap();
        let min_test = *s.test.iter().min().unwrap();
        assert!(
            max_train < min_test,
            "training index {max_train} is not strictly before test index {min_test}"
        );
    }
    // Training window monotone non-decreasing (expanding).
    let sizes: Vec<usize> = folds.iter().map(|s| s.train.len()).collect();
    for w in sizes.windows(2) {
        assert!(w[1] >= w[0]);
    }
}

#[test]
fn timeseries_split_respects_gap_and_max_train_size() {
    // n = 20, gap = 1, test_size = 3, max_train_size = 5, splits = 4.
    let ts = TimeSeriesSplit::new(4)
        .unwrap()
        .test_size(3)
        .gap(1)
        .max_train_size(5);
    let folds = ts.split(20).unwrap();
    assert_eq!(folds.len(), 4);
    for s in &folds {
        assert!(s.train.len() <= 5);
        let max_train = *s.train.iter().max().unwrap();
        let min_test = *s.test.iter().min().unwrap();
        assert!(
            min_test - max_train >= 2,
            "gap not enforced (test - max_train = {})",
            min_test - max_train
        );
    }
}

// ---------------------------------------------------------------------------
// LeaveOneOut
// ---------------------------------------------------------------------------

#[test]
fn leave_one_out_produces_n_folds_each_of_size_one() {
    let folds = LeaveOneOut::new().split(6).unwrap();
    assert_eq!(folds.len(), 6);
    for (i, s) in folds.iter().enumerate() {
        assert_eq!(s.test, vec![i]);
        assert_eq!(s.train.len(), 5);
        assert_disjoint(&s.train, &s.test);
    }
    assert_covers(&folds, 6);
}

// ---------------------------------------------------------------------------
// ShuffleSplit
// ---------------------------------------------------------------------------

#[test]
fn shuffle_split_is_seeded_reproducible() {
    let a = ShuffleSplit::new(5, 0.25).unwrap().seed(7);
    let b = ShuffleSplit::new(5, 0.25).unwrap().seed(7);
    let c = ShuffleSplit::new(5, 0.25).unwrap().seed(8);
    let va = a.split(20).unwrap();
    let vb = b.split(20).unwrap();
    let vc = c.split(20).unwrap();
    assert_eq!(va, vb);
    assert_ne!(va, vc);
    for s in &va {
        assert!(!s.train.is_empty());
        assert!(!s.test.is_empty());
        assert_disjoint(&s.train, &s.test);
        assert_eq!(s.train.len() + s.test.len(), 20);
    }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

use solow_cv::{bootstrap_ci, cross_val_score, BootstrapMethod};

#[test]
fn bootstrap_percentile_ci_covers_the_true_mean() {
    let data: Vec<f64> = (0..200).map(|i| 5.0 + (i as f64 * 0.03).sin()).collect();
    let stat = |idx: &[usize]| {
        Ok::<f64, solow_core::Error>(idx.iter().map(|&i| data[i]).sum::<f64>() / idx.len() as f64)
    };
    let ci = bootstrap_ci(data.len(), stat, 999, 0.95, BootstrapMethod::Percentile, 7).unwrap();
    let true_mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    assert!(ci.low < ci.high);
    assert!(ci.low < true_mean && true_mean < ci.high);
    // Reproducibility: same seed → identical replicates.
    let ci2 = bootstrap_ci(data.len(), stat, 999, 0.95, BootstrapMethod::Percentile, 7).unwrap();
    assert_eq!(ci.replicates, ci2.replicates);
}

#[test]
fn bootstrap_all_methods_are_ordered() {
    let data: Vec<f64> = (0..80)
        .map(|i| 3.0 + (i as f64 * 0.11).cos() * 2.0)
        .collect();
    let stat = |idx: &[usize]| {
        Ok::<f64, solow_core::Error>(idx.iter().map(|&i| data[i]).sum::<f64>() / idx.len() as f64)
    };
    for method in [
        BootstrapMethod::Percentile,
        BootstrapMethod::Basic,
        BootstrapMethod::Bca,
    ] {
        let ci = bootstrap_ci(data.len(), stat, 599, 0.90, method, 42).unwrap();
        assert!(ci.low <= ci.point);
        assert!(ci.point <= ci.high);
        assert!(ci.standard_error() > 0.0);
    }
}

#[test]
fn bootstrap_rejects_bad_inputs() {
    let stat = |_idx: &[usize]| Ok::<f64, solow_core::Error>(0.0);
    assert!(bootstrap_ci(0, stat, 100, 0.95, BootstrapMethod::Percentile, 1).is_err());
    assert!(bootstrap_ci(10, stat, 0, 0.95, BootstrapMethod::Percentile, 1).is_err());
    assert!(bootstrap_ci(10, stat, 100, 1.5, BootstrapMethod::Percentile, 1).is_err());
    assert!(bootstrap_ci(1, stat, 100, 0.95, BootstrapMethod::Bca, 1).is_err());
}

// ---------------------------------------------------------------------------
// cross_val_score
// ---------------------------------------------------------------------------

#[test]
fn cross_val_score_averages_fold_scores() {
    let kf = KFold::new(4).unwrap();
    // Fake "score" that returns the mean of the test-set indices; deterministic.
    let cv = cross_val_score(&kf, 12, |_train, test| {
        Ok(test.iter().sum::<usize>() as f64 / test.len() as f64)
    })
    .unwrap();
    assert_eq!(cv.n_folds, 4);
    // Every fold's score is in [0, 11] because that's the row-index range.
    for s in &cv.scores {
        assert!((0.0..=11.0).contains(s));
    }
    // Mean of the four fold means equals the overall row-index mean (5.5)
    // because the folds are a partition and every fold has three rows.
    assert!((cv.mean() - 5.5).abs() < 1e-12);
}

#[test]
fn cross_val_score_propagates_error() {
    let kf = KFold::new(3).unwrap();
    let result: Result<_, solow_core::Error> = cross_val_score(&kf, 9, |_train, _test| {
        Err(solow_core::Error::Value("boom".into()))
    });
    assert!(result.is_err());
}

#[cfg(feature = "parallel")]
#[test]
fn cross_val_score_parallel_matches_serial() {
    use solow_cv::cross_val_score_parallel;
    let kf = KFold::new(4).unwrap();
    let closure = |_train: &[usize], test: &[usize]| {
        Ok::<f64, solow_core::Error>(test.iter().sum::<usize>() as f64 / test.len() as f64)
    };
    let serial = cross_val_score(&kf, 12, closure).unwrap();
    let parallel = cross_val_score_parallel(&kf, 12, closure).unwrap();
    assert_eq!(serial.scores, parallel.scores);
}

// ---------------------------------------------------------------------------
// GroupKFold
// ---------------------------------------------------------------------------

use solow_cv::{GroupKFold, PurgedKFold, StratifiedGroupKFold};

#[test]
fn group_kfold_keeps_every_group_intact() {
    // 12 rows in 6 groups of 2, 3 folds → each fold gets 2 groups (4 rows).
    let groups = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];
    let gk = GroupKFold::new(3).unwrap();
    let folds = gk.split(&groups).unwrap();
    assert_eq!(folds.len(), 3);
    for fold in &folds {
        assert_eq!(fold.train.len() + fold.test.len(), 12);
        // No group is split between train and test.
        let train_groups: std::collections::HashSet<usize> =
            fold.train.iter().map(|&i| groups[i]).collect();
        let test_groups: std::collections::HashSet<usize> =
            fold.test.iter().map(|&i| groups[i]).collect();
        assert!(train_groups.is_disjoint(&test_groups));
    }
}

#[test]
fn group_kfold_rejects_too_many_splits() {
    let groups = [1, 1, 2, 2];
    assert!(GroupKFold::new(3).unwrap().split(&groups).is_err());
}

#[test]
fn stratified_group_kfold_keeps_prevalence_close() {
    let n = 30usize;
    let y: Vec<usize> = (0..n).map(|i| i % 2).collect();
    // Each group has 3 rows and its own class prevalence.
    let groups: Vec<usize> = (0..n).map(|i| i / 3).collect();
    let sgk = StratifiedGroupKFold::new(3).unwrap();
    let folds = sgk.split(&y, &groups).unwrap();
    assert_eq!(folds.len(), 3);
    let overall_pos = y.iter().filter(|&&v| v == 1).count() as f64 / n as f64;
    for fold in &folds {
        // No group is split.
        let train_groups: std::collections::HashSet<usize> =
            fold.train.iter().map(|&i| groups[i]).collect();
        let test_groups: std::collections::HashSet<usize> =
            fold.test.iter().map(|&i| groups[i]).collect();
        assert!(train_groups.is_disjoint(&test_groups));
        // Per-fold class prevalence within a reasonable band of the overall.
        let pos = fold.test.iter().filter(|&&i| y[i] == 1).count() as f64 / fold.test.len() as f64;
        assert!(
            (pos - overall_pos).abs() < 0.35,
            "fold prevalence {pos} vs overall {overall_pos}"
        );
    }
}

#[test]
fn purged_kfold_drops_purge_and_embargo_from_train() {
    // 20 rows, 4 folds, purge 2, embargo 1. Fold 0 tests [0..5); its purge
    // window on the right is [5..7) and the embargo is [7..8). So train = [8..20).
    let pf = PurgedKFold::new(4, 2, 1).unwrap();
    let folds = pf.split(20).unwrap();
    let f0 = &folds[0];
    assert_eq!(f0.test, (0..5).collect::<Vec<_>>());
    assert_eq!(f0.train, (8..20).collect::<Vec<_>>());
    // Fold 1 tests [5..10); purge left [3..5), purge right [10..12), embargo [12..13).
    // Train = [0..3) ∪ [13..20).
    let f1 = &folds[1];
    let expected_train_1: Vec<usize> = (0..3).chain(13..20).collect();
    assert_eq!(f1.train, expected_train_1);
}

// ---------------------------------------------------------------------------
// CombinatorialPurgedKFold
// ---------------------------------------------------------------------------

use solow_cv::CombinatorialPurgedKFold;

#[test]
fn cpcv_emits_expected_number_of_folds() {
    // C(6, 2) = 15 folds.
    let cpcv = CombinatorialPurgedKFold::new(6, 2, 1, 0).unwrap();
    assert_eq!(cpcv.n_folds(), 15);
    let folds = cpcv.split(30).unwrap();
    assert_eq!(folds.len(), 15);
    // Every test set is the union of two contiguous blocks.
    for fold in &folds {
        assert!(!fold.test.is_empty());
        assert!(!fold.train.is_empty());
        // Train / test disjoint.
        let test_set: std::collections::HashSet<usize> = fold.test.iter().copied().collect();
        for &t in &fold.train {
            assert!(!test_set.contains(&t));
        }
    }
}

#[test]
fn cpcv_rejects_bad_inputs() {
    assert!(CombinatorialPurgedKFold::new(1, 1, 0, 0).is_err()); // n_splits ≥ 2
    assert!(CombinatorialPurgedKFold::new(4, 4, 0, 0).is_err()); // n_test < n_splits
    assert!(CombinatorialPurgedKFold::new(4, 0, 0, 0).is_err()); // n_test ≥ 1
}

#[cfg(feature = "serde")]
#[test]
fn split_round_trips_through_json() {
    let kf = KFold::new(3).unwrap();
    let folds = kf.split(9).unwrap();
    let s = serde_json::to_string(&folds).unwrap();
    let back: Vec<Split> = serde_json::from_str(&s).unwrap();
    assert_eq!(folds, back);
}
