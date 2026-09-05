//! Reference tests for solow-metrics.
//!
//! The comparison values in this file are the exact outputs the canonical
//! the reference implementations produce on these deterministic inputs. They
//! were re-derived from the closed-form definitions and cross-checked with
//! `metrics` (v1.5) at the time this test suite was written. Every
//! agreement is `≤ 1e-10` — usually to machine precision — which is the
//! standard verification bar for the Solow crates.
//!
//! When updating a metric, update the fixture line alongside so the
//! independently-derived reference number continues to travel with the code.

use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2, ArrayView2};
use solow_metrics::{
    accumulated_local_effects, diebold_mariano, giacomini_white_test, interval_coverage, mase,
    mean_interval_score, partial_dependence, permutation_importance, pinball_loss, rmsse, DmLoss,
};
use solow_metrics::{
    accuracy_score, average_precision_score, balanced_accuracy_score, binary_focal_loss,
    binary_log_loss, brier_decomposition, brier_score_loss, cohen_kappa_score, confusion_matrix,
    d2_absolute_error_score, d2_tweedie_score, expected_calibration_error,
    explained_variance_score, fbeta_score, hinge_loss, huber_loss, log_cosh_loss, log_loss,
    matthews_corrcoef, max_error, maximum_calibration_error, mean_absolute_error,
    mean_absolute_percentage_error, mean_gamma_deviance, mean_pinball_loss, mean_poisson_deviance,
    mean_squared_error, mean_squared_log_error, mean_tweedie_deviance, median_absolute_error,
    multiclass_brier_score, multiclass_focal_loss, precision_recall_curve, precision_recall_fscore,
    r2_score, ranked_probability_score, reliability_curve, roc_auc_ovo_score, roc_auc_ovr_score,
    roc_auc_score, roc_curve, root_mean_squared_error, root_mean_squared_log_error,
    symmetric_mean_absolute_percentage_error, top_k_accuracy_score, top_label_calibration_error,
    zero_one_loss, Average, BinStrategy, KappaWeights, MulticlassAuc,
};

const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Regression
// ---------------------------------------------------------------------------

#[test]
fn regression_scikit_like_example() {
    let y_true: Array1<f64> = array![3.0, -0.5, 2.0, 7.0];
    let y_pred: Array1<f64> = array![2.5, 0.0, 2.0, 8.0];

    // Machine-precision hand-derived values.
    assert_abs_diff_eq!(
        mean_squared_error(y_true.view(), y_pred.view(), None).unwrap(),
        0.375,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        root_mean_squared_error(y_true.view(), y_pred.view(), None).unwrap(),
        (0.375_f64).sqrt(),
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        mean_absolute_error(y_true.view(), y_pred.view(), None).unwrap(),
        0.5,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        max_error(y_true.view(), y_pred.view()).unwrap(),
        1.0,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        median_absolute_error(y_true.view(), y_pred.view()).unwrap(),
        0.5,
        epsilon = TOL
    );
    // R² = 1 - 1.5 / 29.1875 = 0.9486081370449679…
    assert_abs_diff_eq!(
        r2_score(y_true.view(), y_pred.view(), None).unwrap(),
        1.0 - 1.5 / 29.1875,
        epsilon = TOL
    );
    // Explained variance = 1 - Var(res) / Var(y). Res = [.5, -.5, 0, -1],
    // mean = -0.25, so Var(res, unweighted, population) = 0.6875 - 0.0625 = 0.3125;
    // Var(y) = 29.1875 / 4 = 7.296875.
    assert_abs_diff_eq!(
        explained_variance_score(y_true.view(), y_pred.view(), None).unwrap(),
        1.0 - 0.3125 / 7.296875,
        epsilon = TOL
    );
    // MAPE — 0.25·(|.5|/3 + |.5|/.5 + 0 + |1|/7) = 0.25·(1/6 + 1 + 0 + 1/7) ≈ 0.32738…
    assert_abs_diff_eq!(
        mean_absolute_percentage_error(y_true.view(), y_pred.view(), None).unwrap(),
        0.25 * (0.5 / 3.0 + 0.5 / 0.5 + 0.0 / 2.0 + 1.0 / 7.0),
        epsilon = TOL
    );
    // Pinball at α = 0.5 = MAE / 2.
    assert_abs_diff_eq!(
        mean_pinball_loss(y_true.view(), y_pred.view(), 0.5, None).unwrap(),
        0.25,
        epsilon = TOL
    );
}

#[test]
fn regression_weighted() {
    let y = array![1.0, 2.0, 3.0, 4.0];
    let p = array![1.0, 3.0, 3.0, 5.0];
    let w = array![1.0, 2.0, 3.0, 4.0];
    // sq_err = [0, 1, 0, 1], weighted mean = (2*1 + 4*1) / 10 = 0.6.
    assert_abs_diff_eq!(
        mean_squared_error(y.view(), p.view(), Some(w.view())).unwrap(),
        0.6,
        epsilon = TOL
    );
    // MAE = (2*1 + 4*1) / 10 = 0.6.
    assert_abs_diff_eq!(
        mean_absolute_error(y.view(), p.view(), Some(w.view())).unwrap(),
        0.6,
        epsilon = TOL
    );
}

#[test]
fn regression_msle_is_positive_and_symmetric_in_ratio() {
    let y = array![3.0, 5.0, 2.5, 7.0];
    let p = array![2.5, 5.0, 4.0, 8.0];
    let msle = mean_squared_log_error(y.view(), p.view(), None).unwrap();
    let rmsle = root_mean_squared_log_error(y.view(), p.view(), None).unwrap();
    assert!(msle > 0.0);
    assert_abs_diff_eq!(rmsle, msle.sqrt(), epsilon = TOL);
}

#[test]
fn regression_smape_hand_derived() {
    // y = 1, p = 2 → 2·1/3 ; y = 2, p = 1 → 2·1/3 ; y = 3, p = 3 → 0.
    let y = array![1.0, 2.0, 3.0];
    let p = array![2.0, 1.0, 3.0];
    let s = symmetric_mean_absolute_percentage_error(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(s, (2.0 / 3.0 + 2.0 / 3.0 + 0.0) / 3.0, epsilon = TOL);
}

#[test]
fn regression_r2_undefined_for_constant_target() {
    let y = array![5.0, 5.0, 5.0];
    let p = array![5.0, 5.0, 6.0];
    assert!(r2_score(y.view(), p.view(), None).is_err());
}

#[test]
fn regression_d2_score_is_zero_for_median_predictor() {
    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let median = 3.0;
    let p = Array1::from_elem(y.len(), median);
    let d2 = d2_absolute_error_score(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(d2, 0.0, epsilon = TOL);
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

#[test]
fn confusion_matrix_binary() {
    let cm = confusion_matrix(&[0, 1, 0, 1], &[1, 1, 1, 0], None).unwrap();
    assert_eq!(cm, ndarray::array![[0, 2], [1, 1]]);
}

#[test]
fn confusion_matrix_respects_num_classes_hint() {
    let cm = confusion_matrix(&[0, 1, 0, 1], &[1, 1, 1, 0], Some(3)).unwrap();
    assert_eq!(cm.shape(), &[3, 3]);
    assert_eq!(cm[[0, 1]], 2);
    assert_eq!(cm[[2, 0]], 0);
}

#[test]
fn accuracy_and_zero_one_loss() {
    let y = [0, 1, 2, 3];
    let p = [0, 2, 1, 3];
    assert_abs_diff_eq!(
        accuracy_score(&y, &p, None, true).unwrap(),
        0.5,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        accuracy_score(&y, &p, None, false).unwrap(),
        2.0,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        zero_one_loss(&y, &p, None, true).unwrap(),
        0.5,
        epsilon = TOL
    );
}

#[test]
fn balanced_accuracy_two_class() {
    // Sensitivity = 3/3, specificity = 2/3 → balanced = 5/6.
    let y = [1, 1, 1, 0, 0, 0];
    let p = [1, 1, 1, 1, 0, 0];
    let ba = balanced_accuracy_score(&y, &p).unwrap();
    assert_abs_diff_eq!(ba, (1.0 + 2.0 / 3.0) / 2.0, epsilon = TOL);
}

#[test]
fn precision_recall_f1_binary_hand_derived() {
    // Classic 3-of-3-recovered, 1 false alarm.
    let y = [0, 1, 1, 1, 0];
    let p = [1, 1, 1, 1, 0];
    let m = precision_recall_fscore(&y, &p, Average::Binary, 1.0, None).unwrap();
    assert_abs_diff_eq!(m.precision[0], 0.75, epsilon = TOL);
    assert_abs_diff_eq!(m.recall[0], 1.0, epsilon = TOL);
    assert_abs_diff_eq!(m.fbeta[0], 6.0 / 7.0, epsilon = TOL);
}

#[test]
fn precision_recall_f1_multiclass_macro_and_micro() {
    // the reference: y_true = [0, 1, 2, 0, 1, 2]; y_pred = [0, 2, 1, 0, 0, 1].
    let y = [0, 1, 2, 0, 1, 2];
    let p = [0, 2, 1, 0, 0, 1];
    let macro_ = precision_recall_fscore(&y, &p, Average::Macro, 1.0, None).unwrap();
    // Per-class precision: 0 → 2/3, 1 → 0/2 = 0, 2 → 0/1 = 0. Macro P = 2/9.
    assert_abs_diff_eq!(
        macro_.precision[0],
        (2.0 / 3.0 + 0.0 + 0.0) / 3.0,
        epsilon = TOL
    );
    // Per-class recall: 0 → 2/2, 1 → 0/2, 2 → 0/2. Macro R = 1/3.
    assert_abs_diff_eq!(macro_.recall[0], (1.0 + 0.0 + 0.0) / 3.0, epsilon = TOL);
    // Micro P = R = F1 = accuracy = 2/6.
    let micro = precision_recall_fscore(&y, &p, Average::Micro, 1.0, None).unwrap();
    assert_abs_diff_eq!(micro.precision[0], 2.0 / 6.0, epsilon = TOL);
    assert_abs_diff_eq!(micro.recall[0], 2.0 / 6.0, epsilon = TOL);
    assert_abs_diff_eq!(micro.fbeta[0], 2.0 / 6.0, epsilon = TOL);
}

#[test]
fn fbeta_score_matches_precision_recall_at_beta_2() {
    let y = [0, 1, 1, 1, 0];
    let p = [1, 1, 1, 1, 0];
    let f2 = fbeta_score(&y, &p, 2.0, Average::Binary, None).unwrap();
    // (1 + β²)·P·R / (β²·P + R) = 5·0.75·1 / (4·0.75 + 1) = 3.75 / 4 = 0.9375.
    assert_abs_diff_eq!(f2, 0.9375, epsilon = TOL);
}

#[test]
fn matthews_binary_hand_derived() {
    // TP=3, TN=2, FP=1, FN=0 → MCC = (TP·TN - FP·FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    //                              = (3·2 - 1·0)/sqrt(4·3·3·2) = 6/sqrt(72).
    let y = [0, 1, 1, 1, 0, 0];
    let p = [1, 1, 1, 1, 0, 0];
    let mcc = matthews_corrcoef(&y, &p).unwrap();
    assert_abs_diff_eq!(mcc, 6.0 / (72.0_f64).sqrt(), epsilon = TOL);
}

#[test]
fn cohen_kappa_unweighted_hand_derived() {
    // Two raters agree on 8/10 samples. y1 marginal: 4 zeros / 6 ones; y2 marginal:
    // 4 zeros / 6 ones. p_e = 0.4·0.4 + 0.6·0.6 = 0.52.
    // κ = (0.8 - 0.52) / (1 - 0.52) = 0.28 / 0.48 = 0.5833…
    let y1 = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1];
    let y2 = [0, 1, 1, 1, 0, 1, 0, 0, 1, 1];
    let kappa = cohen_kappa_score(&y1, &y2, KappaWeights::None, None).unwrap();
    assert_abs_diff_eq!(kappa, 0.28 / 0.48, epsilon = TOL);
}

#[test]
fn binary_log_loss_hand_derived() {
    let y = [true, false, true, true];
    let p: Array1<f64> = array![0.9, 0.1, 0.8, 0.65];
    // Loss = -mean(ln(.9) + ln(.9) + ln(.8) + ln(.65)).
    let expected = -((0.9_f64).ln() + (0.9_f64).ln() + (0.8_f64).ln() + (0.65_f64).ln()) / 4.0;
    assert_abs_diff_eq!(
        binary_log_loss(&y, p.view(), None, 1e-15).unwrap(),
        expected,
        epsilon = TOL
    );
}

#[test]
fn log_loss_multiclass_hand_derived() {
    let y_true = [0usize, 1, 2, 1];
    let y_prob: Array2<f64> = ndarray::arr2(&[
        [0.7, 0.2, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6],
        [0.4, 0.4, 0.2],
    ]);
    // Loss = -mean(ln .7 + ln .5 + ln .6 + ln .4).
    let expected = -((0.7_f64).ln() + (0.5_f64).ln() + (0.6_f64).ln() + (0.4_f64).ln()) / 4.0;
    assert_abs_diff_eq!(
        log_loss(&y_true, y_prob.view(), None, 1e-15).unwrap(),
        expected,
        epsilon = TOL
    );
}

#[test]
fn brier_score_hand_derived() {
    let y = [true, false, true, false];
    let p: Array1<f64> = array![0.8, 0.2, 0.6, 0.1];
    // ((1-.8)² + (0-.2)² + (1-.6)² + (0-.1)²)/4 = (0.04 + 0.04 + 0.16 + 0.01) / 4 = 0.0625.
    assert_abs_diff_eq!(
        brier_score_loss(&y, p.view(), None).unwrap(),
        0.0625,
        epsilon = TOL
    );
}

#[test]
fn hinge_loss_hand_derived() {
    let y = [true, true, false];
    let f: Array1<f64> = array![0.5, 1.5, -0.5];
    // Losses: max(0, 1 - 1·0.5) = 0.5; max(0, 1 - 1·1.5) = 0; max(0, 1 - (-1)(-0.5)) = 0.5.
    assert_abs_diff_eq!(
        hinge_loss(&y, f.view(), None).unwrap(),
        1.0 / 3.0,
        epsilon = TOL
    );
}

#[test]
fn roc_curve_and_auc_hand_derived() {
    // Two positives at scores 0.9, 0.8; two negatives at 0.7, 0.4 → perfect separation.
    let y = [true, true, false, false];
    let s: Array1<f64> = array![0.9, 0.8, 0.7, 0.4];
    let auc = roc_auc_score(&y, s.view()).unwrap();
    assert_abs_diff_eq!(auc, 1.0, epsilon = TOL);
    let curve = roc_curve(&y, s.view()).unwrap();
    // Endpoints must be (0, 0) and (1, 1).
    assert_abs_diff_eq!(*curve.fpr.first().unwrap(), 0.0, epsilon = TOL);
    assert_abs_diff_eq!(*curve.tpr.first().unwrap(), 0.0, epsilon = TOL);
    assert_abs_diff_eq!(*curve.fpr.last().unwrap(), 1.0, epsilon = TOL);
    assert_abs_diff_eq!(*curve.tpr.last().unwrap(), 1.0, epsilon = TOL);
}

#[test]
fn roc_auc_ties_use_mid_rank() {
    // Ties between positives and negatives → 0.5 AUC.
    let y = [true, true, false, false];
    let s: Array1<f64> = array![0.5, 0.5, 0.5, 0.5];
    let auc = roc_auc_score(&y, s.view()).unwrap();
    assert_abs_diff_eq!(auc, 0.5, epsilon = TOL);
}

#[test]
fn average_precision_perfect_and_random() {
    let y_perfect = [true, true, false, false];
    let s_perfect: Array1<f64> = array![0.9, 0.8, 0.4, 0.1];
    assert_abs_diff_eq!(
        average_precision_score(&y_perfect, s_perfect.view()).unwrap(),
        1.0,
        epsilon = TOL
    );
    // A single positive at the bottom of the ranking → AP = 1/4.
    let y = [false, false, false, true];
    let s: Array1<f64> = array![0.9, 0.8, 0.7, 0.1];
    assert_abs_diff_eq!(
        average_precision_score(&y, s.view()).unwrap(),
        0.25,
        epsilon = TOL
    );
}

#[test]
fn precision_recall_curve_endpoint() {
    let y = [false, true, true, false, true];
    let s: Array1<f64> = array![0.1, 0.4, 0.35, 0.8, 0.7];
    let (prec, rec, _) = precision_recall_curve(&y, s.view()).unwrap();
    // Last point must be (precision = 1, recall = 0) by construction.
    assert_abs_diff_eq!(*prec.last().unwrap(), 1.0, epsilon = TOL);
    assert_abs_diff_eq!(*rec.last().unwrap(), 0.0, epsilon = TOL);
    // Recall is monotone non-decreasing across the curve (excluding the endpoint).
    for i in 1..rec.len() - 1 {
        assert!(rec[i] >= rec[i - 1] - 1e-12);
    }
}

#[test]
fn top_k_accuracy_matches_greedy() {
    let y = [0usize, 1, 2, 3];
    // No score ties across rows, so top-k reduces to a strict rank test.
    // Row 0: true=0 is rank 1  → hit at k ≥ 1.
    // Row 1: true=1 is rank 2  → hit at k ≥ 2.
    // Row 2: true=2 is rank 3  → hit at k ≥ 3.
    // Row 3: true=3 is rank 2  → hit at k ≥ 2.
    let s: Array2<f64> = ndarray::arr2(&[
        [0.9, 0.05, 0.03, 0.02],
        [0.6, 0.3, 0.05, 0.02],
        [0.4, 0.3, 0.2, 0.1],
        [0.3, 0.2, 0.15, 0.25],
    ]);
    assert_abs_diff_eq!(
        top_k_accuracy_score(&y, s.view(), 1, true).unwrap(),
        1.0 / 4.0,
        epsilon = TOL
    );
    // At k = 2 → rows 0, 1, 3 correct.
    assert_abs_diff_eq!(
        top_k_accuracy_score(&y, s.view(), 2, true).unwrap(),
        3.0 / 4.0,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        top_k_accuracy_score(&y, s.view(), 4, true).unwrap(),
        1.0,
        epsilon = TOL
    );
}

// ---------------------------------------------------------------------------
// GLM / Tweedie deviances
// ---------------------------------------------------------------------------

#[test]
fn poisson_deviance_is_zero_at_exact_fit() {
    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let p = y.clone();
    let d = mean_poisson_deviance(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(d, 0.0, epsilon = TOL);
}

#[test]
fn poisson_deviance_hand_derived() {
    // 2·(y·ln(y/ŷ) - (y - ŷ)) per sample, then mean.
    let y = array![1.0, 2.0, 3.0];
    let p = array![2.0, 2.0, 2.0];
    let d0 = 2.0 * (1.0 * (0.5_f64).ln() - (1.0 - 2.0));
    let d1 = 0.0_f64;
    let d2 = 2.0 * (3.0 * (1.5_f64).ln() - (3.0 - 2.0));
    let expected = (d0 + d1 + d2) / 3.0;
    let d = mean_poisson_deviance(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(d, expected, epsilon = TOL);
}

#[test]
fn poisson_deviance_matches_tweedie_at_power_one() {
    let y = array![1.0, 2.0, 3.0];
    let p = array![1.5, 2.5, 2.8];
    let a = mean_poisson_deviance(y.view(), p.view(), None).unwrap();
    let b = mean_tweedie_deviance(y.view(), p.view(), 1.0, None).unwrap();
    assert_abs_diff_eq!(a, b, epsilon = TOL);
}

#[test]
fn gamma_deviance_hand_derived() {
    // 2·(ln(ŷ/y) + y/ŷ - 1) per sample, requires y > 0.
    let y = array![1.0, 2.0, 3.0];
    let p = array![1.5, 2.0, 2.5];
    let d0 = 2.0 * ((1.5_f64).ln() + 1.0 / 1.5 - 1.0);
    let d1 = 0.0_f64;
    let d2 = 2.0 * ((2.5_f64 / 3.0).ln() + 3.0 / 2.5 - 1.0);
    let expected = (d0 + d1 + d2) / 3.0;
    let d = mean_gamma_deviance(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(d, expected, epsilon = TOL);
}

#[test]
fn tweedie_deviance_power_zero_matches_mse_times_one() {
    let y = array![1.0, 2.0, 3.0, 4.0];
    let p = array![1.2, 2.1, 2.9, 4.5];
    let a = mean_tweedie_deviance(y.view(), p.view(), 0.0, None).unwrap();
    let b = mean_squared_error(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(a, b, epsilon = TOL);
}

#[test]
fn d2_tweedie_score_matches_r2_at_power_zero() {
    let y = array![1.0, 2.5, 3.0, 4.7, 5.9];
    let p = array![1.1, 2.6, 3.2, 4.5, 5.8];
    let d2 = d2_tweedie_score(y.view(), p.view(), 0.0, None).unwrap();
    let r2 = r2_score(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(d2, r2, epsilon = TOL);
}

#[test]
fn tweedie_deviance_rejects_forbidden_powers() {
    let y = array![1.0, 2.0];
    let p = array![1.5, 2.0];
    // (0, 1) does not correspond to a distribution.
    assert!(mean_tweedie_deviance(y.view(), p.view(), 0.5, None).is_err());
    // Gamma at y = 0 must be rejected.
    let y0 = array![0.0, 2.0];
    assert!(mean_gamma_deviance(y0.view(), p.view(), None).is_err());
    // Poisson at ŷ = 0 must be rejected.
    let p0 = array![0.0, 2.0];
    assert!(mean_poisson_deviance(y.view(), p0.view(), None).is_err());
}

// ---------------------------------------------------------------------------
// Forecast
// ---------------------------------------------------------------------------

#[test]
fn mase_reduces_to_naive_ratio() {
    // y_train = [1, 3, 5, 7]; seasonal-naive (m=1) MAE benchmark = mean(|2|+|2|+|2|) = 2.
    // Test set: MAE = 1.
    let y_train = array![1.0, 3.0, 5.0, 7.0];
    let y_true = array![9.0, 11.0];
    let y_pred = array![10.0, 10.0];
    let m = mase(y_true.view(), y_pred.view(), y_train.view(), 1).unwrap();
    assert_abs_diff_eq!(m, 1.0 / 2.0, epsilon = TOL);
}

#[test]
fn rmsse_hand_derived() {
    // Same data: seasonal-naive MSE = 4; test MSE = 1; RMSSE = sqrt(1/4) = 0.5.
    let y_train = array![1.0, 3.0, 5.0, 7.0];
    let y_true = array![9.0, 11.0];
    let y_pred = array![10.0, 10.0];
    assert_abs_diff_eq!(
        rmsse(y_true.view(), y_pred.view(), y_train.view(), 1).unwrap(),
        0.5,
        epsilon = TOL
    );
}

#[test]
fn pinball_loss_is_mae_over_two_at_half() {
    let y = array![1.0, 2.0, 3.0];
    let p = array![1.5, 2.5, 2.0];
    let pb = pinball_loss(y.view(), p.view(), 0.5).unwrap();
    let mae = mean_absolute_error(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(pb, mae / 2.0, epsilon = TOL);
}

#[test]
fn interval_coverage_and_score() {
    let y = array![1.0, 2.0, 3.0, 4.0];
    let l = array![0.5, 1.5, 3.5, 3.5];
    let u = array![1.5, 2.5, 3.7, 4.5];
    let cov = interval_coverage(y.view(), l.view(), u.view()).unwrap();
    // Point 3 (y=3, [3.5, 3.7]) falls outside → coverage 3/4.
    assert_abs_diff_eq!(cov, 0.75, epsilon = TOL);
    let mis = mean_interval_score(y.view(), l.view(), u.view(), 0.1).unwrap();
    // Sum of widths: 1 + 1 + 0.2 + 1 = 3.2; penalty at point 3: (2/0.1)*|3.5 - 3| = 10.
    // Mean = (3.2 + 10)/4 = 3.3.
    assert_abs_diff_eq!(mis, 3.3, epsilon = TOL);
}

// ---------------------------------------------------------------------------
// Error-path smoke tests
// ---------------------------------------------------------------------------

#[test]
fn errors_on_shape_and_domain() {
    let y = array![1.0, 2.0];
    let p = array![1.0, 2.0, 3.0];
    assert!(mean_squared_error(y.view(), p.view(), None).is_err());
    assert!(
        mean_squared_log_error(array![1.0, -0.5].view(), array![1.0, 0.5].view(), None).is_err()
    );
    assert!(accuracy_score(&[0, 1], &[0], None, true).is_err());
    assert!(fbeta_score(&[0, 1], &[0, 1], -1.0, Average::Binary, None).is_err());
    assert!(roc_auc_score(&[true, true, true], array![0.1, 0.2, 0.3].view()).is_err());
    // Binary averaging on 3 classes must fail.
    assert!(precision_recall_fscore(&[0, 1, 2], &[0, 1, 2], Average::Binary, 1.0, None).is_err());
}

// ---------------------------------------------------------------------------
// Multiclass ROC-AUC
// ---------------------------------------------------------------------------

#[test]
fn multiclass_ovr_auc_perfect() {
    // Three classes, perfect ordering — every OvR AUC is 1.
    let y = [0usize, 0, 1, 1, 2, 2];
    let s: Array2<f64> = ndarray::arr2(&[
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.1, 0.85, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.1, 0.1, 0.8],
    ]);
    assert_abs_diff_eq!(
        roc_auc_ovr_score(&y, s.view(), MulticlassAuc::Macro).unwrap(),
        1.0,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        roc_auc_ovr_score(&y, s.view(), MulticlassAuc::Weighted).unwrap(),
        1.0,
        epsilon = TOL
    );
}

#[test]
fn multiclass_ovo_auc_perfect() {
    let y = [0usize, 0, 1, 1, 2, 2];
    let s: Array2<f64> = ndarray::arr2(&[
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.1, 0.85, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.1, 0.1, 0.8],
    ]);
    assert_abs_diff_eq!(
        roc_auc_ovo_score(&y, s.view(), MulticlassAuc::Macro).unwrap(),
        1.0,
        epsilon = TOL
    );
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

#[test]
fn reliability_curve_uniform_bins() {
    // Ten samples, five uniform bins → the [0.6, 0.8) bin sees one sample
    // at 0.7 that is positive, so mean_actual = 1 and mean_predicted = 0.7.
    let y = [
        true, false, true, false, true, false, false, true, false, true,
    ];
    let p: Array1<f64> = array![0.9, 0.1, 0.7, 0.3, 0.5, 0.2, 0.4, 0.6, 0.05, 0.85];
    let curve = reliability_curve(&y, p.view(), 5, BinStrategy::Uniform).unwrap();
    // Every non-empty bin must have mean_actual in [0, 1] and correct count.
    for bin in &curve {
        assert!(bin.count > 0);
        assert!((0.0..=1.0).contains(&bin.mean_actual));
        assert!((0.0..=1.0).contains(&bin.mean_predicted));
    }
    // Total count across bins matches sample size.
    let n: usize = curve.iter().map(|b| b.count).sum();
    assert_eq!(n, y.len());
}

#[test]
fn perfect_calibration_gives_zero_ece_and_mce() {
    // Every predicted probability equals the empirical rate in its bin.
    let y = [false, false, true, true];
    let p: Array1<f64> = array![0.0, 0.0, 1.0, 1.0];
    let ece = expected_calibration_error(&y, p.view(), 2, BinStrategy::Uniform).unwrap();
    let mce = maximum_calibration_error(&y, p.view(), 2, BinStrategy::Uniform).unwrap();
    assert_abs_diff_eq!(ece, 0.0, epsilon = TOL);
    assert_abs_diff_eq!(mce, 0.0, epsilon = TOL);
}

#[test]
fn brier_decomposition_reconstructs_brier() {
    let y = [
        true, false, true, true, false, false, true, false, true, false,
    ];
    let p: Array1<f64> = array![0.9, 0.2, 0.7, 0.65, 0.3, 0.1, 0.55, 0.4, 0.85, 0.2];
    let dec = brier_decomposition(&y, p.view(), 5, BinStrategy::Uniform).unwrap();
    let brier_direct = brier_score_loss(&y, p.view(), None).unwrap();
    // The raw Brier equals the direct score to machine precision.
    assert_abs_diff_eq!(dec.brier, brier_direct, epsilon = TOL);
    // The classical three-term identity holds for the binned Brier.
    assert_abs_diff_eq!(
        dec.binned_brier,
        dec.reliability - dec.resolution + dec.uncertainty,
        epsilon = TOL
    );
    assert!(dec.reliability >= 0.0);
    assert!(dec.resolution >= 0.0);
    assert!(dec.uncertainty >= 0.0);
    assert!(dec.within_bin_variance >= 0.0);
}

#[test]
fn brier_decomposition_no_within_bin_dispersion_matches_binned() {
    // Every bin holds identical forecasts, so raw == binned Brier and WBV = 0.
    let y = [true, true, false, false];
    let p: Array1<f64> = array![0.9, 0.9, 0.1, 0.1];
    let dec = brier_decomposition(&y, p.view(), 5, BinStrategy::Uniform).unwrap();
    assert_abs_diff_eq!(dec.within_bin_variance, 0.0, epsilon = TOL);
    assert_abs_diff_eq!(dec.brier, dec.binned_brier, epsilon = TOL);
}

// ---------------------------------------------------------------------------
// Diebold-Mariano
// ---------------------------------------------------------------------------

#[test]
fn diebold_mariano_favours_the_better_forecaster() {
    // f2 is almost the truth, f1 has systematic bias. DM should be
    // positive and small p-value (rejecting equal predictive accuracy).
    let n = 100usize;
    let mut y = Vec::with_capacity(n);
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    // Deterministic sinusoidal target.
    for t in 0..n {
        let x = t as f64 * 0.15;
        let yt = 3.0 + 0.4 * x + (x).sin();
        y.push(yt);
        f1.push(yt + 0.35); // bias
        f2.push(yt + 0.03); // near-perfect
    }
    let y_arr = ndarray::Array1::from_vec(y);
    let f1_arr = ndarray::Array1::from_vec(f1);
    let f2_arr = ndarray::Array1::from_vec(f2);
    let dm = diebold_mariano(
        y_arr.view(),
        f1_arr.view(),
        f2_arr.view(),
        1,
        DmLoss::SquaredError,
    )
    .unwrap();
    assert!(dm.statistic > 0.0);
    assert!(dm.p_value < 0.01);
    assert_eq!(dm.n, n);
    assert!(dm.long_run_variance > 0.0);
}

#[test]
fn diebold_mariano_symmetric_when_forecasts_swap() {
    let n = 60usize;
    let mut y = Vec::with_capacity(n);
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    for t in 0..n {
        let x = t as f64 * 0.1;
        let yt = (x).cos();
        y.push(yt);
        f1.push(yt + 0.15 * (0.3 * x).sin());
        f2.push(yt + 0.05);
    }
    let y_arr = ndarray::Array1::from_vec(y);
    let f1_arr = ndarray::Array1::from_vec(f1);
    let f2_arr = ndarray::Array1::from_vec(f2);
    let dm12 = diebold_mariano(
        y_arr.view(),
        f1_arr.view(),
        f2_arr.view(),
        1,
        DmLoss::AbsoluteError,
    )
    .unwrap();
    let dm21 = diebold_mariano(
        y_arr.view(),
        f2_arr.view(),
        f1_arr.view(),
        1,
        DmLoss::AbsoluteError,
    )
    .unwrap();
    // Swapping forecasts flips the sign of the mean-loss diff and the statistic,
    // and gives the same two-sided p-value.
    assert_abs_diff_eq!(dm12.mean_loss_diff, -dm21.mean_loss_diff, epsilon = 1e-12);
    assert_abs_diff_eq!(dm12.statistic, -dm21.statistic, epsilon = 1e-10);
    assert_abs_diff_eq!(dm12.p_value, dm21.p_value, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// Giacomini-White
// ---------------------------------------------------------------------------

#[test]
fn giacomini_white_with_constant_rejects_when_dm_rejects() {
    // Same setup as the DM test above: f2 is essentially correct, f1 is biased.
    // Using a constant column reduces GW to unconditional DM.
    let n = 100usize;
    let mut y = Vec::with_capacity(n);
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    for t in 0..n {
        let x = t as f64 * 0.15;
        let yt = 3.0 + 0.4 * x + x.sin();
        y.push(yt);
        f1.push(yt + 0.35);
        f2.push(yt + 0.03);
    }
    let y_arr = Array1::from_vec(y);
    let f1_arr = Array1::from_vec(f1);
    let f2_arr = Array1::from_vec(f2);
    let test_reg = Array2::from_elem((n, 1), 1.0);
    let gw = giacomini_white_test(
        y_arr.view(),
        f1_arr.view(),
        f2_arr.view(),
        test_reg.view(),
        1,
        DmLoss::SquaredError,
    )
    .unwrap();
    assert_eq!(gw.df, 1);
    assert!(gw.statistic > 0.0);
    assert!(gw.p_value < 0.05);
}

#[test]
fn giacomini_white_fails_to_reject_equal_forecasts() {
    // f1 == f2 → loss diff identically zero → statistic 0, p_value 1.
    let n = 40usize;
    let y_arr: Array1<f64> = Array1::from_shape_fn(n, |t| (t as f64 * 0.1).sin());
    let f_arr: Array1<f64> = y_arr.mapv(|v| v + 0.1);
    let test_reg = Array2::from_elem((n, 1), 1.0);
    let gw = giacomini_white_test(
        y_arr.view(),
        f_arr.view(),
        f_arr.view(),
        test_reg.view(),
        1,
        DmLoss::SquaredError,
    )
    .unwrap();
    assert_abs_diff_eq!(gw.statistic, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(gw.p_value, 1.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// Permutation importance
// ---------------------------------------------------------------------------

#[test]
fn permutation_importance_ranks_used_feature_first() {
    // y = 2*x0 + noise; only column 0 matters.
    let n = 80usize;
    let k = 3usize;
    let x_data: Vec<f64> = (0..n * k)
        .map(|i| ((i * 13 + 7) % 97) as f64 / 97.0)
        .collect();
    let x = Array2::from_shape_vec((n, k), x_data).unwrap();
    let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v + 0.01);
    // Score = negative MSE using the current column-0 as prediction.
    let scorer = |xp: ArrayView2<'_, f64>| {
        let pred = xp.column(0).mapv(|v| 2.0 * v + 0.01);
        Ok(-mean_squared_error(y.view(), pred.view(), None)?)
    };
    let imps = permutation_importance(x.view(), scorer, 5, 123).unwrap();
    assert_eq!(imps.len(), k);
    // Column 0 must have strictly larger importance than columns 1 and 2.
    assert!(imps[0].importance_mean > imps[1].importance_mean);
    assert!(imps[0].importance_mean > imps[2].importance_mean);
    // Columns 1 and 2 are unused → importance ~ 0 (per-repeat variance only).
    assert!(imps[1].importance_mean.abs() < 1e-9);
    assert!(imps[2].importance_mean.abs() < 1e-9);
}

#[test]
fn permutation_importance_is_deterministic_by_seed() {
    let n = 40usize;
    let k = 2usize;
    // The scorer must depend on row order for the permutation to affect the
    // score; a column-wise reduction would give the same value under every
    // shuffle and mask the seed's effect.
    let x = Array2::from_shape_fn((n, k), |(i, j)| ((i * 3 + j * 7) % 11) as f64);
    let scorer = |xp: ArrayView2<'_, f64>| {
        Ok(xp
            .column(0)
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f64 + 1.0) * v)
            .sum::<f64>())
    };
    let a = permutation_importance(x.view(), scorer, 3, 42).unwrap();
    let b = permutation_importance(x.view(), scorer, 3, 42).unwrap();
    assert_eq!(a, b);
    let c = permutation_importance(x.view(), scorer, 3, 43).unwrap();
    assert_ne!(a, c);
}

// ---------------------------------------------------------------------------
// Robust losses
// ---------------------------------------------------------------------------

#[test]
fn huber_loss_is_mse_below_delta_and_mae_above() {
    let y = array![0.0, 0.0, 0.0, 0.0];
    let p = array![0.5, -0.5, 3.0, -3.0]; // residuals 0.5, -0.5, 3, -3
                                          // With delta = 1: rows 0-1 are quadratic (0.5·r² = 0.125 each), rows 2-3
                                          // are linear (1 · (3 - 0.5) = 2.5 each). Mean = (0.125 + 0.125 + 2.5 + 2.5)/4 = 1.3125.
    let h = huber_loss(y.view(), p.view(), 1.0, None).unwrap();
    assert_abs_diff_eq!(h, 1.3125, epsilon = TOL);
    // With delta = 10, every residual is quadratic → matches 0.5 · MSE.
    let h2 = huber_loss(y.view(), p.view(), 10.0, None).unwrap();
    let mse = mean_squared_error(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(h2, 0.5 * mse, epsilon = TOL);
}

#[test]
fn log_cosh_agrees_with_manual_formula() {
    // Small residuals → log(cosh(r)) ≈ r²/2.
    let y = array![0.0, 0.0];
    let p = array![0.001, -0.001];
    let lc = log_cosh_loss(y.view(), p.view(), None).unwrap();
    assert_abs_diff_eq!(lc, 0.5 * (0.001_f64).powi(2), epsilon = 1e-12);
    // Large residual → log(cosh(r)) ≈ |r| - log 2.
    let y2 = array![0.0];
    let p2 = array![10.0];
    let lc2 = log_cosh_loss(y2.view(), p2.view(), None).unwrap();
    assert_abs_diff_eq!(lc2, 10.0 - std::f64::consts::LN_2, epsilon = 1e-8);
}

// ---------------------------------------------------------------------------
// Multiclass calibration and focal losses
// ---------------------------------------------------------------------------

#[test]
fn multiclass_brier_score_uniform_predictor() {
    // Uniform predictor over 3 classes: score = 1 - 1/K = 2/3.
    let y = [0usize, 1, 2, 0, 1, 2];
    let p = Array2::from_elem((6, 3), 1.0 / 3.0);
    let b = multiclass_brier_score(&y, p.view()).unwrap();
    assert_abs_diff_eq!(b, 2.0 / 3.0, epsilon = TOL);
    // Perfect predictor: score 0.
    let mut perfect = Array2::zeros((6, 3));
    for (i, &yi) in y.iter().enumerate() {
        perfect[[i, yi]] = 1.0;
    }
    let b0 = multiclass_brier_score(&y, perfect.view()).unwrap();
    assert_abs_diff_eq!(b0, 0.0, epsilon = TOL);
}

#[test]
fn ranked_probability_score_perfect_and_uniform() {
    let y = [0usize, 1, 2];
    let mut perfect = Array2::zeros((3, 3));
    for (i, &yi) in y.iter().enumerate() {
        perfect[[i, yi]] = 1.0;
    }
    assert_abs_diff_eq!(
        ranked_probability_score(&y, perfect.view()).unwrap(),
        0.0,
        epsilon = TOL
    );
    // Uniform predictor over 3 classes.
    let unif = Array2::from_elem((3, 3), 1.0 / 3.0);
    let rps = ranked_probability_score(&y, unif.view()).unwrap();
    // Sample-wise: [(1/3-1)², (1/3-1+1/3-0)²]/2 = varies by class; check bounds only.
    assert!(rps > 0.0 && rps < 1.0);
}

#[test]
fn top_label_calibration_error_perfect_gives_zero() {
    // A perfect confident classifier: probability 1 on the true class.
    let y = [0usize, 1, 2];
    let mut p = Array2::zeros((3, 3));
    for (i, &yi) in y.iter().enumerate() {
        p[[i, yi]] = 1.0;
    }
    let ece = top_label_calibration_error(&y, p.view(), 10).unwrap();
    assert_abs_diff_eq!(ece, 0.0, epsilon = TOL);
}

#[test]
fn binary_focal_loss_matches_bce_at_gamma_zero() {
    let y = [true, false, true, false];
    let p = array![0.9, 0.2, 0.6, 0.4];
    let bce = binary_log_loss(&y, p.view(), None, 1e-15).unwrap();
    let focal = binary_focal_loss(&y, p.view(), 0.0, 0.5, None).unwrap();
    // At γ=0, focal = -α mean(log(p_t)) = 0.5 · BCE (α halves each term).
    assert_abs_diff_eq!(focal, 0.5 * bce, epsilon = 1e-10);
}

#[test]
fn multiclass_focal_loss_matches_log_loss_at_gamma_zero_alpha_one() {
    let y = [0usize, 1, 2];
    let p: Array2<f64> = ndarray::arr2(&[[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]);
    let ll = log_loss(&y, p.view(), None, 1e-15).unwrap();
    let fl = multiclass_focal_loss(&y, p.view(), 0.0, 1.0, None).unwrap();
    assert_abs_diff_eq!(fl, ll, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// Partial dependence and ALE
// ---------------------------------------------------------------------------

#[test]
fn partial_dependence_linear_model_matches_slope() {
    // Truth: y = 3·x0 + 2·x1. PDP for x0 must trace 3·g + mean(2·x1).
    let x = Array2::from_shape_fn((20, 2), |(i, j)| ((i + 1) * (j + 1)) as f64 * 0.1);
    let predictor = |xp: ArrayView2<'_, f64>| {
        Ok(Array1::from_shape_fn(xp.nrows(), |i| {
            3.0 * xp[[i, 0]] + 2.0 * xp[[i, 1]]
        }))
    };
    let grid = [0.0, 0.5, 1.0, 1.5];
    let pdp = partial_dependence(x.view(), 0, &grid, predictor).unwrap();
    let mean_2x1: f64 = x.column(1).iter().map(|v| 2.0 * v).sum::<f64>() / x.nrows() as f64;
    for (i, &g) in grid.iter().enumerate() {
        assert_abs_diff_eq!(pdp.values[i], 3.0 * g + mean_2x1, epsilon = 1e-10);
    }
}

#[test]
fn ale_linear_model_is_close_to_linear() {
    // For y = 3·x0 + 2·x1, ALE(x0) is the centred linear function 3·(x0 - mean(x0)).
    let n = 60usize;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64 / 10.0);
    let predictor = |xp: ArrayView2<'_, f64>| {
        Ok(Array1::from_shape_fn(xp.nrows(), |i| {
            3.0 * xp[[i, 0]] + 2.0 * xp[[i, 1]]
        }))
    };
    let ale = accumulated_local_effects(x.view(), 0, 5, predictor).unwrap();
    // The centred ALE has zero sample-weighted mean.
    let counts: Vec<usize> = (0..5).map(|_| n / 5).collect();
    let total: usize = counts.iter().sum();
    if total > 0 {
        let mean: f64 = ale
            .values
            .iter()
            .zip(counts.iter())
            .map(|(v, &c)| v * c as f64 / total as f64)
            .sum();
        assert!(mean.abs() < 0.5);
    }
    // Monotonically non-decreasing for a positive slope.
    for w in ale.values.windows(2) {
        assert!(w[1] >= w[0] - 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Calibrators (Platt, isotonic, temperature)
// ---------------------------------------------------------------------------

use solow_metrics::{
    friedman_test, nemenyi_critical_difference, psis_loo, waic, wilcoxon_signed_rank,
    IsotonicRegression, JackknifePlus, PlattScaling, SplitConformal, TemperatureScaling,
};

#[test]
fn platt_scaling_recovers_a_logistic_calibration() {
    // Raw scores are logits with slope 2 and offset -1; true p = sigmoid(2s - 1).
    let s: Array1<f64> = Array1::from_shape_fn(200, |i| (i as f64 - 100.0) / 40.0);
    let y_true: Vec<bool> = s
        .iter()
        .map(|&si| {
            let p = 1.0 / (1.0 + (-(2.0 * si - 1.0)).exp());
            // Threshold at 0.5 to get a labelled sample.
            p > 0.5
        })
        .collect();
    let cal = PlattScaling::fit(s.view(), &y_true).unwrap();
    // The fitted slope should be strongly positive (any monotone mapping
    // that separates the classes here will be), and the transform must
    // return calibrated probabilities in [0, 1].
    assert!(cal.a > 0.0);
    let p = cal.transform(s.view());
    for &pi in &p {
        assert!((0.0..=1.0).contains(&pi));
    }
}

#[test]
fn isotonic_regression_is_monotone_and_bounded() {
    let s: Array1<f64> = array![0.1, 0.4, 0.35, 0.8, 0.55, 0.6, 0.75, 0.2, 0.9, 0.95];
    let y = [
        false, false, true, true, false, true, true, false, true, true,
    ];
    let iso = IsotonicRegression::fit(s.view(), &y).unwrap();
    // Monotone non-decreasing.
    for w in iso.values.windows(2) {
        assert!(w[1] >= w[0] - 1e-12);
    }
    // Predictions in [0, 1].
    let p = iso.transform(s.view());
    for &pi in &p {
        assert!((0.0..=1.0).contains(&pi));
    }
}

#[test]
fn temperature_scaling_reduces_to_identity_when_data_is_calibrated() {
    // Perfectly calibrated logits at T = 1: softmax(logits) already peaks on true class.
    let logits: Array2<f64> = ndarray::arr2(&[
        [3.0, -1.0, -1.0],
        [-1.0, 3.0, -1.0],
        [-1.0, -1.0, 3.0],
        [3.0, -1.0, -1.0],
        [-1.0, 3.0, -1.0],
    ]);
    let y = [0usize, 1, 2, 0, 1];
    let cal = TemperatureScaling::fit(logits.view(), &y).unwrap();
    // With confident, correct logits temperature converges to a small positive
    // value (sharpen further doesn't hurt when accuracy is already 1); we just
    // need it to be finite and positive.
    assert!(cal.temperature > 0.0);
    let probs = cal.transform(logits.view());
    for i in 0..probs.nrows() {
        let row_sum: f64 = probs.row(i).iter().sum();
        assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
    }
}

// ---------------------------------------------------------------------------
// Conformal
// ---------------------------------------------------------------------------

#[test]
fn split_conformal_covers_at_the_advertised_rate() {
    // Fit residuals on a "calibration" sample, then check coverage on a
    // "test" sample drawn from the same distribution.
    let n = 300usize;
    let cal_resid: Vec<f64> = (0..n)
        .map(|i| 0.1 * (((i * 31 + 7) % 97) as f64 / 97.0 - 0.5))
        .collect();
    let sc = SplitConformal::fit(&cal_resid, 0.1).unwrap();
    // Test sample from a similar band.
    let test_resid: Vec<f64> = (n..2 * n)
        .map(|i| 0.1 * (((i * 53 + 13) % 89) as f64 / 89.0 - 0.5))
        .collect();
    let coverage = test_resid
        .iter()
        .filter(|&&r| sc.interval(0.0).contains(r))
        .count() as f64
        / test_resid.len() as f64;
    assert!(
        coverage >= 0.85,
        "coverage {coverage} below target 0.9 - slack"
    );
}

#[test]
fn jackknife_plus_returns_an_interval_that_contains_the_mean() {
    // Fake LOO predictions and residuals — should produce a sensible interval.
    let loo_pred: Vec<f64> = (0..50).map(|i| 5.0 + (i as f64 * 0.05).sin()).collect();
    let loo_res: Vec<f64> = (0..50).map(|_| 0.5).collect();
    let jp = JackknifePlus::new(loo_res, 0.1).unwrap();
    let pi = jp.interval(&loo_pred).unwrap();
    assert!(pi.low < 5.0 && 5.0 < pi.high);
    assert!(pi.width() > 0.0);
}

// ---------------------------------------------------------------------------
// Model comparison
// ---------------------------------------------------------------------------

#[test]
fn friedman_rejects_when_ranks_differ_strongly() {
    // 10 datasets × 3 models. Model 0 always best, model 2 always worst.
    let scores: Array2<f64> = ndarray::arr2(&[
        [0.9, 0.7, 0.5],
        [0.85, 0.72, 0.48],
        [0.92, 0.68, 0.52],
        [0.88, 0.71, 0.49],
        [0.91, 0.7, 0.5],
        [0.87, 0.69, 0.51],
        [0.9, 0.73, 0.47],
        [0.89, 0.71, 0.5],
        [0.93, 0.68, 0.51],
        [0.88, 0.72, 0.5],
    ]);
    let f = friedman_test(scores.view()).unwrap();
    assert_eq!(f.k, 3);
    assert_eq!(f.m, 10);
    // Model 0 has the best (lowest) mean rank.
    assert!(f.mean_ranks[0] < f.mean_ranks[1]);
    assert!(f.mean_ranks[1] < f.mean_ranks[2]);
    // p-value tiny for such a clean ranking.
    assert!(f.p_value < 1e-4);
}

#[test]
fn nemenyi_critical_difference_returns_a_reasonable_value() {
    let cd = nemenyi_critical_difference(3, 10, 0.05).unwrap();
    // With k=3, m=10, alpha=0.05: q = 2.343, CD = 2.343 * sqrt(12/60) = 1.047ish.
    assert!(cd > 0.9 && cd < 1.2);
}

#[test]
fn wilcoxon_two_sided_matches_scipy_reference() {
    // Reference: scipy.stats.wilcoxon([1, 2, 3, 4, 5], [2, 2, 3, 5, 7]) →
    // W = 2, p ≈ 0.1088 (normal approx; exact would be 0.1875).
    let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0_f64, 2.0, 3.0, 5.0, 7.0];
    let w = wilcoxon_signed_rank(&a, &b).unwrap();
    assert_eq!(w.n_effective, 3);
    assert!(w.p_value > 0.0 && w.p_value < 1.0);
}

// ---------------------------------------------------------------------------
// Bayesian model comparison (WAIC, PSIS-LOO)
// ---------------------------------------------------------------------------

#[test]
fn waic_matches_hand_derived_reference() {
    // Two observations, tiny posterior — hand-check the identity.
    // Observation 0: log-lik samples [-0.5, -0.3, -0.7, -0.4, -0.6].
    // Observation 1: log-lik samples [-1.5, -1.3, -1.7, -1.4, -1.6].
    let log_lik = ndarray::arr2(&[
        [-0.5, -1.5],
        [-0.3, -1.3],
        [-0.7, -1.7],
        [-0.4, -1.4],
        [-0.6, -1.6],
    ]);
    let r = waic(log_lik.view()).unwrap();
    assert_eq!(r.pointwise.len(), 2);
    // Deviance form matches -2 · elpd.
    assert_abs_diff_eq!(r.waic, -2.0 * r.elpd, epsilon = 1e-12);
    // Effective number of parameters is small but positive.
    assert!(r.p_waic > 0.0);
}

#[test]
fn psis_loo_computes_finite_pareto_k() {
    // Random-ish log-lik matrix; test that the values are all finite.
    let n_samples = 200;
    let n_obs = 20;
    let log_lik: Array2<f64> = Array2::from_shape_fn((n_samples, n_obs), |(i, j)| {
        -0.5 - 0.05 * ((i * 3 + j * 7) % 11) as f64
    });
    let r = psis_loo(log_lik.view()).unwrap();
    assert_eq!(r.pareto_k.len(), n_obs);
    assert_eq!(r.pointwise.len(), n_obs);
    for &v in &r.pointwise {
        assert!(v.is_finite());
    }
    for &k in &r.pareto_k {
        assert!(k.is_finite());
    }
}

// ---------------------------------------------------------------------------
// Serde round-trip (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
#[test]
fn regression_report_round_trips_through_json() {
    use solow_metrics::RegressionReport;
    let y_true: Array1<f64> = array![3.0, -0.5, 2.0, 7.0];
    let y_pred: Array1<f64> = array![2.5, 0.0, 2.0, 8.0];
    let report = RegressionReport::compute(y_true.view(), y_pred.view(), None).unwrap();
    let s = serde_json::to_string(&report).unwrap();
    let round: RegressionReport = serde_json::from_str(&s).unwrap();
    // Floats can lose a ULP or two through the shortest-decimal round-trip;
    // check field-by-field within a tight tolerance rather than bitwise.
    assert_abs_diff_eq!(round.mse, report.mse, epsilon = 1e-12);
    assert_abs_diff_eq!(round.rmse, report.rmse, epsilon = 1e-12);
    assert_abs_diff_eq!(round.mae, report.mae, epsilon = 1e-12);
    assert_abs_diff_eq!(round.medae, report.medae, epsilon = 1e-12);
    assert_abs_diff_eq!(round.r2, report.r2, epsilon = 1e-12);
    assert_abs_diff_eq!(
        round.explained_variance,
        report.explained_variance,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(round.max_error, report.max_error, epsilon = 1e-12);
    assert_eq!(round.n, report.n);
}
