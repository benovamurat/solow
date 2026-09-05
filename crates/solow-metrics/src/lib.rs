//! # solow-metrics
//!
//! Model-evaluation metrics for the Solow statistical stack — the numbers you
//! report after a fit, not the ones an estimator uses internally.
//!
//! The crate is organised by task:
//!
//! * [`regression`] — continuous-target losses and scores: mean squared error,
//!   RMSE, mean absolute error, MAPE, sMAPE, R², explained variance,
//!   max-error, mean pinball loss for quantile regression, Tweedie / Poisson
//!   / Gamma deviances, and their sample-weighted variants.
//! * [`classification`] — labels and probabilities: confusion matrix, accuracy,
//!   balanced accuracy, precision / recall / Fβ with `macro` / `micro` /
//!   `weighted` / per-class averaging, Matthews correlation, Cohen's κ (linear
//!   and quadratic weighting), zero-one loss, hinge loss, binary log-loss,
//!   Brier score, and — for scored classifiers — the ROC and precision-recall
//!   curves, ROC-AUC (binary, one-vs-rest and Hand-Till one-vs-one for
//!   multiclass), and average precision.
//! * [`calibration`] — probability-calibration diagnostics: reliability curves
//!   with uniform or quantile bins, expected and maximum calibration error,
//!   and the Sanders / Murphy decomposition of the Brier score.
//! * [`forecast`] — time-series specific losses: MASE, RMSSE, pinball loss for
//!   prediction quantiles, interval coverage, the mean interval score
//!   (Winkler) for prediction bands, and the Harvey-Leybourne-Newbold
//!   small-sample-corrected Diebold-Mariano test for equal predictive
//!   accuracy of two forecasts.
//!
//! All functions take ndarray views so they compose with slices, subranges, and
//! matrix rows without copies. Every metric returns a `Result` because pairs
//! with mismatched shapes, negative sample weights, unknown classes, or
//! degenerate inputs (a constant target for R², a single-class label vector for
//! ROC-AUC, …) are user errors, not silent NaNs.
//!
//! The metric names, signatures, and averaging semantics deliberately mirror
//! the canonical `the reference` API so that ports and cross-checks are literal
//! — a reader can translate a the reference example one call at a time.
//!
//! ## Example
//!
//! ```
//! use ndarray::array;
//! use solow_metrics::{mean_squared_error, r2_score};
//!
//! let y_true = array![3.0, -0.5, 2.0, 7.0];
//! let y_pred = array![2.5, 0.0, 2.0, 8.0];
//!
//! let mse = mean_squared_error(y_true.view(), y_pred.view(), None).unwrap();
//! let r2  = r2_score(y_true.view(), y_pred.view(), None).unwrap();
//!
//! assert!((mse - 0.375).abs() < 1e-12);
//! // R² = 1 - SS_res / SS_tot = 1 - 1.5 / 29.1875.
//! assert!((r2 - (1.0 - 1.5 / 29.1875)).abs() < 1e-10);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bayesian;
pub mod calibration;
pub mod calibrators;
pub mod classification;
pub mod cluster;
pub mod comparison;
pub mod conformal;
pub mod effect_size;
pub mod forecast;
pub mod inspection;
pub mod pairwise;
pub mod regression;
pub mod report;

pub use bayesian::{psis_loo, waic, PsisLooResult, WaicResult};
pub use calibration::{
    brier_decomposition, expected_calibration_error, maximum_calibration_error, reliability_curve,
    BinStrategy, BrierDecomposition, ReliabilityBin,
};
pub use calibrators::{IsotonicRegression, PlattScaling, TemperatureScaling};
pub use classification::{
    accuracy_score, average_precision_score, balanced_accuracy_score, binary_focal_loss,
    binary_log_loss, brier_score_loss, cohen_kappa_score, confusion_matrix, fbeta_score,
    hinge_loss, log_loss, matthews_corrcoef, multiclass_brier_score, multiclass_focal_loss,
    precision_recall_curve, precision_recall_fscore, ranked_probability_score, roc_auc_ovo_score,
    roc_auc_ovr_score, roc_auc_score, roc_curve, top_k_accuracy_score, top_label_calibration_error,
    zero_one_loss, Average, KappaWeights, MulticlassAuc, PrecisionRecallFScore,
    RocCurve as RocCurveResult,
};
pub use comparison::{
    friedman_test, nemenyi_critical_difference, wilcoxon_signed_rank, FriedmanResult,
    WilcoxonResult,
};
pub use cluster::{
    adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, completeness_score,
    davies_bouldin_score, fowlkes_mallows_score, homogeneity_score,
    normalized_mutual_info_score, silhouette_score, v_measure_score, MiAverage,
};
pub use conformal::{JackknifePlus, PredictionInterval, SplitConformal};
pub use effect_size::{
    cliffs_delta, cohens_d, cramers_v, eta_squared, glass_delta, hedges_g, omega_squared,
};
pub use forecast::{
    diebold_mariano, giacomini_white_test, interval_coverage, mase, mean_interval_score,
    pinball_loss, rmsse, DieboldMarianoResult, DmLoss, GiacominiWhiteResult,
};
pub use inspection::{
    accumulated_local_effects, partial_dependence, permutation_importance, AccumulatedLocalEffects,
    FeatureImportance, PartialDependence,
};
pub use pairwise::{
    chi2_kernel, cosine_similarity, laplacian_kernel, linear_kernel, pairwise_distances,
    polynomial_kernel, rbf_kernel, sigmoid_kernel, PairwiseMetric,
};
pub use report::{classification_report, ClassificationReport, ClassificationRow};
pub use regression::{
    d2_absolute_error_score, d2_tweedie_score, explained_variance_score, huber_loss, log_cosh_loss,
    max_error, mean_absolute_error, mean_absolute_percentage_error, mean_gamma_deviance,
    mean_pinball_loss, mean_poisson_deviance, mean_squared_error, mean_squared_log_error,
    mean_tweedie_deviance, median_absolute_error, r2_score, root_mean_squared_error,
    root_mean_squared_log_error, symmetric_mean_absolute_percentage_error, RegressionReport,
};

/// Commonly used imports.
///
/// `use solow_metrics::prelude::*;` brings every metric name plus the shared
/// [`Average`], [`KappaWeights`], [`MulticlassAuc`], [`BinStrategy`], and
/// [`DmLoss`] enums into scope in one line.
pub mod prelude {
    pub use crate::bayesian::{psis_loo, waic, PsisLooResult, WaicResult};
    pub use crate::calibration::{
        brier_decomposition, expected_calibration_error, maximum_calibration_error,
        reliability_curve, BinStrategy, BrierDecomposition, ReliabilityBin,
    };
    pub use crate::calibrators::{IsotonicRegression, PlattScaling, TemperatureScaling};
    pub use crate::classification::{
        accuracy_score, average_precision_score, balanced_accuracy_score, binary_focal_loss,
        binary_log_loss, brier_score_loss, cohen_kappa_score, confusion_matrix, fbeta_score,
        hinge_loss, log_loss, matthews_corrcoef, multiclass_brier_score, multiclass_focal_loss,
        precision_recall_curve, precision_recall_fscore, ranked_probability_score,
        roc_auc_ovo_score, roc_auc_ovr_score, roc_auc_score, roc_curve, top_k_accuracy_score,
        top_label_calibration_error, zero_one_loss, Average, KappaWeights, MulticlassAuc,
    };
    pub use crate::comparison::{
        friedman_test, nemenyi_critical_difference, wilcoxon_signed_rank, FriedmanResult,
        WilcoxonResult,
    };
    pub use crate::conformal::{JackknifePlus, PredictionInterval, SplitConformal};
    pub use crate::forecast::{
        diebold_mariano, giacomini_white_test, interval_coverage, mase, mean_interval_score,
        pinball_loss, rmsse, DmLoss,
    };
    pub use crate::inspection::{
        accumulated_local_effects, partial_dependence, permutation_importance,
        AccumulatedLocalEffects, FeatureImportance, PartialDependence,
    };
    pub use crate::regression::{
        d2_absolute_error_score, d2_tweedie_score, explained_variance_score, huber_loss,
        log_cosh_loss, max_error, mean_absolute_error, mean_absolute_percentage_error,
        mean_gamma_deviance, mean_pinball_loss, mean_poisson_deviance, mean_squared_error,
        mean_squared_log_error, mean_tweedie_deviance, median_absolute_error, r2_score,
        root_mean_squared_error, root_mean_squared_log_error,
        symmetric_mean_absolute_percentage_error, RegressionReport,
    };
}
