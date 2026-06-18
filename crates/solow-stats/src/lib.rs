//! # solow-stats
//!
//! Statistical diagnostics and hypothesis tests, validated against an
//! authoritative reference implementation.
//!
//! The crate provides the standard battery of regression-diagnostic and
//! normality tests, simple weighted descriptive statistics, two-sample
//! location tests, autocorrelation tests, and multiple-testing corrections.
//! Every public quantity is cross-validated to tight tolerances against the
//! reference (see `tests/reference.rs`).
//!
//! ```
//! use ndarray::array;
//! use solow_stats::durbin_watson;
//!
//! let resid = array![0.1, -0.2, 0.05, 0.15, -0.1];
//! let dw = durbin_watson(&resid);
//! assert!((0.0..=4.0).contains(&dw));
//! ```

mod anova;
mod contingency;
mod correlation_tools;
mod descriptivestats;
mod diagnostic;
mod dist_dependence;
mod equivalence;
mod influence;
mod inter_rater;
mod mediation;
mod multitest;
mod noncentral;
mod normality;
mod oaxaca;
mod oneway;
mod power;
mod proportion;
mod rates;
mod regdiag;
mod sandwich;
mod srange;
mod tukey;
mod weightstats;

pub use anova::{anova_lm, AnovaRow, AnovaTable, AnovaType, Term};
pub use contingency::{ContingencyResult, Table};
pub use correlation_tools::{
    corr2cov, corr_clipped, corr_nearest, cov2corr, cov2corr_std, cov_nearest, NearestMethod,
};
pub use descriptivestats::{describe, Description, PERCENTILES};
pub use diagnostic::{acorr_ljungbox, het_breuschpagan, het_white, LjungBox};
pub use dist_dependence::{
    as_column, distance_correlation, distance_covariance, distance_covariance_test,
    distance_statistics, distance_variance, DcovTest, DistDependStat,
};
pub use equivalence::{ttost_ind, TostResult};
pub use influence::{kstest_normal, lilliefors, variance_inflation_factor, LillieforsDist};
pub use inter_rater::{aggregate_raters, cohens_kappa, fleiss_kappa, FleissMethod, KappaResults};
pub use mediation::{Mediation, MediationResults};
pub use multitest::{multipletests, MultiTestMethod, MultiTestResult};
pub use noncentral::{nct_cdf, nct_sf};
pub use normality::{durbin_watson, jarque_bera, omni_normtest, JarqueBera};
pub use oaxaca::{OaxacaBlinder, ThreeFold, TwoFold, TwoFoldType};
pub use oneway::{anova_generic, anova_oneway, f_oneway, OnewayResult, UseVar as OnewayUseVar};
pub use power::{NormalIndPower, TTestPower};
pub use proportion::{proportion_confint, proportions_ztest, ConfintMethod};
pub use rates::{test_poisson_2indep, Compare, PoissonMethod, PoissonResult};
pub use regdiag::{
    acorr_breusch_godfrey, acorr_lm, compare_f_test, compare_lr_test, het_arch, linear_reset,
    ResetAug,
};
pub use sandwich::{
    cov_cluster, cov_hac, cov_hc0, cov_hc1, cov_hc2, cov_hc3, hat_diag, robust_bse,
};
pub use srange::{srange_cdf, srange_ppf, srange_sf};
pub use tukey::{pairwise_tukeyhsd, TukeyHsdResult};
pub use weightstats::{ttest_ind, ztest, Alternative, DescrStatsW, TTestResult, UseVar};
