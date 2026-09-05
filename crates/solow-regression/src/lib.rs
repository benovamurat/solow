//! # solow-regression
//!
//! Linear regression models estimated by least squares, with the full standard
//! battery of results and inference statistics. Validated against an
//! authoritative reference.
//!
//! ```
//! use ndarray::{array, Array1};
//! use solow_core::tools::{add_constant, HasConstant};
//! use solow_regression::LinearModel;
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];
//! let design = add_constant(&x, true, HasConstant::Add).unwrap();
//! let res = LinearModel::ols(y, design).unwrap().fit().unwrap();
//! assert!((res.rsquared - 0.997).abs() < 0.01);
//! ```

mod bayesian;
mod dimred;
mod dummy;
mod glsar;
mod huber;
mod lars;
mod lasso_lars_ic;
mod linear;
mod multitask;
mod penalized;
mod penalized_cv;
mod quantile;
mod ridge_classifier;
mod recursive;
mod robust_regression;
mod robustcov;
mod rolling;
mod sgd;

pub use bayesian::{ARDRegression, BayesianRidge};
pub use dimred::{SirResults, SlicedInverseReg};
pub use dummy::{
    DummyClassifier, DummyClassifierStrategy, DummyRegressor, DummyRegressorStrategy,
};
pub use glsar::{Glsar, GlsarResults};
pub use huber::{HuberRegressor, KernelRidge, RidgeKernel};
pub use lars::{Lars, LassoLars, OrthogonalMatchingPursuit};
pub use lasso_lars_ic::{InformationCriterion, LassoLarsIC};
pub use linear::{LinearModel, LinearResults};
pub use multitask::{MultiTaskElasticNet, MultiTaskLasso};
pub use penalized::{ElasticNet, Lasso, Ridge};
pub use penalized_cv::{ElasticNetCV, LassoCV, RidgeCV};
pub use quantile::{QuantReg, QuantRegResults};
pub use ridge_classifier::{RidgeClassifier, RidgeClassifierCV};
pub use recursive::{RecursiveLS, RecursiveLSResults};
pub use robust_regression::{RansacRegressor, TheilSenRegressor};
pub use robustcov::{bse_from_cov, robust_cov, CovType};
pub use rolling::{RollingOLS, RollingOLSResults};
pub use sgd::{
    PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, SgdClassifier, SgdLoss,
    SgdPenalty, SgdRegressor,
};
