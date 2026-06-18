//! # solow-tsa
//!
//! Time-series analysis primitives and models for the Solow statistical stack,
//! validated against an authoritative reference.
//!
//! The crate provides:
//!
//! - Sample second-moment estimators: [`acovf`], [`acf`], [`pacf`], [`ccf`].
//! - The Ljung-Box [`q_stat`] portmanteau statistic.
//! - Design helpers [`lagmat`] and [`add_trend`].
//! - The augmented Dickey-Fuller unit-root test [`adfuller`].
//! - The autoregressive estimator [`AutoReg`].
//!
//! ```
//! use ndarray::Array1;
//! use solow_tsa::{acf, AutoReg, Trend};
//!
//! // A short AR(1)-like series.
//! let y = Array1::from_vec(vec![
//!     0.0, 0.4, 0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.35, 0.75, 0.4, 0.8,
//! ]);
//! let a = acf(&y, 3, false).unwrap();
//! assert!((a[0] - 1.0).abs() < 1e-12);
//!
//! let res = AutoReg::new(y, 1, Trend::C).unwrap().fit().unwrap();
//! assert_eq!(res.params.len(), 2); // const + 1 lag
//! ```

mod ar_model;
mod ar_select;
mod arma_process;
mod coint;
mod deterministic;
mod filters;
mod granger;
mod holtwinters;
mod innovations;
mod kpss;
mod order_select;
mod seasonal;
mod stattools;
mod stattools_ext;
mod stl;
mod tsatools;

pub use ar_model::{AutoReg, AutoRegResults};
pub use ar_select::{ar_select_order, ArIc, ArSelectResult};
pub use arma_process::{arma_acf, arma_acovf, arma_impulse_response, arma_pacf};
pub use coint::{coint, CointResult, CointTrend};
pub use deterministic::DeterministicProcess;
pub use filters::{bkfilter, cffilter, hpfilter};
pub use granger::{grangercausalitytests, GrangerLagResult};
pub use holtwinters::{ExponentialSmoothing, Holt, Seasonal, SimpleExpSmoothing, SmoothingResult};
pub use innovations::{innovations_algo, InnovationsResult};
pub use kpss::{kpss, KpssLags, KpssRegression, KpssResult};
pub use order_select::{arma_order_select_ic, ArmaOrderSelectResult, InfoCriterion};
pub use seasonal::{seasonal_decompose, DecomposeResult, SeasonalModel};
pub use stattools::{
    acf, acf_qstat, acovf, adfuller, ccf, pacf, q_stat, AdfRegression, AdfResult, AutoLag,
    PacfMethod,
};
pub use stattools_ext::{
    breakvar_heteroskedasticity_test, range_unit_root_test, zivot_andrews, BreakvarAlternative,
    RangeUnitRootResult, SubsetLength, ZaRegression, ZivotAndrewsResult,
};
pub use stl::{Stl, StlResult};
pub use tsatools::{add_trend, lagmat, lagmat1d, Original, Trend, Trim};
