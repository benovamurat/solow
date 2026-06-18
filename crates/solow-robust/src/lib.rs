//! # solow-robust
//!
//! Robust linear regression by M-estimation, matching the reference's robust
//! linear model (RLM).
//!
//! The estimator minimizes `Σ ρ((yᵢ − xᵢ·β) / σ)` for a robust criterion `ρ`
//! using iteratively reweighted least squares (IRLS), re-estimating the scale
//! `σ` from the residuals at each step.
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use solow_robust::{norms::TukeyBiweight, Rlm};
//!
//! // A noisy line near y = 2 + 0.5·x with a gross outlier at the last point.
//! let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
//! let y = Array1::from(vec![
//!     2.6, 3.1, 3.4, 4.1, 4.4, 5.1, 5.4, 6.1, 6.4, 100.0,
//! ]);
//! let exog =
//!     Array2::from_shape_fn((10, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] });
//! let res = Rlm::new(y, exog, TukeyBiweight::default())
//!     .unwrap()
//!     .fit()
//!     .unwrap();
//! assert!(res.converged);
//! // The redescending norm fully rejects the outlier ...
//! assert_eq!(res.weights[9], 0.0);
//! // ... so the slope stays close to the clean trend rather than ~10.
//! assert!((res.params[1] - 0.5).abs() < 0.05);
//! ```
//!
//! ## Components
//!
//! * [`norms`] — robust criterion functions ([`norms::HuberT`],
//!   [`norms::TukeyBiweight`], [`norms::AndrewWave`], [`norms::LeastSquares`]).
//! * [`scale`] — robust scale estimators ([`scale::mad`], [`scale::Huber`],
//!   [`scale::HuberScale`]).
//! * [`Rlm`] / [`RlmResults`] — the model and its fitted result.

pub mod norms;
pub mod norms_ext;
pub mod scale;

mod rlm;

pub use norms_ext::{Hampel, RamsayE, TrimmedMean};
pub use rlm::{Conv, Rlm, RlmResults, ScaleEst};
