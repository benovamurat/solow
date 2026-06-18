//! # solow-regime
//!
//! Markov-switching (regime-switching) time-series models, estimated by maximum
//! likelihood via the Hamilton filter.
//!
//! Two models are provided:
//!
//! * [`MarkovRegression`] — a first-order `k`-regime switching regression. The
//!   intercept (and optional exogenous regressors) and, optionally, the error
//!   variance may switch across regimes.
//! * [`MarkovAutoregression`] — a `k`-regime switching autoregression of a given
//!   `order`. The mean, the autoregressive coefficients, and optionally the
//!   variance may switch across regimes.
//!
//! Both maximise the Hamilton-filter log-likelihood. The transition
//! probabilities are mapped through a logistic transform, the variances through
//! a square map, and (for the autoregression) the AR coefficients through the
//! Monahan partial-autocorrelation stationarity transform, so that the BFGS
//! optimiser in [`solow_optimize`] runs unconstrained. Filtered and smoothed
//! regime probabilities are exposed via the Hamilton filter and the Kim
//! smoother respectively.
//!
//! The parameter layout follows the canonical reference implementation. For
//! [`MarkovRegression`] it is
//! `[ transition | mean/exog (per regime) | variance ]`, and for
//! [`MarkovAutoregression`] it is
//! `[ transition | mean (per regime) | variance | autoregressive (per regime) ]`
//! (variance precedes the autoregressive block). For two regimes the transition
//! block is `[p[0->0], p[1->0]]`, i.e. the probability of *staying* in regime 0
//! and the probability of moving from regime 1 to regime 0.

mod filter;
mod markov_autoregression;
mod markov_regression;
mod switching;

pub use markov_autoregression::MarkovAutoregression;
pub use markov_regression::MarkovRegression;
pub use switching::MarkovResults;
