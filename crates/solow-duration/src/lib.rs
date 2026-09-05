//! # solow-duration
//!
//! Survival / duration analysis for the Solow statistical-computing stack.
//!
//! * [`SurvfuncRight`] — the Kaplan–Meier (product-limit) estimator of a
//!   right-censored survival function, with Greenwood standard errors.
//! * [`PHReg`] — Cox proportional-hazards regression estimated by maximizing
//!   the Breslow partial log-likelihood, exposing coefficients, standard
//!   errors, z-statistics, p-values and the partial log-likelihood.
//!
//! Both estimators are cross-validated against golden reference values frozen
//! in `tests/fixtures/duration.json`.

mod cumincidence;
mod hazard;
mod phreg_ties;
mod survdiff;
mod survfunc;

pub use cumincidence::CumIncidenceRight;
pub use hazard::{PHReg, PHRegResults};
pub use phreg_ties::{PHRegTies, PHRegTiesResults, Ties};
pub use survdiff::{survdiff, SurvDiffResult, WeightType};
pub use survfunc::SurvfuncRight;
