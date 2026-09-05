//! # solow-summary
//!
//! Labeled regression-results summary tables for Solow models -- the rendered,
//! human-readable companion to a fitted model's numeric results.
//!
//! The crate is deliberately model-agnostic: a [`RegressionSummary`] is built
//! from plain values (parameter names, estimates, standard errors, test
//! statistics, p-values, confidence intervals, and a set of header
//! statistics), so the results of *any* estimator can be rendered without this
//! crate depending on that estimator's crate.
//!
//! ## Layout
//!
//! A summary renders as two stacked fixed-width text blocks:
//!
//! 1. a **header block** of key statistics, and
//! 2. a **coefficient table** with columns `coef`, `std err`, the test
//!    statistic (`t` or `z`), the corresponding p-value, and the lower/upper
//!    confidence-interval bounds.
//!
//! Both blocks are produced by the general [`SummaryTable`] type, which lays
//! out titled, optionally-headered, row-based tables with right-aligned
//! (or per-column left-aligned) fixed-width columns.
//!
//! The *visual layout* here is Solow's own. Only the displayed *numbers* are
//! cross-checked against a reference implementation (see the crate's
//! `tests/reference.rs`).
//!
//! ## Example
//!
//! ```
//! use solow_summary::{HeaderStats, RegressionSummary};
//!
//! let names = ["const", "x1"];
//! let params = [1.5, -2.0];
//! let bse = [0.07, 0.08];
//! let tvalues = [21.4, -25.0];
//! let pvalues = [1e-20, 1e-22];
//! let conf_int = [(1.35, 1.65), (-2.16, -1.84)];
//!
//! let header = HeaderStats {
//!     model: Some("OLS".into()),
//!     nobs: Some(40.0),
//!     rsquared: Some(0.95),
//!     aic: Some(45.9),
//!     bic: Some(51.0),
//!     ..HeaderStats::new()
//! };
//!
//! let summary =
//!     RegressionSummary::new(&names, &params, &bse, &tvalues, &pvalues, &conf_int, header);
//! let text = summary.to_string();
//! assert!(text.contains("R-squared:"));
//! assert!(text.contains("const"));
//! ```

mod format;
mod regression;
mod table;

pub use format::{format_fixed, format_g, format_pvalue};
pub use regression::{HeaderStats, RegressionSummary, StatKind};
pub use table::{Align, SummaryTable};
