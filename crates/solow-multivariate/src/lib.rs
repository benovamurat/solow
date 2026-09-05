//! # solow-multivariate
//!
//! Multivariate statistical analysis for the Solow stack. This crate provides
//! [`Pca`] (principal component analysis), [`Factor`] (principal-axis factor
//! analysis), [`Manova`] (multivariate analysis of variance with the four
//! classical test statistics) and [`CanCorr`] (canonical correlation analysis),
//! all matching the conventions of the reference implementation
//! (`multivariate.pca`, `.factor`, `.manova`, `.cancorr`).
//!
//! ```
//! use ndarray::array;
//! use solow_multivariate::Pca;
//!
//! let data = array![
//!     [1.0, 2.0, 0.5],
//!     [2.0, 1.0, 1.5],
//!     [3.0, 0.0, 2.5],
//!     [4.0, 1.0, 0.0],
//! ];
//! let pca = Pca::new(data).fit().unwrap();
//! assert_eq!(pca.eigenvals.len(), 3);
//! // Eigenvalues are sorted in descending order.
//! assert!(pca.eigenvals[0] >= pca.eigenvals[1]);
//! ```

mod cancorr;
mod factor;
mod manova;
mod mvstats;
mod pca;
mod rotation;

pub use cancorr::{CanCorr, CanCorrTestRow};
pub use factor::{Factor, FactorResults};
pub use manova::{Manova, ManovaTest};
pub use mvstats::{multivariate_stats, MultivariateStats, MvStat};
pub use pca::{Pca, PcaResults};
pub use rotation::{rotate_factors, rotate_with, Rotation, RotationMethod};
