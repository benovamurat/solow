//! # solow-covariance
//!
//! Robust and shrinkage covariance-matrix estimators — the reference
//! `covariance` family.
//!
//! * [`EmpiricalCovariance`] — classical sample covariance,
//!   `S = (1/n) Σᵢ (xᵢ − μ̂)(xᵢ − μ̂)ᵀ`.
//! * [`ShrunkCovariance`] — sample covariance shrunk toward the
//!   scaled identity `(1 − ρ)·S + ρ · (tr(S)/d) · I` for a
//!   caller-supplied ρ ∈ [0, 1].
//! * [`LedoitWolf`] — the Ledoit-Wolf (2004) automatic-shrinkage
//!   estimator; picks ρ analytically to minimise expected Frobenius
//!   distance to the true covariance.
//! * [`Oas`] — the Chen-Wiesel-Eldar-Hero (2010) OAS shrinkage; a
//!   Gaussian-optimal alternative to Ledoit-Wolf.
//! * [`MinCovDet`] — Rousseeuw's (1984) FAST-MCD minimum-covariance-
//!   determinant robust estimator. Uses a fixed 500-restart schedule
//!   with a caller-provided seed for deterministic re-runs.
//! * [`GraphicalLasso`] — sparse-precision-matrix estimation under
//!   the L1 penalty (Friedman-Hastie-Tibshirani 2008), fit by
//!   block-coordinate descent.
//!
//! # References
//!
//! * Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for
//!   large-dimensional covariance matrices.* JMVA 88(2), 365-411.
//! * Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010).
//!   *Shrinkage algorithms for MMSE covariance estimation.* IEEE TSP
//!   58(10), 5016-5029.
//! * Rousseeuw, P. J. (1984). *Least median of squares regression.*
//!   JASA 79(388), 871-880.
//! * Friedman, J., Hastie, T., & Tibshirani, R. (2008). *Sparse
//!   inverse covariance estimation with the graphical lasso.*
//!   Biostatistics 9(3), 432-441.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod elliptic_envelope;
pub mod empirical;
pub mod graphical_lasso;
pub mod mcd;
pub mod shrinkage;

pub use elliptic_envelope::EllipticEnvelope;
pub use empirical::EmpiricalCovariance;
pub use graphical_lasso::GraphicalLasso;
pub use mcd::MinCovDet;
pub use shrinkage::{LedoitWolf, Oas, ShrunkCovariance};

/// Commonly-used imports.
pub mod prelude {
    pub use crate::elliptic_envelope::EllipticEnvelope;
    pub use crate::empirical::EmpiricalCovariance;
    pub use crate::graphical_lasso::GraphicalLasso;
    pub use crate::mcd::MinCovDet;
    pub use crate::shrinkage::{LedoitWolf, Oas, ShrunkCovariance};
}
