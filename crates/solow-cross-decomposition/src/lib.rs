//! # solow-cross-decomposition
//!
//! Cross-decomposition estimators — the reference `cross_decomposition`
//! family. Every estimator here jointly decomposes an `(n × p)` predictor
//! block `X` and an `(n × q)` response block `Y` into low-rank latent
//! variables under different orthogonality / regression targets.
//!
//! * [`PLSRegression`] — the classic "PLS1/PLS2 regression" mode. Latent
//!   directions maximise `Cov(Xw, Yc)` under `‖w‖ = ‖c‖ = 1`, then `Y` is
//!   regressed on the score matrix `T = XW*`. This is the reference
//!   `PLSRegression`; also known as PLS2 when `q > 1`.
//! * [`PLSCanonical`] — same maximisation objective, but responses are
//!   deflated symmetrically. Latent scores are mutually orthogonal on both
//!   sides. This is the reference `PLSCanonical`.
//! * [`PLSSVD`] — a single SVD of the cross-covariance matrix `X̄ᵀȲ` with
//!   no iteration; matches `cross_decomposition.PLSSVD`.
//! * [`CCA`] — canonical-correlation analysis; the correlation-maximising
//!   dual of PLS. Matches `cross_decomposition.CCA`.
//!
//! # References
//!
//! * Wold, H. (1975). *Path Models with Latent Variables: The NIPALS
//!   Approach.* In Quantitative Sociology.
//! * Rosipal, R., & Krämer, N. (2006). *Overview and Recent Advances in
//!   Partial Least Squares.* Lecture Notes in Computer Science 3940.
//! * Hotelling, H. (1936). *Relations Between Two Sets of Variates.*
//!   Biometrika 28(3/4), 321-377.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cca;
pub mod pls_canonical;
pub mod pls_regression;
pub mod pls_svd;

pub use cca::CCA;
pub use pls_canonical::PLSCanonical;
pub use pls_regression::PLSRegression;
pub use pls_svd::PLSSVD;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::cca::CCA;
    pub use crate::pls_canonical::PLSCanonical;
    pub use crate::pls_regression::PLSRegression;
    pub use crate::pls_svd::PLSSVD;
}
