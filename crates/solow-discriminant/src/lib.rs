//! # solow-discriminant
//!
//! Linear (LDA) and quadratic (QDA) discriminant analysis for
//! classification.
//!
//! Both are Bayes-optimal classifiers under the assumption
//! `p(x | y = c) = 𝒩(x; μ_c, Σ_c)`:
//!
//! * [`LinearDiscriminantAnalysis`] — assumes a **shared** covariance
//!   `Σ_c = Σ` across classes, yielding linear decision boundaries.
//!   Reduces to Fisher's LDA when there are only two classes.
//! * [`QuadraticDiscriminantAnalysis`] — allows a distinct `Σ_c` per
//!   class, giving quadratic boundaries.
//!
//! # Numerics
//!
//! Both use a Cholesky-with-diagonal-regularisation solve for the
//! per-class Mahalanobis distance
//!
//! ```text
//! (x − μ_c)ᵀ Σ_c⁻¹ (x − μ_c),
//! ```
//!
//! so degenerate near-singular covariances degrade gracefully rather
//! than panicking. A small `regularisation = 1e-4 · tr(Σ) / d` is
//! added to the diagonal by default; the constant is exposed as a
//! parameter for tighter control.
//!
//! # References
//!
//! * Fisher, R. A. (1936). *The use of multiple measurements in
//!   taxonomic problems.* Annals of Eugenics 7(2), 179-188.
//! * Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements
//!   of Statistical Learning* (2nd ed.), §4.3.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod lda;
pub mod qda;

pub use lda::LinearDiscriminantAnalysis;
pub use qda::QuadraticDiscriminantAnalysis;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::lda::LinearDiscriminantAnalysis;
    pub use crate::qda::QuadraticDiscriminantAnalysis;
}
