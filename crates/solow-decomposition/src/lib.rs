//! # solow-decomposition
//!
//! Matrix-decomposition estimators complementing [`solow-multivariate`]'s
//! classical PCA / factor / rotation surface.
//!
//! * [`KernelPca`] — Schölkopf-Smola-Müller (1998) kernel PCA with
//!   `Linear`, `Rbf`, and `Polynomial` kernels; centering in feature
//!   space matches the reference `KernelPCA(fit_inverse_transform=False)`.
//! * [`FastIca`] — Hyvärinen (1999) FastICA with the `logcosh` and
//!   `exp` nonlinearities and symmetric decorrelation.
//! * [`Nmf`] — Lee-Seung (2001) multiplicative-update non-negative
//!   matrix factorisation for the Frobenius objective.
//!
//! All three consume a dense `n × d` matrix and produce an `n × k`
//! projection (KernelPCA, FastICA) or a `n × k` × `k × d`
//! factorisation (NMF). Deterministic under a caller-supplied seed.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dictionary_learning;
pub mod ica;
pub mod incremental_pca;
pub mod kernel_pca;
pub mod lda;
pub mod minibatch_dict;
pub mod minibatch_nmf;
pub mod nmf;
pub mod random_projection;
pub mod sparse_pca;
pub mod truncated_svd;

pub use dictionary_learning::DictionaryLearning;
pub use ica::{FastIca, IcaFun};
pub use incremental_pca::IncrementalPCA;
pub use kernel_pca::{KernelKind, KernelPca};
pub use lda::LatentDirichletAllocation;
pub use minibatch_dict::MiniBatchDictionaryLearning;
pub use minibatch_nmf::MiniBatchNmf;
pub use nmf::Nmf;
pub use random_projection::{
    johnson_lindenstrauss_min_dim, GaussianRandomProjection, SparseRandomProjection,
};
pub use sparse_pca::SparsePCA;
pub use truncated_svd::TruncatedSVD;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::dictionary_learning::DictionaryLearning;
    pub use crate::ica::{FastIca, IcaFun};
    pub use crate::incremental_pca::IncrementalPCA;
    pub use crate::kernel_pca::{KernelKind, KernelPca};
    pub use crate::lda::LatentDirichletAllocation;
    pub use crate::minibatch_dict::MiniBatchDictionaryLearning;
    pub use crate::minibatch_nmf::MiniBatchNmf;
    pub use crate::nmf::Nmf;
    pub use crate::random_projection::{
        johnson_lindenstrauss_min_dim, GaussianRandomProjection, SparseRandomProjection,
    };
    pub use crate::sparse_pca::SparsePCA;
    pub use crate::truncated_svd::TruncatedSVD;
}
