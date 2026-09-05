//! # solow-manifold
//!
//! Manifold-learning dimensionality reduction.
//!
//! * [`Isomap`] — Tenenbaum-de Silva-Langford (2000) geodesic MDS.
//!   Builds a k-NN graph, computes shortest-path distances via
//!   Dijkstra, then runs classical MDS on the geodesic distance
//!   matrix.
//! * [`LocallyLinearEmbedding`] — Roweis-Saul (2000). Fits local
//!   linear reconstruction weights in the high-dimensional space and
//!   embeds by solving the sparse eigen-system `(I − W)ᵀ(I − W)`.
//! * [`Tsne`] — van der Maaten-Hinton (2008) t-SNE with the standard
//!   perplexity-based Gaussian kernel and Student-t low-dim kernel.
//!   O(n²) full-matrix implementation — suitable up to a few
//!   thousand points; deterministic under a caller-supplied seed.
//!
//! All three consume an `n × d_in` matrix and produce an `n × d_out`
//! embedding.
//!
//! # References
//!
//! * Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). *A
//!   global geometric framework for nonlinear dimensionality
//!   reduction.* Science 290(5500), 2319-2323.
//! * Roweis, S. T., & Saul, L. K. (2000). *Nonlinear dimensionality
//!   reduction by locally linear embedding.* Science 290(5500),
//!   2323-2326.
//! * van der Maaten, L., & Hinton, G. (2008). *Visualizing data using
//!   t-SNE.* JMLR 9, 2579-2605.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod isomap;
pub mod lle;
pub mod mds;
pub mod spectral_embedding;
pub mod tsne;

pub use isomap::Isomap;
pub use lle::LocallyLinearEmbedding;
pub use mds::MDS;
pub use spectral_embedding::SpectralEmbedding;
pub use tsne::Tsne;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::isomap::Isomap;
    pub use crate::lle::LocallyLinearEmbedding;
    pub use crate::mds::MDS;
    pub use crate::spectral_embedding::SpectralEmbedding;
    pub use crate::tsne::Tsne;
}
