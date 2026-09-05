//! # solow-cluster
//!
//! Unsupervised clustering for the Solow statistical stack.
//!
//! The crate ships three families that together cover the classical
//! unsupervised-learning use cases:
//!
//! * **Centroid-based:** [`KMeans`] with the Arthur-Vassilvitskii
//!   [`k-means++`](https://en.wikipedia.org/wiki/K-means%2B%2B)
//!   initialisation and Lloyd's iterative refinement. Deterministic
//!   under a caller-supplied seed via a portable MMIX-LCG PRNG.
//! * **Density-based:** [`Dbscan`] — the Ester-Kriegel-Sander-Xu
//!   (KDD 1996) `ε`, `min_samples` classifier. Reports per-point
//!   `Core` / `Border` / `Noise` roles alongside the cluster label.
//! * **Hierarchical:** [`AgglomerativeClustering`] with `Single`,
//!   `Complete`, `Average`, or `Ward` linkage. Uses the Lance-Williams
//!   update recurrence so the whole dendrogram is built in `O(n² · log n)`
//!   time with `O(n²)` space.
//!
//! Every clusterer exposes the classical `fit_predict(x) -> labels`
//! call, plus a richer typed result carrying centroids / linkage
//! matrices / core-point flags when applicable. All types derive
//! `Serialize` / `Deserialize` under the opt-in `serde` feature.
//!
//! ## Convergence and complexity
//!
//! * `KMeans`: each Lloyd iteration is `O(n · k · d)`. `k-means++`
//!   pre-fit is `O(n · k · d)` in expectation and guarantees an
//!   `O(log k)`-competitive initial inertia (Arthur & Vassilvitskii 2007).
//! * `DBSCAN`: naive neighbour lookup is `O(n²)`; use
//!   [`solow-neighbors`](https://docs.rs/solow-neighbors) `KDTree`
//!   externally to obtain `O(n log n)` for low-dimensional data.
//! * `AgglomerativeClustering`: `O(n² · log n)` time, `O(n²)` space
//!   via Müllner (2011) *Modern hierarchical, agglomerative clustering
//!   algorithms*.
//!
//! ## Determinism
//!
//! Every stochastic component (`KMeans` init, `MiniBatchKMeans` batch
//! draws) uses a portable 64-bit MMIX linear-congruential generator
//! seeded by the caller. Fits are bit-identical across runs and
//! platforms given a fixed seed and input.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod affinity_propagation;
pub mod agglomerative;
pub mod bgmm;
pub mod birch;
pub mod dbscan;
pub mod gmm;
pub mod hdbscan;
pub mod kmeans;
pub mod meanshift;
pub mod minibatch;
pub mod optics;
pub mod spectral;

pub use affinity_propagation::AffinityPropagation;
pub use agglomerative::{AgglomerativeClustering, DendrogramNode, Linkage};
pub use bgmm::BayesianGaussianMixture;
pub use birch::Birch;
pub use dbscan::{Dbscan, DbscanResult, PointRole};
pub use gmm::{CovType, GaussianMixture};
pub use hdbscan::Hdbscan;
pub use kmeans::{KMeans, KMeansInit, KMeansResult};
pub use meanshift::MeanShift;
pub use minibatch::{BisectingKMeans, MiniBatchKMeans};
pub use optics::Optics;
pub use spectral::SpectralClustering;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::affinity_propagation::AffinityPropagation;
    pub use crate::agglomerative::{AgglomerativeClustering, DendrogramNode, Linkage};
    pub use crate::birch::Birch;
    pub use crate::dbscan::{Dbscan, DbscanResult, PointRole};
    pub use crate::hdbscan::Hdbscan;
    pub use crate::kmeans::{KMeans, KMeansInit, KMeansResult};
    pub use crate::minibatch::{BisectingKMeans, MiniBatchKMeans};
    pub use crate::optics::Optics;
    pub use crate::spectral::SpectralClustering;
}
