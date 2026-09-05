//! # solow-svm
//!
//! Linear support vector machines fit by Pegasos-style stochastic
//! sub-gradient descent (Shalev-Shwartz-Singer-Srebro 2007).
//!
//! * [`LinearSvc`] — binary hinge-loss linear classifier
//!   `L₂`-regularised. Multi-class extension is one-vs-rest;
//!   `LinearSvc::fit_multiclass` returns per-class classifiers and
//!   picks the argmax margin at predict time.
//! * [`LinearSvr`] — ε-insensitive Vapnik regressor, same solver.
//!
//! # Algorithm
//!
//! For binary hinge with regularisation `λ = 1 / (n · C)`:
//!
//! ```text
//! w ← w − η_t · ∇L_t,     η_t = 1 / (λ · t),
//! ∇L_t = λ·w − y_i·x_i · [y_i (wᵀx_i) < 1].
//! ```
//!
//! Each pass over the sample is one epoch; `max_iter` epochs run in
//! total. Sample ordering per epoch is a deterministic MMIX-LCG
//! shuffle of `[0, n)`, so a fixed seed produces bit-identical
//! coefficients across runs and platforms.
//!
//! # References
//!
//! * Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011).
//!   *Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.*
//!   Mathematical Programming 127(1), 3-30.
//! * Vapnik, V. (1995). *The Nature of Statistical Learning Theory.*
//!   Springer.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod kernel;
pub mod linear;
pub mod nu_svm;
pub mod one_class;

pub use kernel::{KernelKind, Svc, Svr};
pub use linear::{LinearSvc, LinearSvr};
pub use nu_svm::{NuSvc, NuSvr};
pub use one_class::OneClassSvm;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::kernel::{KernelKind, Svc, Svr};
    pub use crate::linear::{LinearSvc, LinearSvr};
    pub use crate::nu_svm::{NuSvc, NuSvr};
    pub use crate::one_class::OneClassSvm;
}
