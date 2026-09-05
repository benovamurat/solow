//! # solow-neural
//!
//! Small feed-forward multi-layer perceptrons trained by SGD or Adam
//! back-propagation.
//!
//! * [`MlpRegressor`] — least-squares output layer with `Identity` /
//!   `Logistic` / `Tanh` / `ReLU` hidden activations.
//! * [`MlpClassifier`] — softmax output layer with cross-entropy loss,
//!   trained the same way. `predict_proba` returns the row-softmax.
//!
//! The two optimisers ship as [`Solver::Sgd`] with the classical
//! momentum-optional stochastic sub-gradient, and [`Solver::Adam`] with
//! Kingma-Ba's adaptive-moment scheme (β₁ = 0.9, β₂ = 0.999, ε = 1e-8).
//! Both use a deterministic MMIX-LCG shuffle for sample ordering per
//! epoch and Xavier / Glorot weight initialisation seeded from the
//! same LCG — a given seed produces bit-identical weights across runs.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod mlp;
pub mod rbm;

pub use mlp::{Activation, MlpClassifier, MlpRegressor, Solver};
pub use rbm::BernoulliRbm;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::mlp::{Activation, MlpClassifier, MlpRegressor, Solver};
    pub use crate::rbm::BernoulliRbm;
}
