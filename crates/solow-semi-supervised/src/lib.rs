//! # solow-semi-supervised
//!
//! Semi-supervised learners — the reference `semi_supervised`
//! family. Unlabelled samples are marked with `label = -1` on input.
//!
//! * [`LabelPropagation`] — Zhu-Ghahramani (2002) label-propagation via
//!   the clamped normalised graph Laplacian.
//! * [`LabelSpreading`] — Zhou et al. (2004) soft-clamp variant with a
//!   symmetric normalised Laplacian.
//! * [`SelfTrainingClassifier`] — the reference `SelfTrainingClassifier`
//!   wrapper that repeatedly re-labels the highest-confidence unlabelled
//!   samples with a caller-supplied base estimator.
//!
//! # References
//!
//! * Zhu, X., & Ghahramani, Z. (2002). *Learning from Labeled and
//!   Unlabeled Data with Label Propagation.* CMU-CALD-02-107.
//! * Zhou, D., Bousquet, O., Lal, T., Weston, J., & Schölkopf, B. (2004).
//!   *Learning with Local and Global Consistency.* NIPS 16.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod label_propagation;
pub mod label_spreading;
pub mod self_training;

pub use label_propagation::LabelPropagation;
pub use label_spreading::LabelSpreading;
pub use self_training::{BaseClassifier, SelfTrainingClassifier};

/// Commonly-used imports.
pub mod prelude {
    pub use crate::label_propagation::LabelPropagation;
    pub use crate::label_spreading::LabelSpreading;
    pub use crate::self_training::{BaseClassifier, SelfTrainingClassifier};
}
