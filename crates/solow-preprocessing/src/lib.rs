//! # solow-preprocessing
//!
//! Feature preprocessing for the Solow statistical stack — scalers, encoders,
//! and feature construction.
//!
//! Every preprocessor exposes the classical `fit` / `transform` /
//! `fit_transform` / `inverse_transform` shape from
//! `preprocessing`, so a the reference practitioner can migrate a
//! pipeline call-for-call. The types are also composable through
//! [`solow-cv`](https://docs.rs/solow-cv)-style cross-validated scoring
//! and through the higher-level `solow-fit` formula surface.
//!
//! ## Modules
//!
//! * [`scalers`] — [`StandardScaler`], [`MinMaxScaler`], [`RobustScaler`],
//!   [`MaxAbsScaler`], [`Normalizer`]. All support `partial_fit`-style
//!   incremental statistics update where applicable and derive
//!   `Serialize` / `Deserialize` under the opt-in `serde` feature.
//! * [`encoders`] — [`LabelEncoder`] (1-D categorical `usize` ↔ label),
//!   [`OrdinalEncoder`] (per-column labels for a matrix), and
//!   [`OneHotEncoder`] with an optional `drop_first` for regression
//!   design-matrix compatibility.
//! * [`polynomial`] — [`PolynomialFeatures`] with `degree`,
//!   `interaction_only`, and `include_bias` matching the reference
//!   semantics exactly.
//! * [`discretize`] — [`KBinsDiscretizer`] with `Uniform`, `Quantile`,
//!   and `KMeans` binning strategies.
//!
//! Every preprocessor's `transform` is deterministic and side-effect-free;
//! there is no hidden RNG. `KBinsDiscretizer::KMeans` is the only
//! stochastic component and takes an explicit `seed`.
//!
//! ## Example — a full preprocessing chain
//!
//! ```
//! use ndarray::array;
//! use solow_preprocessing::{StandardScaler, PolynomialFeatures};
//!
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//!
//! // Standardize.
//! let scaler = StandardScaler::fit(x.view()).unwrap();
//! let x_std = scaler.transform(x.view()).unwrap();
//!
//! // Expand to polynomial features up to degree 2 (no bias column).
//! let poly = PolynomialFeatures::new(2).include_bias(false);
//! let x_poly = poly.fit_transform(x_std.view()).unwrap();
//!
//! assert_eq!(x_poly.ncols(), 5); // x0, x1, x0², x0·x1, x1²
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod binarizer;
pub mod discretize;
pub mod encoders;
pub mod function_transformer;
pub mod label_binarizer;
pub mod polynomial;
pub mod power;
pub mod scalers;
pub mod spline;
pub mod target_encoder;

pub use binarizer::Binarizer;
pub use discretize::{BinStrategy, KBinsDiscretizer};
pub use encoders::{LabelEncoder, OneHotEncoder, OrdinalEncoder};
pub use function_transformer::FunctionTransformer;
pub use label_binarizer::{LabelBinarizer, MultiLabelBinarizer};
pub use polynomial::PolynomialFeatures;
pub use power::{PowerMethod, PowerTransformer, QuantileOutput, QuantileTransformer};
pub use scalers::{MaxAbsScaler, MinMaxScaler, NormKind, Normalizer, RobustScaler, StandardScaler};
pub use spline::SplineTransformer;
pub use target_encoder::TargetEncoder;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::binarizer::Binarizer;
    pub use crate::discretize::{BinStrategy, KBinsDiscretizer};
    pub use crate::encoders::{LabelEncoder, OneHotEncoder, OrdinalEncoder};
    pub use crate::function_transformer::FunctionTransformer;
    pub use crate::label_binarizer::{LabelBinarizer, MultiLabelBinarizer};
    pub use crate::polynomial::PolynomialFeatures;
    pub use crate::scalers::{
        MaxAbsScaler, MinMaxScaler, NormKind, Normalizer, RobustScaler, StandardScaler,
    };
    pub use crate::spline::SplineTransformer;
    pub use crate::target_encoder::TargetEncoder;
}
