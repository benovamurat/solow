//! # solow-discrete
//!
//! Discrete-choice and count regression models estimated by maximum likelihood:
//! [`Logit`], [`Probit`], and [`Poisson`]. Each model is fit with a full Newton
//! step using analytic log-likelihood, score, and Hessian, converging to the true
//! optimum so that results agree with the canonical reference to machine precision.
//!
//! ```
//! use ndarray::{array, Array2};
//! use solow_discrete::Logit;
//!
//! let mut x = Array2::<f64>::ones((5, 2));
//! x.column_mut(1)
//!     .assign(&array![0.1, -0.4, 1.2, 0.7, -1.1]);
//! let y = array![0.0, 0.0, 1.0, 1.0, 0.0];
//! let res = Logit::new(y, x).unwrap().fit().unwrap();
//! assert!(res.converged);
//! assert_eq!(res.params.len(), 2);
//! ```

mod conditional;
mod genpoisson;
mod model;
mod multinomial;
mod negbin;
mod ordered;
mod summary;
mod truncated;
mod zip;

pub use conditional::{ConditionalLogit, ConditionalModel, ConditionalPoisson, ConditionalResults};
pub use genpoisson::{GeneralizedPoisson, GeneralizedPoissonResults};
pub use model::{DiscreteResults, Logit, Poisson, Probit};
pub use multinomial::{MNLogit, MNLogitResults};
pub use negbin::{NegativeBinomial, NegativeBinomialResults};
pub use ordered::{Distr, OrderedModel, OrderedResults};
pub use truncated::{
    HurdleCountModel, HurdleCountResults, TruncatedLFPoisson, TruncatedLFPoissonResults,
};
pub use zip::{ZeroInflatedPoisson, ZeroInflatedPoissonResults};
