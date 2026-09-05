//! # solow-naive-bayes
//!
//! Naive Bayes classifiers under the standard conditional-independence
//! assumption `p(x | y) = ∏_j p(x_j | y)`. Four families ship here,
//! each specialised to a feature domain:
//!
//! * [`GaussianNB`] — continuous features with per-class-per-feature
//!   Gaussian likelihood. Uses the numerically-safe log-sum-exp form
//!   for the posterior and a shared variance-smoothing term
//!   (`var_smoothing = 1e-9`) matching the reference `GaussianNB` default.
//! * [`MultinomialNB`] — integer count features with per-class
//!   multinomial likelihood and Laplace / Lidstone smoothing.
//! * [`BernoulliNB`] — binary features `{0, 1}` with Bernoulli
//!   likelihood and smoothing.
//! * [`ComplementNB`] — Rennie et al. (2003) complement multinomial,
//!   more robust on unbalanced text corpora.
//!
//! All four expose `fit` / `predict` / `predict_proba` / `predict_log_proba`
//! and are deterministic — there's no random component.
//!
//! # References
//!
//! * Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction
//!   to Information Retrieval*, §13.
//! * Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003).
//!   *Tackling the poor assumptions of naive Bayes text classifiers.*
//!   ICML 2003, 616-623.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bernoulli;
pub mod categorical;
pub mod complement;
pub mod gaussian;
pub mod multinomial;

pub use bernoulli::BernoulliNB;
pub use categorical::CategoricalNB;
pub use complement::ComplementNB;
pub use gaussian::GaussianNB;
pub use multinomial::MultinomialNB;

/// Commonly-used imports.
pub mod prelude {
    pub use crate::bernoulli::BernoulliNB;
    pub use crate::complement::ComplementNB;
    pub use crate::gaussian::GaussianNB;
    pub use crate::multinomial::MultinomialNB;
}
