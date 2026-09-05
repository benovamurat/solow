//! # solow-text
//!
//! Text feature extraction — bag-of-words and TF-IDF matrices from
//! raw string corpora.
//!
//! * [`CountVectorizer`] — `feature_extraction.text.CountVectorizer`
//!   at API-parity: `min_df` / `max_df` filtering, `lowercase`, word
//!   `ngram_range = (n_min, n_max)`, and an optional caller-supplied
//!   stop-word set.
//! * [`TfidfVectorizer`] — combines [`CountVectorizer`] with the
//!   Salton-Buckley (1988) TF-IDF weighting `tf · idf` where
//!   `idf(t) = log((1 + n) / (1 + df(t))) + 1` (the reference default,
//!   `smooth_idf=True`, `sublinear_tf=False`). Rows are L2-normalised.
//!
//! Tokenisation is a Unicode-word regex substitute: lowercase (if
//! enabled), keep runs of alphanumerics length ≥ 2, drop everything
//! else. This matches the reference default `token_pattern`
//! `r"(?u)\b\w\w+\b"` on plain ASCII / Latin-1 input.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dict;
pub mod hashing;
pub mod vectorizer;

pub use dict::DictVectorizer;
pub use hashing::{FeatureHasher, HashingVectorizer};
pub use vectorizer::{CountVectorizer, TfidfVectorizer};

/// Commonly-used imports.
pub mod prelude {
    pub use crate::dict::DictVectorizer;
    pub use crate::hashing::{FeatureHasher, HashingVectorizer};
    pub use crate::vectorizer::{CountVectorizer, TfidfVectorizer};
}
