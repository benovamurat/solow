//! DictVectorizer — the reference feature-map dictionary vectoriser.
//!
//! Accepts `[HashMap<String, f64>]` rows and turns them into a dense
//! `(n_samples × n_features)` matrix, learning the feature vocabulary at
//! fit time.

use std::collections::BTreeMap;

use ndarray::Array2;
use solow_core::{Error, Result};

/// Fitted DictVectorizer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct DictVectorizer {
    /// Learned feature name → column index map (sorted).
    pub feature_names: Vec<String>,
    /// Whether missing keys are treated as zero (the reference default).
    pub sparse_default: f64,
}

impl DictVectorizer {
    /// Fit and remember the union of all keys across the training rows.
    pub fn fit(rows: &[BTreeMap<String, f64>]) -> Result<Self> {
        if rows.is_empty() {
            return Err(Error::Value("DictVectorizer::fit: no rows".into()));
        }
        let mut names: std::collections::BTreeSet<String> = Default::default();
        for row in rows {
            for k in row.keys() {
                names.insert(k.clone());
            }
        }
        Ok(Self {
            feature_names: names.into_iter().collect(),
            sparse_default: 0.0,
        })
    }

    /// Transform new rows.
    pub fn transform(&self, rows: &[BTreeMap<String, f64>]) -> Result<Array2<f64>> {
        let n = rows.len();
        let p = self.feature_names.len();
        let mut out = Array2::<f64>::from_elem((n, p), self.sparse_default);
        for (i, row) in rows.iter().enumerate() {
            for (name, &v) in row {
                if let Ok(idx) = self.feature_names.binary_search(name) {
                    out[[i, idx]] = v;
                }
            }
        }
        Ok(out)
    }

    /// Fit and transform in one call.
    pub fn fit_transform(rows: &[BTreeMap<String, f64>]) -> Result<(Self, Array2<f64>)> {
        let v = Self::fit(rows)?;
        let m = v.transform(rows)?;
        Ok((v, m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dict_vectorizer_learns_the_union_of_keys() {
        let mut r1: BTreeMap<String, f64> = Default::default();
        r1.insert("a".into(), 1.0);
        r1.insert("b".into(), 2.0);
        let mut r2: BTreeMap<String, f64> = Default::default();
        r2.insert("b".into(), 3.0);
        r2.insert("c".into(), 4.0);
        let v = DictVectorizer::fit(&[r1.clone(), r2.clone()]).unwrap();
        assert_eq!(v.feature_names, vec!["a", "b", "c"]);
        let m = v.transform(&[r1, r2]).unwrap();
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[0, 1]], 2.0);
        assert_eq!(m[[0, 2]], 0.0);
        assert_eq!(m[[1, 0]], 0.0);
        assert_eq!(m[[1, 1]], 3.0);
        assert_eq!(m[[1, 2]], 4.0);
    }
}
