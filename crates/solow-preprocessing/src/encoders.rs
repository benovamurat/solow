//! Categorical encoders.
//!
//! * [`LabelEncoder`] — 1-D categorical values (any hashable) ↔ dense
//!   `usize` indices, with a stable-sorted vocabulary so a given input
//!   distribution always produces the same encoding across runs.
//! * [`OrdinalEncoder`] — the multi-column generalisation for a full
//!   feature matrix; each column has its own [`LabelEncoder`].
//! * [`OneHotEncoder`] — expands each categorical column into a dense
//!   `k`-column binary indicator block. Supports `drop_first` for
//!   dummy-variable regression parity with `pandas.get_dummies(..., drop_first=True)`,
//!   and reports the fitted per-column categories so unseen categories
//!   at transform time raise a clean [`Error::Value`] instead of a panic.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use solow_core::{Error, Result};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// LabelEncoder
// ---------------------------------------------------------------------------

/// 1-D label ↔ `usize` encoder with a lexicographically-sorted vocabulary.
///
/// # Complexity
///
/// * `fit`: `O(n · log n)` time (sort + dedup), `O(k)` space where `k`
///   is the number of distinct classes.
/// * `transform`: `O(n · log k)` time via binary search.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LabelEncoder {
    /// Sorted, deduplicated class values as strings.
    /// Storing labels as owned `String`s keeps the encoder generic and
    /// `Serialize`-friendly without pulling in a hashable-trait bound.
    pub classes: Vec<String>,
}

impl LabelEncoder {
    /// Fit onto an iterable of labels (each rendered with `to_string`).
    pub fn fit<I, T>(labels: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: ToString,
    {
        let mut set: BTreeMap<String, ()> = BTreeMap::new();
        for l in labels {
            set.insert(l.to_string(), ());
        }
        Self {
            classes: set.into_keys().collect(),
        }
    }

    /// Number of distinct classes.
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Transform each label to its class index. Errors on an unseen label.
    pub fn transform<I, T>(&self, labels: I) -> Result<Array1<usize>>
    where
        I: IntoIterator<Item = T>,
        T: ToString,
    {
        let ll: Vec<String> = labels.into_iter().map(|l| l.to_string()).collect();
        let mut out = Array1::<usize>::zeros(ll.len());
        for (i, l) in ll.iter().enumerate() {
            let idx = self
                .classes
                .binary_search(l)
                .map_err(|_| Error::Value(format!("LabelEncoder: unseen label {l:?}")))?;
            out[i] = idx;
        }
        Ok(out)
    }

    /// Inverse-map class indices back to labels.
    pub fn inverse_transform(&self, encoded: ArrayView1<'_, usize>) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(encoded.len());
        for &c in encoded.iter() {
            if c >= self.classes.len() {
                return Err(Error::Value(format!(
                    "LabelEncoder::inverse_transform: class {c} is out of range for {}",
                    self.classes.len()
                )));
            }
            out.push(self.classes[c].clone());
        }
        Ok(out)
    }

    /// One-call fit + transform.
    pub fn fit_transform<I, T>(labels: I) -> (Self, Array1<usize>)
    where
        I: IntoIterator<Item = T>,
        T: ToString,
        T: Clone,
    {
        // Materialize once to avoid consuming the iterator twice.
        let materialized: Vec<T> = labels.into_iter().collect();
        let enc = Self::fit(materialized.iter().cloned());
        let arr = enc
            .transform(materialized.into_iter())
            .expect("fit vocabulary must cover every input label");
        (enc, arr)
    }
}

// ---------------------------------------------------------------------------
// OrdinalEncoder
// ---------------------------------------------------------------------------

/// Multi-column [`LabelEncoder`] — one per feature column.
///
/// Takes a `n × d` matrix of numeric categorical codes and returns a
/// `n × d` matrix of `usize` class indices per column. This is the
/// natural preprocessor for a tree-based estimator that expects
/// integer categorical inputs.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct OrdinalEncoder {
    /// One encoder per column, in column order.
    pub encoders: Vec<LabelEncoder>,
}

impl OrdinalEncoder {
    /// Fit onto a numeric-categorical matrix.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "OrdinalEncoder::fit: x must have at least one row and one column".into(),
            ));
        }
        let mut encoders = Vec::with_capacity(x.ncols());
        for j in 0..x.ncols() {
            let col: Vec<String> = x.column(j).iter().map(|v| v.to_string()).collect();
            encoders.push(LabelEncoder::fit(col.into_iter()));
        }
        Ok(Self { encoders })
    }

    /// Transform a matrix column-by-column.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<usize>> {
        if x.ncols() != self.encoders.len() {
            return Err(Error::Shape(format!(
                "OrdinalEncoder::transform: expected {} columns, got {}",
                self.encoders.len(),
                x.ncols()
            )));
        }
        let mut out = Array2::<usize>::zeros((x.nrows(), x.ncols()));
        for j in 0..x.ncols() {
            let col: Vec<String> = x.column(j).iter().map(|v| v.to_string()).collect();
            let enc = self.encoders[j].transform(col.into_iter())?;
            for i in 0..x.nrows() {
                out[[i, j]] = enc[i];
            }
        }
        Ok(out)
    }

    /// One-call fit + transform.
    pub fn fit_transform(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<usize>)> {
        let e = Self::fit(x)?;
        let t = e.transform(x)?;
        Ok((e, t))
    }
}

// ---------------------------------------------------------------------------
// OneHotEncoder
// ---------------------------------------------------------------------------

/// One-hot / dummy-variable encoder for categorical numeric input.
///
/// For a `n × d` matrix, produces a `n × Σⱼ kⱼ` (or `n × Σⱼ (kⱼ - 1)`
/// with `drop_first`) dense indicator matrix. Column ordering is
/// deterministic — for column `j` with sorted categories
/// `[c0, c1, …, ck-1]`, the encoded block appears in that order.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct OneHotEncoder {
    /// One `LabelEncoder` per input column, in column order.
    pub encoders: Vec<LabelEncoder>,
    /// Whether to drop the first category of every column (dummy-variable
    /// convention, avoids the classical `X'X` rank-deficiency of a full
    /// dummy set in a design matrix with an intercept).
    pub drop_first: bool,
}

impl OneHotEncoder {
    /// Fit — no drop by default.
    pub fn fit(x: ArrayView2<'_, f64>) -> Result<Self> {
        Self::fit_with(x, false)
    }

    /// Fit with explicit `drop_first` flag.
    pub fn fit_with(x: ArrayView2<'_, f64>, drop_first: bool) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "OneHotEncoder::fit: x must have at least one row and one column".into(),
            ));
        }
        let ord = OrdinalEncoder::fit(x)?;
        Ok(Self {
            encoders: ord.encoders,
            drop_first,
        })
    }

    /// Compute the total number of output columns.
    pub fn n_output_columns(&self) -> usize {
        self.encoders
            .iter()
            .map(|e| {
                let k = e.n_classes();
                if self.drop_first {
                    k.saturating_sub(1)
                } else {
                    k
                }
            })
            .sum()
    }

    /// Transform.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.encoders.len() {
            return Err(Error::Shape(format!(
                "OneHotEncoder::transform: expected {} columns, got {}",
                self.encoders.len(),
                x.ncols()
            )));
        }
        let cols_out = self.n_output_columns();
        let mut out = Array2::<f64>::zeros((x.nrows(), cols_out));
        // For each row and each input column, set the appropriate output col to 1.
        let mut base_col = 0usize;
        for (j, enc) in self.encoders.iter().enumerate() {
            let k = enc.n_classes();
            let drop = if self.drop_first { 1 } else { 0 };
            let block_width = k - drop;
            let col: Vec<String> = x.column(j).iter().map(|v| v.to_string()).collect();
            let codes = enc.transform(col.into_iter())?;
            for i in 0..x.nrows() {
                let c = codes[i];
                if c < drop {
                    // Dropped baseline — everything in this block is zero.
                    continue;
                }
                out[[i, base_col + (c - drop)]] = 1.0;
            }
            base_col += block_width;
        }
        Ok(out)
    }

    /// One-call fit + transform.
    pub fn fit_transform(x: ArrayView2<'_, f64>) -> Result<(Self, Array2<f64>)> {
        let e = Self::fit(x)?;
        let t = e.transform(x)?;
        Ok((e, t))
    }
}

// Silence "unused import" when the module compiles without exercising it.
#[allow(dead_code)]
fn _touch(_: Axis) {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn label_encoder_round_trip() {
        let (enc, codes) = LabelEncoder::fit_transform(["b", "a", "c", "a", "b"]);
        assert_eq!(
            enc.classes,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(codes.to_vec(), vec![1, 0, 2, 0, 1]);
        let back = enc.inverse_transform(codes.view()).unwrap();
        assert_eq!(back, vec!["b", "a", "c", "a", "b"]);
    }

    #[test]
    fn label_encoder_rejects_unseen() {
        let enc = LabelEncoder::fit(["a", "b", "c"]);
        assert!(enc.transform(["a", "d"].into_iter()).is_err());
    }

    #[test]
    fn one_hot_no_drop_produces_full_block() {
        let x = array![[0.0], [1.0], [2.0], [0.0]];
        let (enc, oh) = OneHotEncoder::fit_transform(x.view()).unwrap();
        assert_eq!(enc.n_output_columns(), 3);
        assert_eq!(oh.dim(), (4, 3));
        // Row 0 encodes category "0" → the first column of the block.
        assert_eq!(oh.row(0).to_vec(), vec![1.0, 0.0, 0.0]);
        assert_eq!(oh.row(1).to_vec(), vec![0.0, 1.0, 0.0]);
        assert_eq!(oh.row(2).to_vec(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn one_hot_drop_first_drops_baseline() {
        let x = array![[0.0], [1.0], [2.0]];
        let enc = OneHotEncoder::fit_with(x.view(), true).unwrap();
        let oh = enc.transform(x.view()).unwrap();
        assert_eq!(oh.dim(), (3, 2));
        assert_eq!(oh.row(0).to_vec(), vec![0.0, 0.0]); // baseline
        assert_eq!(oh.row(1).to_vec(), vec![1.0, 0.0]);
        assert_eq!(oh.row(2).to_vec(), vec![0.0, 1.0]);
    }
}
