//! `LabelBinarizer` and `MultiLabelBinarizer` — turn integer / string
//! label vectors into a binary indicator matrix.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// LabelBinarizer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LabelBinarizer {
    /// Sorted unique labels seen at fit.
    pub classes: Vec<i64>,
    /// Whether the fit saw exactly 2 classes → returns a column vector.
    pub is_binary: bool,
}

impl LabelBinarizer {
    /// Fit.
    pub fn fit(y: &[i64]) -> Result<Self> {
        if y.is_empty() {
            return Err(Error::Value("LabelBinarizer::fit: empty label vector".into()));
        }
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort();
        classes.dedup();
        let is_binary = classes.len() == 2;
        Ok(Self { classes, is_binary })
    }

    /// Transform.
    pub fn transform(&self, y: &[i64]) -> Result<Array2<f64>> {
        let n = y.len();
        if self.is_binary {
            let mut out = Array2::<f64>::zeros((n, 1));
            for i in 0..n {
                if y[i] == self.classes[1] {
                    out[[i, 0]] = 1.0;
                }
            }
            Ok(out)
        } else {
            let k = self.classes.len();
            let mut out = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                if let Some(idx) = self.classes.iter().position(|&c| c == y[i]) {
                    out[[i, idx]] = 1.0;
                }
            }
            Ok(out)
        }
    }

    /// Inverse-transform: argmax by column, mapped back to labels.
    pub fn inverse_transform(&self, y: ArrayView2<'_, f64>) -> Result<Vec<i64>> {
        let n = y.nrows();
        if self.is_binary {
            if y.ncols() != 1 {
                return Err(Error::Shape("LabelBinarizer::inverse_transform: expected 1 column".into()));
            }
            Ok((0..n).map(|i| if y[[i, 0]] >= 0.5 { self.classes[1] } else { self.classes[0] }).collect())
        } else {
            if y.ncols() != self.classes.len() {
                return Err(Error::Shape("LabelBinarizer::inverse_transform: column count mismatch".into()));
            }
            let k = self.classes.len();
            Ok((0..n).map(|i| {
                let mut best = 0;
                let mut best_v = y[[i, 0]];
                for c in 1..k {
                    if y[[i, c]] > best_v {
                        best_v = y[[i, c]];
                        best = c;
                    }
                }
                self.classes[best]
            }).collect())
        }
    }
}

/// MultiLabelBinarizer — like LabelBinarizer but each sample can carry
/// multiple labels (represented as a `Vec<i64>`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MultiLabelBinarizer {
    /// Sorted unique labels seen at fit.
    pub classes: Vec<i64>,
}

impl MultiLabelBinarizer {
    /// Fit.
    pub fn fit(y: &[Vec<i64>]) -> Result<Self> {
        if y.is_empty() {
            return Err(Error::Value("MultiLabelBinarizer::fit: empty input".into()));
        }
        let mut classes: Vec<i64> = y.iter().flatten().copied().collect();
        classes.sort();
        classes.dedup();
        Ok(Self { classes })
    }

    /// Transform.
    pub fn transform(&self, y: &[Vec<i64>]) -> Result<Array2<f64>> {
        let n = y.len();
        let k = self.classes.len();
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for &lbl in &y[i] {
                if let Some(idx) = self.classes.iter().position(|&c| c == lbl) {
                    out[[i, idx]] = 1.0;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_binarizer_binary_returns_a_single_column() {
        let y = vec![0_i64, 1, 0, 1];
        let lb = LabelBinarizer::fit(&y).unwrap();
        let z = lb.transform(&y).unwrap();
        assert_eq!(z.shape(), &[4, 1]);
    }

    #[test]
    fn label_binarizer_multiclass_returns_k_columns() {
        let y = vec![0_i64, 1, 2, 0];
        let lb = LabelBinarizer::fit(&y).unwrap();
        let z = lb.transform(&y).unwrap();
        assert_eq!(z.shape(), &[4, 3]);
    }

    #[test]
    fn multilabel_binarizer_returns_a_correct_indicator_matrix() {
        let y = vec![vec![0_i64, 1], vec![1], vec![2, 0]];
        let mlb = MultiLabelBinarizer::fit(&y).unwrap();
        let z = mlb.transform(&y).unwrap();
        assert_eq!(z.shape(), &[3, 3]);
        assert_eq!(z[[0, 0]], 1.0);
        assert_eq!(z[[0, 1]], 1.0);
        assert_eq!(z[[0, 2]], 0.0);
    }
}
