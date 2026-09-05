//! SimpleImputer — replace missing values (represented as `NaN`) with a
//! per-column statistic (mean, median, most-frequent, or a caller-supplied
//! constant).

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Imputation strategy.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimpleStrategy {
    /// Column mean of observed values.
    Mean,
    /// Column median of observed values.
    Median,
    /// Most-frequent observed value in the column.
    MostFrequent,
    /// A caller-supplied constant.
    Constant(f64),
}

/// Fitted SimpleImputer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleImputer {
    /// Per-column fill value.
    pub statistics: Array1<f64>,
    /// Strategy used.
    pub strategy: SimpleStrategy,
    /// Column count at fit time.
    pub n_features_in: usize,
}

impl SimpleImputer {
    /// Fit with the given strategy.
    pub fn fit(x: ArrayView2<'_, f64>, strategy: SimpleStrategy) -> Result<Self> {
        let d = x.ncols();
        let n = x.nrows();
        if n == 0 || d == 0 {
            return Err(Error::Value("SimpleImputer: empty input".into()));
        }
        let mut stats = Array1::<f64>::zeros(d);
        for j in 0..d {
            let observed: Vec<f64> = (0..n).map(|i| x[[i, j]]).filter(|v| v.is_finite()).collect();
            stats[j] = match strategy {
                SimpleStrategy::Mean => {
                    if observed.is_empty() {
                        0.0
                    } else {
                        observed.iter().sum::<f64>() / observed.len() as f64
                    }
                }
                SimpleStrategy::Median => {
                    if observed.is_empty() {
                        0.0
                    } else {
                        let mut sorted = observed.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let m = sorted.len();
                        if m % 2 == 0 {
                            0.5 * (sorted[m / 2 - 1] + sorted[m / 2])
                        } else {
                            sorted[m / 2]
                        }
                    }
                }
                SimpleStrategy::MostFrequent => {
                    if observed.is_empty() {
                        0.0
                    } else {
                        let mut counts: std::collections::BTreeMap<u64, (f64, usize)> =
                            Default::default();
                        for &v in &observed {
                            let key = v.to_bits();
                            let e = counts.entry(key).or_insert((v, 0));
                            e.1 += 1;
                        }
                        counts.values().max_by_key(|(_, c)| *c).unwrap().0
                    }
                }
                SimpleStrategy::Constant(c) => c,
            };
        }
        Ok(Self {
            statistics: stats,
            strategy,
            n_features_in: d,
        })
    }

    /// Transform: replace `NaN`s with the fitted statistics.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_in {
            return Err(Error::Shape("SimpleImputer::transform: column count mismatch".into()));
        }
        let n = x.nrows();
        let d = x.ncols();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                out[[i, j]] = if x[[i, j]].is_finite() {
                    x[[i, j]]
                } else {
                    self.statistics[j]
                };
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn simple_imputer_fills_missing_with_column_mean() {
        let x = array![
            [1.0_f64, f64::NAN, 3.0],
            [2.0, 4.0, f64::NAN],
            [f64::NAN, 6.0, 7.0]
        ];
        let m = SimpleImputer::fit(x.view(), SimpleStrategy::Mean).unwrap();
        let z = m.transform(x.view()).unwrap();
        assert!((z[[2, 0]] - 1.5).abs() < 1e-12);
        assert!((z[[0, 1]] - 5.0).abs() < 1e-12);
        assert!((z[[1, 2]] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn simple_imputer_fills_missing_with_constant() {
        let x = array![[1.0_f64, f64::NAN], [f64::NAN, 2.0]];
        let m = SimpleImputer::fit(x.view(), SimpleStrategy::Constant(-1.0)).unwrap();
        let z = m.transform(x.view()).unwrap();
        assert_eq!(z[[0, 1]], -1.0);
        assert_eq!(z[[1, 0]], -1.0);
    }
}
