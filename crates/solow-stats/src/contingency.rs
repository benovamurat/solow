//! Contingency-table analysis: chi-squared test of nominal association.
//!
//! Mirrors the reference `contingency_tables.Table.test_nominal_association`:
//! the expected counts under independence are the outer product of the row and
//! column marginals scaled by the grand total; the chi-squared statistic is the
//! sum of Pearson contributions `(obs − exp)² / exp` with
//! `(rows − 1)(cols − 1)` degrees of freedom (no continuity correction).

use ndarray::Array2;
use solow_distributions::chi2_sf;

/// Result of a contingency-table chi-squared test of independence.
#[derive(Debug, Clone)]
pub struct ContingencyResult {
    /// Pearson chi-squared statistic.
    pub statistic: f64,
    /// Degrees of freedom `(rows − 1)(cols − 1)`.
    pub df: usize,
    /// p-value from the chi-squared distribution.
    pub pvalue: f64,
    /// Expected cell counts under independence.
    pub expected: Array2<f64>,
}

/// A two-way contingency table of observed counts.
#[derive(Debug, Clone)]
pub struct Table {
    table: Array2<f64>,
}

impl Table {
    /// Build a table from a matrix of observed counts.
    pub fn new(table: Array2<f64>) -> Self {
        Table { table }
    }

    /// Estimated marginal probability distributions `(row, col)`.
    pub fn marginal_probabilities(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.table.sum();
        let row: Vec<f64> = self
            .table
            .sum_axis(ndarray::Axis(1))
            .iter()
            .map(|&v| v / n)
            .collect();
        let col: Vec<f64> = self
            .table
            .sum_axis(ndarray::Axis(0))
            .iter()
            .map(|&v| v / n)
            .collect();
        (row, col)
    }

    /// Fitted (expected) cell counts under the independence model.
    pub fn fittedvalues(&self) -> Array2<f64> {
        let (row, col) = self.marginal_probabilities();
        let total = self.table.sum();
        let (r, c) = self.table.dim();
        let mut fit = Array2::<f64>::zeros((r, c));
        for i in 0..r {
            for j in 0..c {
                fit[[i, j]] = total * row[i] * col[j];
            }
        }
        fit
    }

    /// Chi-squared test of independence between rows and columns.
    pub fn test_nominal_association(&self) -> ContingencyResult {
        let expected = self.fittedvalues();
        let (r, c) = self.table.dim();
        let mut statistic = 0.0;
        for i in 0..r {
            for j in 0..c {
                let e = expected[[i, j]];
                let o = self.table[[i, j]];
                let resid = (o - e) / e.sqrt();
                statistic += resid * resid;
            }
        }
        let df = (r - 1) * (c - 1);
        let pvalue = chi2_sf(statistic, df as f64);
        ContingencyResult {
            statistic,
            df,
            pvalue,
            expected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn expected_row_sums_match_observed() {
        let t = Table::new(array![[10.0, 20.0, 30.0], [6.0, 9.0, 17.0]]);
        let exp = t.fittedvalues();
        // Row marginals of expected equal those of observed.
        for i in 0..2 {
            let so: f64 = (0..3).map(|j| t.table[[i, j]]).sum();
            let se: f64 = (0..3).map(|j| exp[[i, j]]).sum();
            assert!((so - se).abs() < 1e-9);
        }
    }

    #[test]
    fn statistic_nonnegative() {
        let t = Table::new(array![[10.0, 20.0, 30.0], [6.0, 9.0, 17.0]]);
        let res = t.test_nominal_association();
        assert!(res.statistic >= 0.0);
        assert_eq!(res.df, 2);
    }
}
