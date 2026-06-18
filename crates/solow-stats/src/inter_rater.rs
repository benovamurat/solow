//! Inter-rater agreement measures.
//!
//! Provides Cohen's kappa for two raters ([`cohens_kappa`]) with its asymptotic
//! variance, confidence interval and zero-test, Fleiss'/Randolph's kappa for
//! many raters ([`fleiss_kappa`]), and the [`aggregate_raters`] helper that
//! turns a `(subject, rater)` assignment matrix into the `(subject, category
//! counts)` form expected by [`fleiss_kappa`]. Mirrors the reference
//! `inter_rater` module.

use ndarray::Array2;
use solow_distributions::{norm_isf, norm_sf};

/// Chance-correction convention used by [`fleiss_kappa`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FleissMethod {
    /// Fleiss' kappa: the chance outcome uses the sample marginal of categories.
    Fleiss,
    /// Randolph's (uniform) kappa: the chance outcome assumes a uniform
    /// distribution over categories.
    Randolph,
}

/// Aggregate a `(subject, rater)` assignment matrix into category counts.
///
/// `data` has subjects in rows and raters in columns; each entry is a category
/// label. The labels are mapped to consecutive integers `0..n_cat-1` (only
/// levels with non-zero counts are kept) and the result is a `(n_subject,
/// n_cat)` matrix whose `[i, c]` entry counts how many raters assigned category
/// `c` to subject `i`, together with the distinct category labels in sorted
/// order. Mirrors the reference `aggregate_raters` (with `n_cat=None`).
pub fn aggregate_raters(data: &Array2<f64>) -> (Array2<f64>, Vec<f64>) {
    // Distinct category labels, sorted, deduplicated.
    let mut cats: Vec<f64> = data.iter().copied().collect();
    cats.sort_by(|a, b| a.total_cmp(b));
    cats.dedup();
    let n_cat = cats.len();
    let n_rows = data.nrows();

    let cat_index = |v: f64| -> usize {
        cats.iter()
            .position(|&c| c == v)
            .expect("category present in label set")
    };

    let mut tt = Array2::<f64>::zeros((n_rows, n_cat));
    for (i, row) in data.rows().into_iter().enumerate() {
        for &v in row {
            tt[[i, cat_index(v)]] += 1.0;
        }
    }
    (tt, cats)
}

/// Fleiss' or Randolph's kappa for multi-rater agreement.
///
/// `table` has subjects in rows and category counts in columns (the output of
/// [`aggregate_raters`]); every subject must be rated the same number of times.
/// Method [`FleissMethod::Fleiss`] defines the chance agreement from the sample
/// category marginal, [`FleissMethod::Randolph`] from a uniform category
/// distribution. Mirrors the reference `fleiss_kappa`.
pub fn fleiss_kappa(table: &Array2<f64>, method: FleissMethod) -> f64 {
    let n_cat = table.ncols() as f64;
    let n_total: f64 = table.sum();
    // n_rat: number of ratings per subject (assumed constant, == row max sum).
    let n_rat = table
        .rows()
        .into_iter()
        .map(|r| r.sum())
        .fold(f64::NEG_INFINITY, f64::max);

    // Marginal category frequencies.
    let p_cat: Vec<f64> = (0..table.ncols())
        .map(|j| table.column(j).sum() / n_total)
        .collect();

    // Per-subject agreement.
    let mut p_sum = 0.0;
    for row in table.rows() {
        let sq: f64 = row.iter().map(|&v| v * v).sum();
        p_sum += (sq - n_rat) / (n_rat * (n_rat - 1.0));
    }
    let p_mean = p_sum / table.nrows() as f64;

    let p_mean_exp = match method {
        FleissMethod::Fleiss => p_cat.iter().map(|&p| p * p).sum::<f64>(),
        FleissMethod::Randolph => 1.0 / n_cat,
    };

    (p_mean - p_mean_exp) / (1.0 - p_mean_exp)
}

/// Results of [`cohens_kappa`].
#[derive(Debug, Clone)]
pub struct KappaResults {
    /// Cohen's (simple) kappa coefficient.
    pub kappa: f64,
    /// Maximum attainable kappa given the marginals.
    pub kappa_max: f64,
    /// Asymptotic variance of kappa.
    pub var_kappa: f64,
    /// Asymptotic variance of kappa under H0: kappa = 0.
    pub var_kappa0: f64,
    /// Asymptotic standard error of kappa, `sqrt(var_kappa)`.
    pub std_kappa: f64,
    /// Standard error under H0, `sqrt(var_kappa0)`.
    pub std_kappa0: f64,
    /// Test statistic `kappa / std_kappa0` for H0: kappa = 0 (standard normal).
    pub z_value: f64,
    /// One-sided p-value for H0: kappa = 0 vs H1: kappa > 0.
    pub pvalue_one_sided: f64,
    /// Two-sided p-value for H0: kappa = 0 vs H1: kappa != 0.
    pub pvalue_two_sided: f64,
    /// Lower `(1 - 2*alpha)` confidence limit for kappa.
    pub kappa_low: f64,
    /// Upper `(1 - 2*alpha)` confidence limit for kappa.
    pub kappa_upp: f64,
}

/// Cohen's (simple) kappa with variance, confidence interval and zero-test.
///
/// `table` is a square contingency matrix of two raters (rater 1 in rows, rater
/// 2 in columns). `alpha` is the one-sided tail probability for the confidence
/// interval (the reference default is `0.025`, giving a 95% interval). Mirrors
/// the unweighted branch of the reference `cohens_kappa`.
pub fn cohens_kappa(table: &Array2<f64>, alpha: f64) -> KappaResults {
    let n = table.nrows();
    let nobs: f64 = table.sum();

    // Observed agreement on the diagonal.
    let agree: f64 = (0..n).map(|i| table[[i, i]]).sum();

    // Probabilities and marginals.
    let freq_row: Vec<f64> = (0..n).map(|i| table.row(i).sum() / nobs).collect();
    let freq_col: Vec<f64> = (0..n).map(|j| table.column(j).sum() / nobs).collect();
    // prob_exp[i, j] = freq_col[j] * freq_row[i]
    let agree_exp: f64 = (0..n).map(|i| freq_col[i] * freq_row[i]).sum();

    let kappa = (agree / nobs - agree_exp) / (1.0 - agree_exp);

    // Asymptotic variance (SAS / Fleiss formulas).
    let probs_diag: Vec<f64> = (0..n).map(|i| table[[i, i]] / nobs).collect();
    let mut term_a = 0.0;
    for i in 0..n {
        let inner = 1.0 - (freq_row[i] + freq_col[i]) * (1.0 - kappa);
        term_a += probs_diag[i] * inner * inner;
    }
    let mut term_b = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Reference: term_b[i,j] = probs[i,j] * (freq_col[i] + freq_row[j])^2.
            let inner = freq_col[i] + freq_row[j];
            term_b += (table[[i, j]] / nobs) * inner * inner;
        }
    }
    term_b *= (1.0 - kappa) * (1.0 - kappa);
    let term_c = (kappa - agree_exp * (1.0 - kappa)).powi(2);
    let var_kappa = (term_a + term_b - term_c) / ((1.0 - agree_exp).powi(2) * nobs);

    // Variance under H0: kappa = 0.
    let term_c0: f64 = (0..n)
        .map(|i| freq_col[i] * freq_row[i] * (freq_col[i] + freq_row[i]))
        .sum();
    let var_kappa0 =
        (agree_exp + agree_exp * agree_exp - term_c0) / ((1.0 - agree_exp).powi(2) * nobs);

    let kappa_max =
        ((0..n).map(|i| freq_row[i].min(freq_col[i])).sum::<f64>() - agree_exp) / (1.0 - agree_exp);

    let std_kappa = var_kappa.sqrt();
    let std_kappa0 = var_kappa0.sqrt();
    let z_value = kappa / std_kappa0;
    let pvalue_one_sided = norm_sf(z_value);
    let pvalue_two_sided = norm_sf(z_value.abs()) * 2.0;
    let delta = norm_isf(alpha) * std_kappa;

    KappaResults {
        kappa,
        kappa_max,
        var_kappa,
        var_kappa0,
        std_kappa,
        std_kappa0,
        z_value,
        pvalue_one_sided,
        pvalue_two_sided,
        kappa_low: kappa - delta,
        kappa_upp: kappa + delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn perfect_agreement_kappa_one() {
        let t = array![[10.0, 0.0], [0.0, 15.0]];
        let r = cohens_kappa(&t, 0.025);
        assert!((r.kappa - 1.0).abs() < 1e-12);
    }

    #[test]
    fn aggregate_then_fleiss_runs() {
        // 3 subjects, 4 raters, categories {0, 1, 2}.
        let data = array![
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 2.0, 2.0],
            [0.0, 1.0, 2.0, 0.0]
        ];
        let (tt, cats) = aggregate_raters(&data);
        assert_eq!(cats, vec![0.0, 1.0, 2.0]);
        assert_eq!(tt.dim(), (3, 3));
        let k = fleiss_kappa(&tt, FleissMethod::Fleiss);
        assert!(k.is_finite());
    }
}
