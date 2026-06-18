//! A labeled regression-results summary table.
//!
//! [`RegressionSummary`] collects the values any model's results can supply --
//! parameter names and their estimates, standard errors, test statistics,
//! p-values, and confidence intervals, plus a handful of header statistics --
//! and renders them as a two-block fixed-width text table:
//!
//! * a *header block* of key statistics (model name, observations, degrees of
//!   freedom, R-squared, F-statistic, log-likelihood, AIC, BIC), and
//! * a *coefficient table* with columns `coef`, `std err`, the test statistic
//!   (`t` or `z`), `P>|t|` (or `P>|z|`), and the two confidence-interval
//!   bounds.
//!
//! The layout is Solow's own; only the numbers are ever cross-checked against
//! a reference implementation.

use crate::format::{format_fixed, format_g, format_pvalue};
use crate::table::{Align, SummaryTable};
use std::fmt;

/// Whether the coefficient table reports a `t` statistic or a `z` statistic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatKind {
    /// Student's t (column header `t`, p-value header `P>|t|`).
    T,
    /// Standard normal z (column header `z`, p-value header `P>|z|`).
    Z,
}

impl StatKind {
    fn stat_label(self) -> &'static str {
        match self {
            StatKind::T => "t",
            StatKind::Z => "z",
        }
    }

    fn pvalue_label(self) -> &'static str {
        match self {
            StatKind::T => "P>|t|",
            StatKind::Z => "P>|z|",
        }
    }
}

/// Header statistics shown in the top block of the summary.
///
/// All fields are optional: a field set to `None` is simply omitted from the
/// rendered header, so models that do not produce (say) an F-statistic can
/// still build a summary.
#[derive(Clone, Debug, Default)]
pub struct HeaderStats {
    pub model: Option<String>,
    pub dep_variable: Option<String>,
    pub method: Option<String>,
    pub nobs: Option<f64>,
    pub df_model: Option<f64>,
    pub df_resid: Option<f64>,
    pub rsquared: Option<f64>,
    pub rsquared_adj: Option<f64>,
    pub fvalue: Option<f64>,
    pub f_pvalue: Option<f64>,
    pub llf: Option<f64>,
    pub aic: Option<f64>,
    pub bic: Option<f64>,
}

impl HeaderStats {
    /// Start from an all-`None` header.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Builder for a labeled regression-results summary.
#[derive(Clone, Debug)]
pub struct RegressionSummary {
    title: String,
    stat_kind: StatKind,
    alpha: f64,
    names: Vec<String>,
    params: Vec<f64>,
    bse: Vec<f64>,
    tvalues: Vec<f64>,
    pvalues: Vec<f64>,
    conf_int: Vec<(f64, f64)>,
    header: HeaderStats,
}

impl RegressionSummary {
    /// Create a summary from the per-parameter vectors and a confidence
    /// interval given as a `k x 2` set of `(lower, upper)` rows.
    ///
    /// All slices must share the parameter count `k`. The default confidence
    /// level is 95% (`alpha = 0.05`); use [`with_alpha`](Self::with_alpha) to
    /// change the displayed CI bounds.
    ///
    /// # Panics
    ///
    /// Panics if the input lengths do not all match.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        names: &[impl AsRef<str>],
        params: &[f64],
        bse: &[f64],
        tvalues: &[f64],
        pvalues: &[f64],
        conf_int: &[(f64, f64)],
        header: HeaderStats,
    ) -> Self {
        let k = names.len();
        assert_eq!(params.len(), k, "params length must match name count");
        assert_eq!(bse.len(), k, "bse length must match name count");
        assert_eq!(tvalues.len(), k, "tvalues length must match name count");
        assert_eq!(pvalues.len(), k, "pvalues length must match name count");
        assert_eq!(
            conf_int.len(),
            k,
            "conf_int row count must match name count"
        );
        RegressionSummary {
            title: "Regression Results".to_string(),
            stat_kind: StatKind::T,
            alpha: 0.05,
            names: names.iter().map(|s| s.as_ref().to_string()).collect(),
            params: params.to_vec(),
            bse: bse.to_vec(),
            tvalues: tvalues.to_vec(),
            pvalues: pvalues.to_vec(),
            conf_int: conf_int.to_vec(),
            header,
        }
    }

    /// Override the table title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Report a `z` statistic (and `P>|z|`) instead of a `t` statistic.
    pub fn with_stat_kind(mut self, kind: StatKind) -> Self {
        self.stat_kind = kind;
        self
    }

    /// Set the confidence level used for the CI column headers (e.g. `0.05`
    /// for a 95% interval renders `[0.025  0.975]`). This does not recompute
    /// the bounds -- those are supplied to [`new`](Self::new) -- it only sets
    /// the displayed quantile labels.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Build the header block as a two-column key/value table.
    fn header_table(&self) -> SummaryTable {
        let mut t = SummaryTable::new()
            .title(self.title.clone())
            .header(["Statistic", "Value"])
            .aligns([Align::Left, Align::Right]);

        let h = &self.header;
        if let Some(m) = &h.model {
            t.push_row(["Model:".to_string(), m.clone()]);
        }
        if let Some(d) = &h.dep_variable {
            t.push_row(["Dep. Variable:".to_string(), d.clone()]);
        }
        if let Some(m) = &h.method {
            t.push_row(["Method:".to_string(), m.clone()]);
        }
        if let Some(n) = h.nobs {
            t.push_row(["No. Observations:".to_string(), format_fixed(n, 0)]);
        }
        if let Some(d) = h.df_model {
            t.push_row(["Df Model:".to_string(), format_fixed(d, 0)]);
        }
        if let Some(d) = h.df_resid {
            t.push_row(["Df Residuals:".to_string(), format_fixed(d, 0)]);
        }
        if let Some(r) = h.rsquared {
            t.push_row(["R-squared:".to_string(), format_fixed(r, 3)]);
        }
        if let Some(r) = h.rsquared_adj {
            t.push_row(["Adj. R-squared:".to_string(), format_fixed(r, 3)]);
        }
        if let Some(fv) = h.fvalue {
            t.push_row(["F-statistic:".to_string(), format_g(fv, 4)]);
        }
        if let Some(fp) = h.f_pvalue {
            t.push_row(["Prob (F-statistic):".to_string(), format_g(fp, 4)]);
        }
        if let Some(l) = h.llf {
            t.push_row(["Log-Likelihood:".to_string(), format_g(l, 5)]);
        }
        if let Some(a) = h.aic {
            t.push_row(["AIC:".to_string(), format_g(a, 5)]);
        }
        if let Some(b) = h.bic {
            t.push_row(["BIC:".to_string(), format_g(b, 5)]);
        }
        t
    }

    /// CI column labels, e.g. `[0.025` and `0.975]` for `alpha = 0.05`.
    fn ci_labels(&self) -> (String, String) {
        let lo = self.alpha / 2.0;
        let hi = 1.0 - self.alpha / 2.0;
        (
            format!("[{}", format_fixed(lo, 3)),
            format!("{}]", format_fixed(hi, 3)),
        )
    }

    /// Build the coefficient table.
    fn coef_table(&self) -> SummaryTable {
        let (ci_lo, ci_hi) = self.ci_labels();
        let mut t = SummaryTable::new()
            .header([
                "".to_string(),
                "coef".to_string(),
                "std err".to_string(),
                self.stat_kind.stat_label().to_string(),
                self.stat_kind.pvalue_label().to_string(),
                ci_lo,
                ci_hi,
            ])
            .aligns([
                Align::Left,
                Align::Right,
                Align::Right,
                Align::Right,
                Align::Right,
                Align::Right,
                Align::Right,
            ]);

        for i in 0..self.names.len() {
            let (lo, hi) = self.conf_int[i];
            t.push_row([
                self.names[i].clone(),
                format_g(self.params[i], 4),
                format_g(self.bse[i], 4),
                format_fixed(self.tvalues[i], 3),
                format_pvalue(self.pvalues[i]),
                format_fixed(lo, 3),
                format_fixed(hi, 3),
            ]);
        }
        t
    }
}

impl fmt::Display for RegressionSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\n\n{}", self.header_table(), self.coef_table())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RegressionSummary {
        let names = ["const", "x1", "x2"];
        let params = [1.49339521, -1.90617444, 0.71064927];
        let bse = [0.0728888, 0.08152201, 0.07324544];
        let tvalues = [20.48867957, -23.38232831, 9.70229997];
        let pvalues = [8.69594441e-22, 9.22865862e-24, 1.03889592e-11];
        let conf_int = [
            (1.345708481321229, 1.6410819449282237),
            (-2.0713537205255537, -1.7409951503714414),
            (0.562239906015244, 0.859058641160409),
        ];
        let header = HeaderStats {
            model: Some("OLS".to_string()),
            nobs: Some(40.0),
            df_model: Some(2.0),
            df_resid: Some(37.0),
            rsquared: Some(0.9505900603362948),
            rsquared_adj: Some(0.9479192527869054),
            fvalue: Some(355.9185911967316),
            f_pvalue: Some(6.848019459935387e-25),
            llf: Some(-19.95229542493003),
            aic: Some(45.90459084986006),
            bic: Some(50.97122921220186),
            ..HeaderStats::new()
        };
        RegressionSummary::new(&names, &params, &bse, &tvalues, &pvalues, &conf_int, header)
    }

    #[test]
    fn renders_all_parameter_names() {
        let s = sample().to_string();
        assert!(s.contains("const"));
        assert!(s.contains("x1"));
        assert!(s.contains("x2"));
    }

    #[test]
    fn header_contains_key_stats() {
        let s = sample().to_string();
        assert!(s.contains("R-squared:"));
        assert!(s.contains("0.951"));
        assert!(s.contains("Adj. R-squared:"));
        assert!(s.contains("AIC:"));
        assert!(s.contains("BIC:"));
        assert!(s.contains("Model:"));
        assert!(s.contains("OLS"));
    }

    #[test]
    fn coef_columns_present() {
        let s = sample().to_string();
        assert!(s.contains("coef"));
        assert!(s.contains("std err"));
        // default is t statistic
        assert!(s.contains("P>|t|"));
        assert!(s.contains("[0.025"));
        assert!(s.contains("0.975]"));
    }

    #[test]
    fn z_kind_changes_labels() {
        let s = sample().with_stat_kind(StatKind::Z).to_string();
        assert!(s.contains("P>|z|"));
        assert!(!s.contains("P>|t|"));
    }

    #[test]
    fn coef_values_formatted() {
        let s = sample().to_string();
        // coef ~4 sig figs
        assert!(s.contains("1.493"), "expected const coef; got:\n{s}");
        assert!(s.contains("-1.906"));
        // std err
        assert!(s.contains("0.07289"));
        // t to 3 decimals
        assert!(s.contains("20.489"));
        assert!(s.contains("-23.382"));
        // tiny p-value collapses
        assert!(s.contains("0.000"));
    }

    #[test]
    #[should_panic(expected = "params length")]
    fn mismatched_lengths_panic() {
        let names = ["a", "b"];
        let params = [1.0];
        let bse = [1.0, 1.0];
        let tv = [1.0, 1.0];
        let pv = [0.1, 0.1];
        let ci = [(0.0, 1.0), (0.0, 1.0)];
        let _ = RegressionSummary::new(&names, &params, &bse, &tv, &pv, &ci, HeaderStats::new());
    }
}
