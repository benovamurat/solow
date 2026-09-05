//! Continuous distributions used for statistical inference: the normal,
//! Student-t, F, and chi-squared families. Each provides `pdf`, `cdf`, `sf`
//! (survival), `ppf` (quantile / inverse CDF), and `isf`.

use crate::special::{betainc, betaincinv, erfc, erfinv, gammainc, gammaincc, gammaincinv, lgamma};
use std::f64::consts::PI;

const SQRT_2: f64 = std::f64::consts::SQRT_2;

// ---------------------------------------------------------------------------
// Free-function API (standard parameterizations)
// ---------------------------------------------------------------------------

/// Standard normal PDF.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}
/// Standard normal CDF.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}
/// Standard normal survival function `1 − Φ(x)`.
pub fn norm_sf(x: f64) -> f64 {
    0.5 * erfc(x / SQRT_2)
}
/// Standard normal quantile (inverse CDF).
pub fn norm_ppf(p: f64) -> f64 {
    SQRT_2 * erfinv(2.0 * p - 1.0)
}
/// Standard normal inverse survival function.
pub fn norm_isf(p: f64) -> f64 {
    norm_ppf(1.0 - p)
}

/// Student-t PDF with `df` degrees of freedom.
pub fn t_pdf(x: f64, df: f64) -> f64 {
    let c = (lgamma(0.5 * (df + 1.0)) - lgamma(0.5 * df)).exp() / (df * PI).sqrt();
    c * (1.0 + x * x / df).powf(-0.5 * (df + 1.0))
}
/// Student-t CDF.
pub fn t_cdf(x: f64, df: f64) -> f64 {
    let xt = df / (df + x * x);
    let ib = betainc(0.5 * df, 0.5, xt);
    if x >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}
/// Student-t survival function.
pub fn t_sf(x: f64, df: f64) -> f64 {
    t_cdf(-x, df)
}
/// Student-t quantile (inverse CDF).
pub fn t_ppf(p: f64, df: f64) -> f64 {
    if p == 0.5 {
        return 0.0;
    }
    let (pp, sign) = if p < 0.5 {
        (2.0 * p, -1.0)
    } else {
        (2.0 * (1.0 - p), 1.0)
    };
    let x = betaincinv(0.5 * df, 0.5, pp);
    let t = (df * (1.0 - x) / x).sqrt();
    sign * t
}
/// Student-t inverse survival function.
pub fn t_isf(p: f64, df: f64) -> f64 {
    t_ppf(1.0 - p, df)
}

/// F-distribution PDF with `dfn`/`dfd` degrees of freedom.
pub fn f_pdf(x: f64, dfn: f64, dfd: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let lbeta = lgamma(0.5 * dfn) + lgamma(0.5 * dfd) - lgamma(0.5 * (dfn + dfd));
    let lnum = 0.5 * dfn * (dfn / dfd).ln() + (0.5 * dfn - 1.0) * x.ln();
    let lden = 0.5 * (dfn + dfd) * (1.0 + dfn * x / dfd).ln();
    (lnum - lden - lbeta).exp()
}
/// F-distribution CDF.
pub fn f_cdf(x: f64, dfn: f64, dfd: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let y = dfn * x / (dfn * x + dfd);
    betainc(0.5 * dfn, 0.5 * dfd, y)
}
/// F-distribution survival function.
pub fn f_sf(x: f64, dfn: f64, dfd: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let y = dfd / (dfd + dfn * x);
    betainc(0.5 * dfd, 0.5 * dfn, y)
}
/// F-distribution quantile (inverse CDF).
pub fn f_ppf(p: f64, dfn: f64, dfd: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let y = betaincinv(0.5 * dfn, 0.5 * dfd, p);
    dfd * y / (dfn * (1.0 - y))
}
/// F-distribution inverse survival function.
pub fn f_isf(p: f64, dfn: f64, dfd: f64) -> f64 {
    f_ppf(1.0 - p, dfn, dfd)
}

/// Chi-squared PDF with `df` degrees of freedom.
pub fn chi2_pdf(x: f64, df: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    let k = 0.5 * df;
    ((k - 1.0) * x.ln() - 0.5 * x - k * 2.0_f64.ln() - lgamma(k)).exp()
}
/// Chi-squared CDF.
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    gammainc(0.5 * df, 0.5 * x)
}
/// Chi-squared survival function.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gammaincc(0.5 * df, 0.5 * x)
}
/// Chi-squared quantile (inverse CDF).
pub fn chi2_ppf(p: f64, df: f64) -> f64 {
    2.0 * gammaincinv(0.5 * df, p)
}
/// Chi-squared inverse survival function.
pub fn chi2_isf(p: f64, df: f64) -> f64 {
    chi2_ppf(1.0 - p, df)
}

// ---------------------------------------------------------------------------
// Struct API
// ---------------------------------------------------------------------------

/// A normal (Gaussian) distribution with location `loc` and scale `scale`.
#[derive(Clone, Copy, Debug)]
pub struct Normal {
    pub loc: f64,
    pub scale: f64,
}

impl Default for Normal {
    fn default() -> Self {
        Normal {
            loc: 0.0,
            scale: 1.0,
        }
    }
}

impl Normal {
    pub fn new(loc: f64, scale: f64) -> Self {
        Normal { loc, scale }
    }
    pub fn pdf(&self, x: f64) -> f64 {
        norm_pdf((x - self.loc) / self.scale) / self.scale
    }
    pub fn cdf(&self, x: f64) -> f64 {
        norm_cdf((x - self.loc) / self.scale)
    }
    pub fn sf(&self, x: f64) -> f64 {
        norm_sf((x - self.loc) / self.scale)
    }
    pub fn ppf(&self, p: f64) -> f64 {
        self.loc + self.scale * norm_ppf(p)
    }
    pub fn isf(&self, p: f64) -> f64 {
        self.loc + self.scale * norm_isf(p)
    }
}

/// A Student-t distribution with `df` degrees of freedom (standard location/scale).
#[derive(Clone, Copy, Debug)]
pub struct StudentT {
    pub df: f64,
}

impl StudentT {
    pub fn new(df: f64) -> Self {
        StudentT { df }
    }
    pub fn pdf(&self, x: f64) -> f64 {
        t_pdf(x, self.df)
    }
    pub fn cdf(&self, x: f64) -> f64 {
        t_cdf(x, self.df)
    }
    pub fn sf(&self, x: f64) -> f64 {
        t_sf(x, self.df)
    }
    pub fn ppf(&self, p: f64) -> f64 {
        t_ppf(p, self.df)
    }
    pub fn isf(&self, p: f64) -> f64 {
        t_isf(p, self.df)
    }
}

/// An F distribution with numerator/denominator degrees of freedom.
#[derive(Clone, Copy, Debug)]
pub struct FDist {
    pub dfn: f64,
    pub dfd: f64,
}

impl FDist {
    pub fn new(dfn: f64, dfd: f64) -> Self {
        FDist { dfn, dfd }
    }
    pub fn pdf(&self, x: f64) -> f64 {
        f_pdf(x, self.dfn, self.dfd)
    }
    pub fn cdf(&self, x: f64) -> f64 {
        f_cdf(x, self.dfn, self.dfd)
    }
    pub fn sf(&self, x: f64) -> f64 {
        f_sf(x, self.dfn, self.dfd)
    }
    pub fn ppf(&self, p: f64) -> f64 {
        f_ppf(p, self.dfn, self.dfd)
    }
    pub fn isf(&self, p: f64) -> f64 {
        f_isf(p, self.dfn, self.dfd)
    }
}

/// A chi-squared distribution with `df` degrees of freedom.
#[derive(Clone, Copy, Debug)]
pub struct ChiSquared {
    pub df: f64,
}

impl ChiSquared {
    pub fn new(df: f64) -> Self {
        ChiSquared { df }
    }
    pub fn pdf(&self, x: f64) -> f64 {
        chi2_pdf(x, self.df)
    }
    pub fn cdf(&self, x: f64) -> f64 {
        chi2_cdf(x, self.df)
    }
    pub fn sf(&self, x: f64) -> f64 {
        chi2_sf(x, self.df)
    }
    pub fn ppf(&self, p: f64) -> f64 {
        chi2_ppf(p, self.df)
    }
    pub fn isf(&self, p: f64) -> f64 {
        chi2_isf(p, self.df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn normal_symmetry_and_roundtrip() {
        assert_abs_diff_eq!(norm_cdf(0.0), 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(norm_cdf(1.0) + norm_cdf(-1.0), 1.0, epsilon = 1e-14);
        // 97.5th percentile ≈ 1.959963985
        assert_abs_diff_eq!(norm_ppf(0.975), 1.959_963_984_540_054, epsilon = 1e-9);
        for &p in &[0.01, 0.25, 0.5, 0.84, 0.99] {
            assert_abs_diff_eq!(norm_cdf(norm_ppf(p)), p, epsilon = 1e-10);
        }
    }

    #[test]
    fn student_t_roundtrip_and_tail() {
        for &df in &[1.0, 5.0, 30.0] {
            for &p in &[0.05, 0.5, 0.975] {
                assert_abs_diff_eq!(t_cdf(t_ppf(p, df), df), p, epsilon = 1e-9);
            }
            assert_abs_diff_eq!(t_cdf(0.0, df), 0.5, epsilon = 1e-12);
            assert_abs_diff_eq!(t_sf(1.0, df) + t_cdf(1.0, df), 1.0, epsilon = 1e-12);
        }
        // t(10) 0.975 quantile ≈ 2.228138852
        assert_abs_diff_eq!(t_ppf(0.975, 10.0), 2.228_138_851_986_273, epsilon = 1e-8);
    }

    #[test]
    fn f_and_chi2_roundtrip() {
        for &p in &[0.1, 0.5, 0.9] {
            let x = f_ppf(p, 3.0, 20.0);
            assert_abs_diff_eq!(f_cdf(x, 3.0, 20.0), p, epsilon = 1e-9);
            assert_abs_diff_eq!(
                f_sf(x, 3.0, 20.0) + f_cdf(x, 3.0, 20.0),
                1.0,
                epsilon = 1e-12
            );
        }
        for &p in &[0.1, 0.5, 0.95] {
            let x = chi2_ppf(p, 7.0);
            assert_abs_diff_eq!(chi2_cdf(x, 7.0), p, epsilon = 1e-9);
        }
    }
}
