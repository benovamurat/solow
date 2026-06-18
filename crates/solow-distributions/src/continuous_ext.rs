//! A library of continuous distributions matching the `scipy.stats`
//! parameterization. Each is a small `Copy` struct exposing `pdf`, `logpdf`,
//! `cdf`, `sf`, `ppf`, `mean`, and `var`.
//!
//! All CDFs reduce to the regularized incomplete gamma / beta integrals (or
//! elementary closed forms) from [`crate::special`], and all quantiles invert
//! those same integrals, so the values agree with the reference to ~1e-10
//! (PDF/CDF/SF) and ~1e-8 (PPF).

use crate::special::{betainc, betaincinv, gammainc, gammaincc, gammaincinv, lbeta, lgamma};
use std::f64::consts::PI;

const LN_2: f64 = std::f64::consts::LN_2;

// =========================================================================
// Gamma{a, scale}
// =========================================================================

/// Gamma distribution, `scipy.stats.gamma(a, scale=scale)`.
///
/// Support `x > 0`; `pdf(x) = x^{a-1} e^{-x/scale} / (Γ(a) scale^a)`.
#[derive(Clone, Copy, Debug)]
pub struct Gamma {
    /// Shape parameter `a > 0`.
    pub a: f64,
    /// Scale parameter `scale > 0`.
    pub scale: f64,
}

impl Gamma {
    /// Create a gamma distribution with shape `a` and scale `scale`.
    pub fn new(a: f64, scale: f64) -> Self {
        Gamma { a, scale }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            return if self.a < 1.0 {
                f64::INFINITY
            } else if self.a == 1.0 {
                1.0 / self.scale
            } else {
                0.0
            };
        }
        self.logpdf(x).exp()
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = x / self.scale;
        (self.a - 1.0) * z.ln() - z - lgamma(self.a) - self.scale.ln()
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        gammainc(self.a, x / self.scale)
    }
    /// Survival function `P(X > x)`.
    pub fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        gammaincc(self.a, x / self.scale)
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        self.scale * gammaincinv(self.a, p)
    }
    /// Mean `a · scale`.
    pub fn mean(&self) -> f64 {
        self.a * self.scale
    }
    /// Variance `a · scale²`.
    pub fn var(&self) -> f64 {
        self.a * self.scale * self.scale
    }
}

// =========================================================================
// Beta{a, b}
// =========================================================================

/// Beta distribution, `scipy.stats.beta(a, b)` on `[0, 1]`.
#[derive(Clone, Copy, Debug)]
pub struct Beta {
    /// First shape `a > 0`.
    pub a: f64,
    /// Second shape `b > 0`.
    pub b: f64,
}

impl Beta {
    /// Create a beta distribution with shapes `a`, `b`.
    pub fn new(a: f64, b: f64) -> Self {
        Beta { a, b }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        self.logpdf(x).exp()
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }
        (self.a - 1.0) * x.ln() + (self.b - 1.0) * (1.0 - x).ln() - lbeta(self.a, self.b)
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        betainc(self.a, self.b, x)
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        betaincinv(self.a, self.b, p)
    }
    /// Mean `a / (a + b)`.
    pub fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
    /// Variance.
    pub fn var(&self) -> f64 {
        let s = self.a + self.b;
        self.a * self.b / (s * s * (s + 1.0))
    }
}

// =========================================================================
// Exponential{scale}
// =========================================================================

/// Exponential distribution, `scipy.stats.expon(scale=scale)`.
#[derive(Clone, Copy, Debug)]
pub struct Exponential {
    /// Scale `scale > 0` (= 1/rate).
    pub scale: f64,
}

impl Exponential {
    /// Create an exponential distribution with scale `scale`.
    pub fn new(scale: f64) -> Self {
        Exponential { scale }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        (-x / self.scale).exp() / self.scale
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        -x / self.scale - self.scale.ln()
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        -(-x / self.scale).exp_m1()
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        (-x / self.scale).exp()
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        -self.scale * (1.0 - p).ln()
    }
    /// Mean `scale`.
    pub fn mean(&self) -> f64 {
        self.scale
    }
    /// Variance `scale²`.
    pub fn var(&self) -> f64 {
        self.scale * self.scale
    }
}

// =========================================================================
// LogNormal{s, scale}
// =========================================================================

/// Log-normal distribution, `scipy.stats.lognorm(s, scale=scale)`.
///
/// `log X ~ Normal(ln(scale), s²)`.
#[derive(Clone, Copy, Debug)]
pub struct LogNormal {
    /// Shape `s > 0` (= sigma of the underlying normal).
    pub s: f64,
    /// Scale `scale > 0` (= exp(mu) of the underlying normal).
    pub scale: f64,
}

impl LogNormal {
    /// Create a log-normal distribution.
    pub fn new(s: f64, scale: f64) -> Self {
        LogNormal { s, scale }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.logpdf(x).exp()
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = (x / self.scale).ln() / self.s;
        -0.5 * z * z - (self.s * x * (2.0 * PI).sqrt()).ln()
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x / self.scale).ln() / self.s;
        std_norm_cdf(z)
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let z = (x / self.scale).ln() / self.s;
        std_norm_cdf(-z)
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        self.scale * (self.s * std_norm_ppf(p)).exp()
    }
    /// Mean `scale · exp(s²/2)`.
    pub fn mean(&self) -> f64 {
        self.scale * (0.5 * self.s * self.s).exp()
    }
    /// Variance.
    pub fn var(&self) -> f64 {
        let s2 = self.s * self.s;
        (s2.exp() - 1.0) * self.scale * self.scale * s2.exp()
    }
}

// =========================================================================
// Uniform{loc, scale}
// =========================================================================

/// Uniform distribution on `[loc, loc + scale]`, `scipy.stats.uniform(loc, scale)`.
#[derive(Clone, Copy, Debug)]
pub struct Uniform {
    /// Lower bound `loc`.
    pub loc: f64,
    /// Width `scale > 0`.
    pub scale: f64,
}

impl Uniform {
    /// Create a uniform distribution on `[loc, loc + scale]`.
    pub fn new(loc: f64, scale: f64) -> Self {
        Uniform { loc, scale }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        if x < self.loc || x > self.loc + self.scale {
            0.0
        } else {
            1.0 / self.scale
        }
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        if x < self.loc || x > self.loc + self.scale {
            f64::NEG_INFINITY
        } else {
            -self.scale.ln()
        }
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        ((x - self.loc) / self.scale).clamp(0.0, 1.0)
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        self.loc + p * self.scale
    }
    /// Mean.
    pub fn mean(&self) -> f64 {
        self.loc + 0.5 * self.scale
    }
    /// Variance `scale²/12`.
    pub fn var(&self) -> f64 {
        self.scale * self.scale / 12.0
    }
}

// =========================================================================
// WeibullMin{c}
// =========================================================================

/// Weibull (minimum) distribution, `scipy.stats.weibull_min(c)`.
///
/// Standard (loc = 0, scale = 1); `pdf(x) = c x^{c-1} e^{-x^c}` for `x > 0`.
#[derive(Clone, Copy, Debug)]
pub struct WeibullMin {
    /// Shape `c > 0`.
    pub c: f64,
}

impl WeibullMin {
    /// Create a Weibull-min distribution with shape `c`.
    pub fn new(c: f64) -> Self {
        WeibullMin { c }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            return if self.c < 1.0 {
                f64::INFINITY
            } else if self.c == 1.0 {
                1.0
            } else {
                0.0
            };
        }
        self.logpdf(x).exp()
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        self.c.ln() + (self.c - 1.0) * x.ln() - x.powf(self.c)
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        -(-x.powf(self.c)).exp_m1()
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        (-x.powf(self.c)).exp()
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        (-(1.0 - p).ln()).powf(1.0 / self.c)
    }
    /// Mean `Γ(1 + 1/c)`.
    pub fn mean(&self) -> f64 {
        lgamma(1.0 + 1.0 / self.c).exp()
    }
    /// Variance `Γ(1 + 2/c) − Γ(1 + 1/c)²`.
    pub fn var(&self) -> f64 {
        let g1 = lgamma(1.0 + 1.0 / self.c).exp();
        let g2 = lgamma(1.0 + 2.0 / self.c).exp();
        g2 - g1 * g1
    }
}

// =========================================================================
// Laplace{loc, scale}
// =========================================================================

/// Laplace (double-exponential) distribution, `scipy.stats.laplace(loc, scale)`.
#[derive(Clone, Copy, Debug)]
pub struct Laplace {
    /// Location `loc`.
    pub loc: f64,
    /// Scale `scale > 0`.
    pub scale: f64,
}

impl Laplace {
    /// Create a Laplace distribution.
    pub fn new(loc: f64, scale: f64) -> Self {
        Laplace { loc, scale }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc).abs() / self.scale;
        0.5 * (-z).exp() / self.scale
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        let z = (x - self.loc).abs() / self.scale;
        -z - LN_2 - self.scale.ln()
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        }
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            1.0 - 0.5 * z.exp()
        } else {
            0.5 * (-z).exp()
        }
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        let z = if p <= 0.5 {
            (2.0 * p).ln()
        } else {
            -(2.0 * (1.0 - p)).ln()
        };
        self.loc + self.scale * z
    }
    /// Mean `loc`.
    pub fn mean(&self) -> f64 {
        self.loc
    }
    /// Variance `2 · scale²`.
    pub fn var(&self) -> f64 {
        2.0 * self.scale * self.scale
    }
}

// =========================================================================
// Logistic{loc, scale}
// =========================================================================

/// Logistic distribution, `scipy.stats.logistic(loc, scale)`.
#[derive(Clone, Copy, Debug)]
pub struct Logistic {
    /// Location `loc`.
    pub loc: f64,
    /// Scale `scale > 0`.
    pub scale: f64,
}

impl Logistic {
    /// Create a logistic distribution.
    pub fn new(loc: f64, scale: f64) -> Self {
        Logistic { loc, scale }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        self.logpdf(x).exp()
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        // log f = -z - 2 log(1 + e^{-z}) - log scale, written stably.
        // softplus(-z) = log(1 + e^{-z}).
        let softplus_neg = if z <= 0.0 {
            -z + z.exp().ln_1p()
        } else {
            (-z).exp().ln_1p()
        };
        -z - 2.0 * softplus_neg - self.scale.ln()
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        // 1 / (1 + e^{-z}) computed stably.
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let e = z.exp();
            e / (1.0 + e)
        }
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z >= 0.0 {
            let e = (-z).exp();
            e / (1.0 + e)
        } else {
            1.0 / (1.0 + z.exp())
        }
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        self.loc + self.scale * (p / (1.0 - p)).ln()
    }
    /// Mean `loc`.
    pub fn mean(&self) -> f64 {
        self.loc
    }
    /// Variance `scale² π²/3`.
    pub fn var(&self) -> f64 {
        self.scale * self.scale * PI * PI / 3.0
    }
}

// =========================================================================
// Cauchy{loc, scale}
// =========================================================================

/// Cauchy distribution, `scipy.stats.cauchy(loc, scale)`. Mean and variance
/// are undefined (returned as `NaN`).
#[derive(Clone, Copy, Debug)]
pub struct Cauchy {
    /// Location `loc`.
    pub loc: f64,
    /// Scale `scale > 0`.
    pub scale: f64,
}

impl Cauchy {
    /// Create a Cauchy distribution.
    pub fn new(loc: f64, scale: f64) -> Self {
        Cauchy { loc, scale }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        1.0 / (PI * self.scale * (1.0 + z * z))
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        -(PI * self.scale).ln() - (1.0 + z * z).ln()
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        0.5 + z.atan() / PI
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        0.5 - z.atan() / PI
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        self.loc + self.scale * (PI * (p - 0.5)).tan()
    }
    /// Mean (undefined → `NaN`).
    pub fn mean(&self) -> f64 {
        f64::NAN
    }
    /// Variance (undefined → `NaN`).
    pub fn var(&self) -> f64 {
        f64::NAN
    }
}

// =========================================================================
// Pareto{b}
// =========================================================================

/// Pareto distribution, `scipy.stats.pareto(b)` (standard, support `x ≥ 1`).
#[derive(Clone, Copy, Debug)]
pub struct Pareto {
    /// Shape `b > 0`.
    pub b: f64,
}

impl Pareto {
    /// Create a Pareto distribution with shape `b`.
    pub fn new(b: f64) -> Self {
        Pareto { b }
    }
    /// Probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 1.0 {
            return 0.0;
        }
        self.b / x.powf(self.b + 1.0)
    }
    /// Log probability density at `x`.
    pub fn logpdf(&self, x: f64) -> f64 {
        if x < 1.0 {
            return f64::NEG_INFINITY;
        }
        self.b.ln() - (self.b + 1.0) * x.ln()
    }
    /// Cumulative distribution `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        if x < 1.0 {
            return 0.0;
        }
        1.0 - x.powf(-self.b)
    }
    /// Survival function.
    pub fn sf(&self, x: f64) -> f64 {
        if x < 1.0 {
            return 1.0;
        }
        x.powf(-self.b)
    }
    /// Quantile (inverse CDF).
    pub fn ppf(&self, p: f64) -> f64 {
        (1.0 - p).powf(-1.0 / self.b)
    }
    /// Mean `b/(b−1)` (for `b > 1`, else `∞`).
    pub fn mean(&self) -> f64 {
        if self.b <= 1.0 {
            f64::INFINITY
        } else {
            self.b / (self.b - 1.0)
        }
    }
    /// Variance (for `b > 2`, else `∞`).
    pub fn var(&self) -> f64 {
        if self.b <= 2.0 {
            f64::INFINITY
        } else {
            let bm1 = self.b - 1.0;
            self.b / (bm1 * bm1 * (self.b - 2.0))
        }
    }
}

// ---------------------------------------------------------------------------
// Local standard-normal helpers (kept private to avoid leaking new free fns).
// ---------------------------------------------------------------------------

fn std_norm_cdf(x: f64) -> f64 {
    crate::continuous::norm_cdf(x)
}
fn std_norm_ppf(p: f64) -> f64 {
    crate::continuous::norm_ppf(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn roundtrip<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(cdf: F, ppf: G, ps: &[f64]) {
        for &p in ps {
            let x = ppf(p);
            assert_abs_diff_eq!(cdf(x), p, epsilon = 1e-9);
        }
    }

    #[test]
    fn gamma_roundtrip_and_moments() {
        let g = Gamma::new(2.5, 3.0);
        roundtrip(|x| g.cdf(x), |p| g.ppf(p), &[0.05, 0.5, 0.95]);
        assert_abs_diff_eq!(g.cdf(2.0) + g.sf(2.0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(g.mean(), 7.5, epsilon = 1e-12);
        assert_abs_diff_eq!(g.var(), 22.5, epsilon = 1e-12);
    }

    #[test]
    fn beta_and_uniform() {
        let b = Beta::new(2.0, 3.0);
        roundtrip(|x| b.cdf(x), |p| b.ppf(p), &[0.1, 0.5, 0.9]);
        assert_abs_diff_eq!(b.mean(), 0.4, epsilon = 1e-12);
        let u = Uniform::new(-1.0, 4.0);
        assert_abs_diff_eq!(u.cdf(1.0), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(u.ppf(0.25), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn logpdf_consistency() {
        let ln = LogNormal::new(0.5, 2.0);
        let wb = WeibullMin::new(1.5);
        let lg = Logistic::new(0.0, 1.0);
        for &x in &[0.5, 1.5, 3.0] {
            assert_abs_diff_eq!(ln.pdf(x).ln(), ln.logpdf(x), epsilon = 1e-10);
            assert_abs_diff_eq!(wb.pdf(x).ln(), wb.logpdf(x), epsilon = 1e-10);
            assert_abs_diff_eq!(lg.pdf(x).ln(), lg.logpdf(x), epsilon = 1e-10);
        }
    }

    #[test]
    fn laplace_logistic_cauchy_pareto() {
        let l = Laplace::new(1.0, 2.0);
        roundtrip(|x| l.cdf(x), |p| l.ppf(p), &[0.1, 0.5, 0.9]);
        let lg = Logistic::new(-1.0, 0.7);
        roundtrip(|x| lg.cdf(x), |p| lg.ppf(p), &[0.05, 0.5, 0.95]);
        let c = Cauchy::new(0.0, 1.0);
        roundtrip(|x| c.cdf(x), |p| c.ppf(p), &[0.2, 0.5, 0.8]);
        let pa = Pareto::new(3.0);
        roundtrip(|x| pa.cdf(x), |p| pa.ppf(p), &[0.1, 0.5, 0.9]);
        assert_abs_diff_eq!(pa.mean(), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn exponential() {
        let e = Exponential::new(2.0);
        assert_abs_diff_eq!(e.mean(), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.var(), 4.0, epsilon = 1e-12);
        roundtrip(|x| e.cdf(x), |p| e.ppf(p), &[0.1, 0.5, 0.99]);
    }
}
