//! A library of discrete distributions matching the `scipy.stats`
//! parameterization. Each is a small `Copy` struct exposing `pmf`, `logpmf`,
//! `cdf`, `sf`, `mean`, and `var`.
//!
//! All CDFs reduce to the regularized incomplete gamma / beta integrals (or
//! elementary closed forms) from [`crate::special`], matching the reference to
//! ~1e-10.

use crate::special::{betainc, gammaincc, lgamma};

/// Log binomial coefficient `ln C(n, k)` via log-gamma.
fn ln_binom(n: f64, k: f64) -> f64 {
    lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)
}

// =========================================================================
// Poisson{mu}
// =========================================================================

/// Poisson distribution, `scipy.stats.poisson(mu)`.
#[derive(Clone, Copy, Debug)]
pub struct Poisson {
    /// Rate `mu ≥ 0`.
    pub mu: f64,
}

impl Poisson {
    /// Create a Poisson distribution with rate `mu`.
    pub fn new(mu: f64) -> Self {
        Poisson { mu }
    }
    /// Probability mass at integer `k`.
    pub fn pmf(&self, k: u64) -> f64 {
        self.logpmf(k).exp()
    }
    /// Log probability mass at integer `k`.
    pub fn logpmf(&self, k: u64) -> f64 {
        let kf = k as f64;
        kf * self.mu.ln() - self.mu - lgamma(kf + 1.0)
    }
    /// Cumulative distribution `P(X ≤ k)` for real `k` (floored, as scipy).
    pub fn cdf(&self, k: f64) -> f64 {
        if k < 0.0 {
            return 0.0;
        }
        let m = k.floor();
        // P(X ≤ m) = Q(m+1, mu) = gammaincc(m+1, mu)
        gammaincc(m + 1.0, self.mu)
    }
    /// Survival function `P(X > k)`.
    pub fn sf(&self, k: f64) -> f64 {
        1.0 - self.cdf(k)
    }
    /// Mean `mu`.
    pub fn mean(&self) -> f64 {
        self.mu
    }
    /// Variance `mu`.
    pub fn var(&self) -> f64 {
        self.mu
    }
}

// =========================================================================
// Binomial{n, p}
// =========================================================================

/// Binomial distribution, `scipy.stats.binom(n, p)`.
#[derive(Clone, Copy, Debug)]
pub struct Binomial {
    /// Number of trials `n`.
    pub n: u64,
    /// Success probability `p ∈ [0, 1]`.
    pub p: f64,
}

impl Binomial {
    /// Create a binomial distribution.
    pub fn new(n: u64, p: f64) -> Self {
        Binomial { n, p }
    }
    /// Probability mass at integer `k`.
    pub fn pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return 0.0;
        }
        self.logpmf(k).exp()
    }
    /// Log probability mass at integer `k`.
    pub fn logpmf(&self, k: u64) -> f64 {
        if k > self.n {
            return f64::NEG_INFINITY;
        }
        let nf = self.n as f64;
        let kf = k as f64;
        // Handle the p∈{0,1} boundaries (0·log 0 = 0 convention).
        if self.p <= 0.0 {
            if k == 0 {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else if self.p >= 1.0 {
            if k == self.n {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            ln_binom(nf, kf) + kf * self.p.ln() + (nf - kf) * (1.0 - self.p).ln()
        }
    }
    /// Cumulative distribution `P(X ≤ k)` for real `k` (floored, as scipy).
    pub fn cdf(&self, k: f64) -> f64 {
        if k < 0.0 {
            return 0.0;
        }
        let m = k.floor();
        if m >= self.n as f64 {
            return 1.0;
        }
        // P(X ≤ m) = I_{1-p}(n-m, m+1) = betainc(n-m, m+1, 1-p)
        let nf = self.n as f64;
        betainc(nf - m, m + 1.0, 1.0 - self.p)
    }
    /// Survival function `P(X > k)`.
    pub fn sf(&self, k: f64) -> f64 {
        1.0 - self.cdf(k)
    }
    /// Mean `n p`.
    pub fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    /// Variance `n p (1 − p)`.
    pub fn var(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

// =========================================================================
// Geometric{p}
// =========================================================================

/// Geometric distribution, `scipy.stats.geom(p)`, supported on `k = 1, 2, …`.
#[derive(Clone, Copy, Debug)]
pub struct Geometric {
    /// Success probability `p ∈ (0, 1]`.
    pub p: f64,
}

impl Geometric {
    /// Create a geometric distribution.
    pub fn new(p: f64) -> Self {
        Geometric { p }
    }
    /// Probability mass at integer `k ≥ 1`.
    pub fn pmf(&self, k: u64) -> f64 {
        if k < 1 {
            return 0.0;
        }
        self.logpmf(k).exp()
    }
    /// Log probability mass at integer `k ≥ 1`.
    pub fn logpmf(&self, k: u64) -> f64 {
        if k < 1 {
            return f64::NEG_INFINITY;
        }
        (k as f64 - 1.0) * (1.0 - self.p).ln() + self.p.ln()
    }
    /// Cumulative distribution `P(X ≤ k)` for real `k` (floored, as scipy).
    pub fn cdf(&self, k: f64) -> f64 {
        if k < 1.0 {
            return 0.0;
        }
        let m = k.floor();
        // 1 − (1−p)^m
        -((m) * (1.0 - self.p).ln()).exp_m1()
    }
    /// Survival function `P(X > k)`.
    pub fn sf(&self, k: f64) -> f64 {
        if k < 1.0 {
            return 1.0;
        }
        let m = k.floor();
        (m * (1.0 - self.p).ln()).exp()
    }
    /// Mean `1/p`.
    pub fn mean(&self) -> f64 {
        1.0 / self.p
    }
    /// Variance `(1 − p)/p²`.
    pub fn var(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
}

// =========================================================================
// NegativeBinomial{n, p}
// =========================================================================

/// Negative-binomial distribution, `scipy.stats.nbinom(n, p)`, supported on
/// `k = 0, 1, …` (number of failures before the `n`-th success).
#[derive(Clone, Copy, Debug)]
pub struct NegativeBinomial {
    /// Number of successes `n > 0` (may be non-integer in scipy).
    pub n: f64,
    /// Success probability `p ∈ (0, 1]`.
    pub p: f64,
}

impl NegativeBinomial {
    /// Create a negative-binomial distribution.
    pub fn new(n: f64, p: f64) -> Self {
        NegativeBinomial { n, p }
    }
    /// Probability mass at integer `k ≥ 0`.
    pub fn pmf(&self, k: u64) -> f64 {
        self.logpmf(k).exp()
    }
    /// Log probability mass at integer `k ≥ 0`.
    pub fn logpmf(&self, k: u64) -> f64 {
        let kf = k as f64;
        // C(k+n-1, k) p^n (1-p)^k
        ln_binom(kf + self.n - 1.0, kf) + self.n * self.p.ln() + kf * (1.0 - self.p).ln()
    }
    /// Cumulative distribution `P(X ≤ k)` for real `k` (floored, as scipy).
    pub fn cdf(&self, k: f64) -> f64 {
        if k < 0.0 {
            return 0.0;
        }
        let m = k.floor();
        // P(X ≤ m) = I_p(n, m+1) = betainc(n, m+1, p)
        betainc(self.n, m + 1.0, self.p)
    }
    /// Survival function `P(X > k)`.
    pub fn sf(&self, k: f64) -> f64 {
        1.0 - self.cdf(k)
    }
    /// Mean `n(1 − p)/p`.
    pub fn mean(&self) -> f64 {
        self.n * (1.0 - self.p) / self.p
    }
    /// Variance `n(1 − p)/p²`.
    pub fn var(&self) -> f64 {
        self.n * (1.0 - self.p) / (self.p * self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn poisson_basic() {
        let d = Poisson::new(3.5);
        // sum of pmf over a wide range ~ 1
        let s: f64 = (0..40).map(|k| d.pmf(k)).sum();
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.cdf(2.0) + d.sf(2.0), 1.0, epsilon = 1e-14);
        // cdf(k) - cdf(k-1) = pmf(k)
        assert_abs_diff_eq!(d.cdf(5.0) - d.cdf(4.0), d.pmf(5), epsilon = 1e-12);
        assert_abs_diff_eq!(d.mean(), 3.5);
        assert_abs_diff_eq!(d.var(), 3.5);
    }

    #[test]
    fn binomial_basic() {
        let d = Binomial::new(10, 0.3);
        let s: f64 = (0..=10).map(|k| d.pmf(k)).sum();
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.cdf(3.0) - d.cdf(2.0), d.pmf(3), epsilon = 1e-12);
        assert_abs_diff_eq!(d.mean(), 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.var(), 2.1, epsilon = 1e-12);
    }

    #[test]
    fn geometric_basic() {
        let d = Geometric::new(0.3);
        let s: f64 = (1..200).map(|k| d.pmf(k)).sum();
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d.cdf(3.0) - d.cdf(2.0), d.pmf(3), epsilon = 1e-12);
        assert_abs_diff_eq!(d.mean(), 1.0 / 0.3, epsilon = 1e-12);
    }

    #[test]
    fn negbinom_basic() {
        let d = NegativeBinomial::new(5.0, 0.4);
        let s: f64 = (0..300).map(|k| d.pmf(k)).sum();
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d.cdf(3.0) - d.cdf(2.0), d.pmf(3), epsilon = 1e-12);
        assert_abs_diff_eq!(d.mean(), 7.5, epsilon = 1e-12);
        assert_abs_diff_eq!(d.var(), 18.75, epsilon = 1e-12);
    }
}
