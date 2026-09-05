//! Special functions: log-gamma, digamma, regularized incomplete gamma and beta
//! (and their inverses), and the error function.
//!
//! These are the analytic backbone of the statistical distributions. Algorithms
//! follow the standard, well-tested forms (Lanczos for `lgamma`; series /
//! continued-fraction for the incomplete integrals) and are validated against an
//! authoritative reference to ~1e-12.

use std::f64::consts::PI;

const TINY: f64 = 1.0e-300;

/// Lanczos coefficients (`g = 7`, 9 terms). The literals carry a couple of extra
/// guard digits that round to the same `f64`; the precision lint is benign here.
const LANCZOS_G: f64 = 7.0;
#[allow(clippy::excessive_precision)]
const LANCZOS_COEF: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural logarithm of the gamma function, `ln Γ(x)`, for real `x`.
pub fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1−x) = π / sin(πx).
        (PI / (PI * x).sin()).ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS_COEF[0];
        let t = x + LANCZOS_G + 0.5;
        for (i, &c) in LANCZOS_COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// The gamma function `Γ(x)`.
pub fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        lgamma(x).exp()
    }
}

/// Natural logarithm of the beta function, `ln B(a, b)`.
pub fn lbeta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// The digamma function `ψ(x) = d/dx ln Γ(x)`.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Reflection for negative arguments.
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x < 0.0 {
        result -= PI / (PI * x).tan();
        x = 1.0 - x;
    }
    // Recurrence up to x >= 6.
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic series.
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln()
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0))))
}

// ---------------------------------------------------------------------------
// Incomplete gamma
// ---------------------------------------------------------------------------

/// Series expansion for the regularized lower incomplete gamma `P(a, x)`,
/// suitable for `x < a + 1`.
fn gammainc_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..1000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x + a * x.ln() - lgamma(a)).exp()
}

/// Continued fraction (Lentz) for the regularized upper incomplete gamma
/// `Q(a, x)`, suitable for `x >= a + 1`.
fn gammaincc_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x + a * x.ln() - lgamma(a)).exp() * h
}

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`.
pub fn gammainc(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gammainc_series(a, x)
    } else {
        1.0 - gammaincc_cf(a, x)
    }
}

/// Regularized upper incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 − P(a, x)`.
pub fn gammaincc(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gammainc_series(a, x)
    } else {
        gammaincc_cf(a, x)
    }
}

/// Inverse of the regularized lower incomplete gamma: returns `x` with
/// `P(a, x) = p`. Safeguarded Newton iteration with bisection fallback.
pub fn gammaincinv(a: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Bracket [lo, hi].
    let mut lo = 0.0_f64;
    let mut hi = a.max(1.0);
    while gammainc(a, hi) < p {
        hi *= 2.0;
        if hi > 1e300 {
            break;
        }
    }
    let lga = lgamma(a);
    let mut x = 0.5 * (lo + hi);
    for _ in 0..200 {
        let err = gammainc(a, x) - p;
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let deriv = ((a - 1.0) * x.ln() - x - lga).exp();
        let mut xnew = if deriv > 0.0 { x - err / deriv } else { x };
        if !(xnew > lo && xnew < hi) {
            xnew = 0.5 * (lo + hi);
        }
        if (xnew - x).abs() <= 1e-14 * x.abs().max(1e-300) {
            x = xnew;
            break;
        }
        x = xnew;
    }
    x
}

// ---------------------------------------------------------------------------
// Incomplete beta
// ---------------------------------------------------------------------------

/// Continued fraction for the incomplete beta integral (Lentz).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)`.
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta(a, b)).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Inverse of the regularized incomplete beta: returns `x` with `I_x(a, b) = p`.
/// Safeguarded Newton iteration with bisection fallback.
pub fn betaincinv(a: f64, b: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let lbab = lbeta(a, b);
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut x = a / (a + b);
    for _ in 0..200 {
        let err = betainc(a, b, x) - p;
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        // d/dx I_x(a,b) = x^(a−1) (1−x)^(b−1) / B(a,b)
        let deriv = ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - lbab).exp();
        let mut xnew = if deriv.is_finite() && deriv > 0.0 {
            x - err / deriv
        } else {
            x
        };
        if !(xnew > lo && xnew < hi) {
            xnew = 0.5 * (lo + hi);
        }
        if (xnew - x).abs() <= 1e-15 * x.max(1e-300) {
            x = xnew;
            break;
        }
        x = xnew;
    }
    x
}

// ---------------------------------------------------------------------------
// Error function (via the incomplete gamma) and its inverse
// ---------------------------------------------------------------------------

/// The error function `erf(x)`.
pub fn erf(x: f64) -> f64 {
    let z = x * x;
    // Guard the overflow regime: for |x| ≳ 1.34e154, `x*x` is `+inf` and the
    // incomplete gamma would return NaN. `erf` has long since saturated to ±1 there
    // (indeed for any |x| ≥ 6, `erf(x)` rounds to ±1 in f64), so return the limit.
    if !z.is_finite() {
        return x.signum();
    }
    if x >= 0.0 {
        gammainc(0.5, z)
    } else {
        -gammainc(0.5, z)
    }
}

/// The complementary error function `erfc(x) = 1 − erf(x)`.
pub fn erfc(x: f64) -> f64 {
    let z = x * x;
    // Guard the overflow regime (see [`erf`]): `erfc(+huge) = 0`, `erfc(−huge) = 2`.
    if !z.is_finite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    if x >= 0.0 {
        gammaincc(0.5, z)
    } else {
        1.0 + gammainc(0.5, z)
    }
}

/// The inverse error function `erfinv(y)` for `y ∈ (−1, 1)`.
pub fn erfinv(y: f64) -> f64 {
    if y <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if y >= 1.0 {
        return f64::INFINITY;
    }
    if y == 0.0 {
        return 0.0;
    }
    // Winitzki initial approximation.
    let w = 0.147;
    let ln = (1.0 - y * y).ln();
    let part = 2.0 / (PI * w) + ln / 2.0;
    let mut x = ((part * part - ln / w).sqrt() - part).sqrt().copysign(y);
    // Newton refinement: f(x) = erf(x) − y, f'(x) = 2/√π e^{−x²}.
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..6 {
        let err = erf(x) - y;
        let deriv = two_over_sqrt_pi * (-x * x).exp();
        if deriv == 0.0 {
            break;
        }
        let dx = err / deriv;
        x -= dx;
        if dx.abs() <= 1e-15 * x.abs().max(1e-300) {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn lgamma_known_values() {
        // Γ(5) = 24 → lgamma(5) = ln 24
        assert_abs_diff_eq!(lgamma(5.0), 24.0_f64.ln(), epsilon = 1e-12);
        // Γ(1/2) = √π
        assert_abs_diff_eq!(lgamma(0.5), PI.sqrt().ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(gamma(6.0), 120.0, epsilon = 1e-9);
    }

    #[test]
    fn erf_erfc_saturate_at_extreme_args() {
        // The overflow regime must return the analytic limit, not NaN.
        assert_eq!(erf(1e300), 1.0);
        assert_eq!(erf(-1e300), -1.0);
        assert_eq!(erfc(1e300), 0.0);
        assert_eq!(erfc(-1e300), 2.0);
        assert!(erf(f64::MAX).is_finite() && erfc(f64::MAX).is_finite());
        // Normal values are unchanged (parity).
        assert_abs_diff_eq!(erf(0.5), 0.5204998778130465, epsilon = 1e-12);
        assert_abs_diff_eq!(erfc(1.5), 0.033894853524689274, epsilon = 1e-12);
    }

    #[test]
    fn erf_inverse_roundtrip() {
        for &v in &[-0.9, -0.3, 0.1, 0.5, 0.99] {
            assert_abs_diff_eq!(erf(erfinv(v)), v, epsilon = 1e-12);
        }
        assert_abs_diff_eq!(erf(0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn incomplete_gamma_complement() {
        for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (0.5, 1.0), (10.0, 8.0)] {
            assert_abs_diff_eq!(gammainc(a, x) + gammaincc(a, x), 1.0, epsilon = 1e-13);
        }
        // Inverse round trip.
        for &(a, p) in &[(2.0, 0.3), (5.0, 0.9), (0.5, 0.5)] {
            let x = gammaincinv(a, p);
            assert_abs_diff_eq!(gammainc(a, x), p, epsilon = 1e-10);
        }
    }

    #[test]
    fn incomplete_beta_complement_and_inverse() {
        for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 0.5, 0.7), (5.0, 2.0, 0.2)] {
            assert_abs_diff_eq!(
                betainc(a, b, x) + betainc(b, a, 1.0 - x),
                1.0,
                epsilon = 1e-12
            );
        }
        for &(a, b, p) in &[(2.0, 3.0, 0.25), (0.5, 0.5, 0.6), (4.0, 7.0, 0.9)] {
            let x = betaincinv(a, b, p);
            assert_abs_diff_eq!(betainc(a, b, x), p, epsilon = 1e-10);
        }
    }
}
