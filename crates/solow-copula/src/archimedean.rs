//! Bivariate Archimedean copulas in closed form.
//!
//! Each copula is parameterised by a single dependence parameter `theta`.
//! The `cdf`/`pdf` expressions and the Kendall's-tau mappings reproduce the
//! reference `distributions.copula.archimedean` implementation.

/// Clayton copula, `theta > 0` (the independence limit is `theta -> 0`).
///
/// CDF: `C(u, v) = (u^-theta + v^-theta - 1)^(-1/theta)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClaytonCopula {
    /// Dependence parameter.
    pub theta: f64,
}

impl ClaytonCopula {
    /// Construct a Clayton copula with the given dependence parameter.
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Cumulative distribution function `C(u, v)`.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        let th = self.theta;
        (u.powf(-th) + v.powf(-th) - 1.0).powf(-1.0 / th)
    }

    /// Probability density (copula density) `c(u, v)`.
    pub fn pdf(&self, u: f64, v: f64) -> f64 {
        let th = self.theta;
        let a = (th + 1.0) * (u * v).powf(-(th + 1.0));
        let b = u.powf(-th) + v.powf(-th) - 1.0;
        let c = -(2.0 * th + 1.0) / th;
        a * b.powf(c)
    }

    /// Kendall's tau implied by `theta`: `tau = theta / (theta + 2)`.
    pub fn tau(&self) -> f64 {
        self.theta / (self.theta + 2.0)
    }

    /// Invert the Kendall's-tau mapping: `theta = 2 tau / (1 - tau)`.
    pub fn theta_from_tau(tau: f64) -> f64 {
        2.0 * tau / (1.0 - tau)
    }
}

/// Gumbel copula, `theta >= 1` (independence at `theta = 1`).
///
/// CDF: `C(u, v) = exp(-((-ln u)^theta + (-ln v)^theta)^(1/theta))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GumbelCopula {
    /// Dependence parameter.
    pub theta: f64,
}

impl GumbelCopula {
    /// Construct a Gumbel copula with the given dependence parameter.
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Cumulative distribution function `C(u, v)`.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        let th = self.theta;
        let h = (-u.ln()).powf(th) + (-v.ln()).powf(th);
        (-h.powf(1.0 / th)).exp()
    }

    /// Probability density (copula density) `c(u, v)`.
    pub fn pdf(&self, u: f64, v: f64) -> f64 {
        let th = self.theta;
        let x = -u.ln();
        let y = -v.ln();
        let xt = x.powf(th);
        let yt = y.powf(th);
        let s = xt + yt;
        let s_pow = s.powf(1.0 / th);

        let a = (-s_pow).exp();
        let b = s_pow + th - 1.0;
        let c = s.powf(1.0 / th - 2.0);
        let d = (x * y).powf(th - 1.0);
        let e = (u * v).recip();

        a * b * c * d * e
    }

    /// Kendall's tau implied by `theta`: `tau = (theta - 1) / theta`.
    pub fn tau(&self) -> f64 {
        (self.theta - 1.0) / self.theta
    }

    /// Invert the Kendall's-tau mapping: `theta = 1 / (1 - tau)`.
    pub fn theta_from_tau(tau: f64) -> f64 {
        1.0 / (1.0 - tau)
    }
}

/// Frank copula, `theta != 0` (independence limit `theta -> 0`).
///
/// CDF: `C(u, v) = -1/theta * ln(1 - (1-e^{-theta u})(1-e^{-theta v}) / (1 - e^{-theta}))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrankCopula {
    /// Dependence parameter.
    pub theta: f64,
}

impl FrankCopula {
    /// Construct a Frank copula with the given dependence parameter.
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Cumulative distribution function `C(u, v)`.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        let th = self.theta;
        // dim == 2: den = (1 - e^{-theta})^(dim-1) = 1 - e^{-theta}.
        let num = (1.0 - (-th * u).exp()) * (1.0 - (-th * v).exp());
        let den = 1.0 - (-th).exp();
        -1.0 / th * (1.0 - num / den).ln()
    }

    /// Probability density (copula density) `c(u, v)`.
    pub fn pdf(&self, u: f64, v: f64) -> f64 {
        let th = self.theta;
        let g_ = (-th * (u + v)).exp() - 1.0;
        let g1 = (-th).exp() - 1.0;

        let num = -th * g1 * (1.0 + g_);
        let aux = ((-th * u).exp() - 1.0) * ((-th * v).exp() - 1.0) + g1;
        let den = aux * aux;
        num / den
    }

    /// Kendall's tau implied by `theta`.
    ///
    /// For `theta <= 1` a Taylor expansion is used; otherwise the closed
    /// form `tau = 1 + 4 (D_1(theta) - 1) / theta` with the first-order
    /// Debye function `D_1` is evaluated. This matches the branch logic of
    /// the reference implementation.
    pub fn tau(&self) -> f64 {
        let th = self.theta;
        if th <= 1.0 {
            tau_frank_expansion(th)
        } else {
            let d = debye1(th);
            1.0 + 4.0 * (d - 1.0) / th
        }
    }
}

/// Taylor-series approximation of Frank's Kendall tau, valid for small
/// `|theta|`. Coefficients reproduce the reference expansion.
fn tau_frank_expansion(x: f64) -> f64 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    let x11 = x9 * x2;
    x / 9.0 - x3 / 900.0 + x5 / 52920.0 - x7 / 2_721_600.0 + x9 / 131_725_440.0
        - x11 * 691.0 / 4_249_941_696_000.0
}

/// First-order Debye function `D_1(a) = (1/a) * integral_0^a t/(e^t - 1) dt`.
///
/// Evaluated by adaptive Gauss-Kronrod (G7-K15) quadrature, mirroring the
/// `scipy.integrate.quad` call used by the reference. The lower limit is
/// shifted off zero by the same tiny epsilon the reference uses, so the two
/// integrals agree to well below 1e-9.
fn debye1(a: f64) -> f64 {
    const EPSILON: f64 = f64::EPSILON * 100.0;
    let integrand = |t: f64| t / (t.exp() - 1.0);
    adaptive_quad(integrand, EPSILON, a, 1e-13, 40) / a
}

/// Gauss-Kronrod (7-15) nodes/weights on `[-1, 1]`.
const GK_NODES: [f64; 15] = [
    -0.991_455_371_120_813,
    -0.949_107_912_342_758_5,
    -0.864_864_423_359_769_1,
    -0.741_531_185_599_394_4,
    -0.586_087_235_467_691_1,
    -0.405_845_151_377_397_2,
    -0.207_784_955_007_898_47,
    0.0,
    0.207_784_955_007_898_47,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_4,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_813,
];

const GK_WEIGHTS: [f64; 15] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_88,
    0.209_482_141_084_727_83,
    0.204_432_940_075_298_88,
    0.190_350_578_064_785_4,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_92,
    0.104_790_010_322_250_18,
    0.063_092_092_629_978_55,
    0.022_935_322_010_529_22,
];

/// Gauss (7-point) weights for the nodes at odd indices of `GK_NODES`.
const G_WEIGHTS: [f64; 7] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_67,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
    0.381_830_050_505_118_9,
    0.279_705_391_489_276_67,
    0.129_484_966_168_869_7,
];

/// One Gauss-Kronrod panel on `[a, b]`, returning `(integral, error_estimate)`.
fn gk15<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let mut kron = 0.0;
    let mut gauss = 0.0;
    for i in 0..15 {
        let x = c + h * GK_NODES[i];
        let fx = f(x);
        kron += GK_WEIGHTS[i] * fx;
        if i % 2 == 1 {
            gauss += G_WEIGHTS[i / 2] * fx;
        }
    }
    kron *= h;
    gauss *= h;
    (kron, (kron - gauss).abs())
}

/// Adaptive Gauss-Kronrod integration of `f` over `[a, b]`.
fn adaptive_quad<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64, max_depth: u32) -> f64 {
    fn recur<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, whole: f64, tol: f64, depth: u32) -> f64 {
        let m = 0.5 * (a + b);
        let (left, _) = gk15(f, a, m);
        let (right, _) = gk15(f, m, b);
        let err = (left + right - whole).abs();
        if depth == 0 || err <= tol {
            left + right
        } else {
            recur(f, a, m, left, 0.5 * tol, depth - 1) + recur(f, m, b, right, 0.5 * tol, depth - 1)
        }
    }
    let (whole, _) = gk15(&f, a, b);
    recur(&f, a, b, whole, tol, max_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clayton_tau_roundtrip() {
        for &tau in &[0.1, 0.3, 0.5, 0.7] {
            let th = ClaytonCopula::theta_from_tau(tau);
            assert!((ClaytonCopula::new(th).tau() - tau).abs() < 1e-12);
        }
    }

    #[test]
    fn gumbel_tau_roundtrip() {
        for &tau in &[0.1, 0.3, 0.5, 0.7] {
            let th = GumbelCopula::theta_from_tau(tau);
            assert!((GumbelCopula::new(th).tau() - tau).abs() < 1e-12);
        }
    }

    #[test]
    fn debye1_known_value() {
        // D_1(1) ~ 0.7775046341122482 (standard reference value).
        assert!((debye1(1.0) - 0.777_504_634_112_248_2).abs() < 1e-12);
    }

    #[test]
    fn frank_tau_branch_continuity() {
        // Expansion (theta=1) and the Debye branch agree near the boundary.
        let lo = FrankCopula::new(1.0).tau();
        let hi = FrankCopula::new(1.000_000_1).tau();
        assert!((lo - hi).abs() < 1e-6);
    }

    #[test]
    fn copula_grounds_at_corners() {
        // C(u, 1) = u and C(1, v) = v (uniform margins) for all families.
        for u in [0.2, 0.5, 0.8] {
            assert!((ClaytonCopula::new(2.0).cdf(u, 1.0 - 1e-12) - u).abs() < 1e-9);
            assert!((GumbelCopula::new(2.0).cdf(u, 1.0 - 1e-12) - u).abs() < 1e-9);
            assert!((FrankCopula::new(2.0).cdf(u, 1.0 - 1e-9) - u).abs() < 1e-7);
        }
    }
}
