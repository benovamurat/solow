//! Kernel algebra for Gaussian processes.
//!
//! `k(x, y)` maps two `d`-vectors to a real number. Every kernel here is
//! stationary or dot-product-based, so the derivative structure is
//! straightforward and closed under sum / product / exponentiation.

use ndarray::{Array2, ArrayView1, ArrayView2};

/// The `Kernel` trait every leaf kernel implements.
pub trait Kernel: Send + Sync {
    /// Pointwise evaluation.
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64;

    /// Gram matrix `K = [k(xᵢ, xⱼ)]`.
    fn gram(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let v = self.call(x.row(i), x.row(j));
                out[[i, j]] = v;
                out[[j, i]] = v;
            }
        }
        out
    }

    /// Cross Gram `[k(aᵢ, bⱼ)]`.
    fn cross(&self, a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
        let n = a.nrows();
        let m = b.nrows();
        let mut out = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                out[[i, j]] = self.call(a.row(i), b.row(j));
            }
        }
        out
    }

    /// `k(x, x)` for every row — useful for predictive variances.
    fn diag(&self, x: ArrayView2<'_, f64>) -> Vec<f64> {
        (0..x.nrows()).map(|i| self.call(x.row(i), x.row(i))).collect()
    }
}

/// Radial-basis-function kernel: `σ² · exp(−‖x − y‖² / (2ℓ²))`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rbf {
    /// Length-scale ℓ.
    pub length_scale: f64,
    /// Output scale σ.
    pub sigma: f64,
}

impl Rbf {
    /// Construct with ℓ, σ = 1.
    pub fn new(length_scale: f64) -> Self {
        Self { length_scale, sigma: 1.0 }
    }
}

impl Kernel for Rbf {
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        let mut s = 0.0_f64;
        for i in 0..x.len() {
            let d = x[i] - y[i];
            s += d * d;
        }
        self.sigma * self.sigma * (-s / (2.0 * self.length_scale * self.length_scale)).exp()
    }
}

/// Matérn kernel with `ν ∈ {½, 3/2, 5/2}` closed forms.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matern {
    /// Smoothness parameter (0.5, 1.5, or 2.5).
    pub nu: f64,
    /// Length-scale ℓ.
    pub length_scale: f64,
}

impl Matern {
    /// Construct.
    pub fn new(nu: f64, length_scale: f64) -> Self {
        Self { nu, length_scale }
    }
}

impl Kernel for Matern {
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        let mut s = 0.0_f64;
        for i in 0..x.len() {
            let d = x[i] - y[i];
            s += d * d;
        }
        let r = s.sqrt() / self.length_scale;
        if (self.nu - 0.5).abs() < 1e-8 {
            (-r).exp()
        } else if (self.nu - 1.5).abs() < 1e-8 {
            let a = (3.0_f64).sqrt() * r;
            (1.0 + a) * (-a).exp()
        } else if (self.nu - 2.5).abs() < 1e-8 {
            let a = (5.0_f64).sqrt() * r;
            (1.0 + a + a * a / 3.0) * (-a).exp()
        } else {
            // Fall back to RBF at ν → ∞.
            (-0.5 * r * r).exp()
        }
    }
}

/// Rational-quadratic kernel: `(1 + ‖x − y‖² / (2 α ℓ²))^(−α)`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RationalQuadratic {
    /// Length-scale.
    pub length_scale: f64,
    /// Scale-mixture α.
    pub alpha: f64,
}

impl Kernel for RationalQuadratic {
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        let mut s = 0.0_f64;
        for i in 0..x.len() {
            let d = x[i] - y[i];
            s += d * d;
        }
        (1.0 + s / (2.0 * self.alpha * self.length_scale * self.length_scale)).powf(-self.alpha)
    }
}

/// Constant kernel: `c · 1`. Holds its single value.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantKernel(pub f64);

impl Kernel for ConstantKernel {
    fn call(&self, _x: ArrayView1<'_, f64>, _y: ArrayView1<'_, f64>) -> f64 {
        self.0
    }
}

/// White kernel: `σ² · δ(x, y)`. Holds its noise level.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhiteKernel(pub f64);

impl Kernel for WhiteKernel {
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        let mut eq = true;
        for i in 0..x.len() {
            if x[i] != y[i] {
                eq = false;
                break;
            }
        }
        if eq {
            self.0
        } else {
            0.0
        }
    }
}

/// Dot-product kernel: `σ² + x·y`. Holds its bias.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DotProduct(pub f64);

impl Kernel for DotProduct {
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        let mut s = self.0;
        for i in 0..x.len() {
            s += x[i] * y[i];
        }
        s
    }
}

/// Kernel sum.
pub struct Sum<A: Kernel, B: Kernel> {
    /// First component.
    pub a: A,
    /// Second component.
    pub b: B,
}

impl<A: Kernel, B: Kernel> Kernel for Sum<A, B> {
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        self.a.call(x.view(), y.view()) + self.b.call(x, y)
    }
}

/// Kernel product.
pub struct Product<A: Kernel, B: Kernel> {
    /// First component.
    pub a: A,
    /// Second component.
    pub b: B,
}

impl<A: Kernel, B: Kernel> Kernel for Product<A, B> {
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        self.a.call(x.view(), y.view()) * self.b.call(x, y)
    }
}

/// Kernel exponentiation `k(x, y)^p`.
pub struct Exponentiation<A: Kernel> {
    /// Base kernel.
    pub a: A,
    /// Power.
    pub exponent: f64,
}

impl<A: Kernel> Kernel for Exponentiation<A> {
    fn call(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        self.a.call(x, y).powf(self.exponent)
    }
}
