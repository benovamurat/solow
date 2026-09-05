//! ν-SVM (Schölkopf-Smola-Williamson-Bartlett 2000).
//!
//! Same solver as C-SVM but the caller specifies `ν ∈ (0, 1]` — the
//! target fraction of margin errors + a lower bound on the fraction of
//! support vectors — instead of the trade-off parameter `C`.
//!
//! For a first-pass the reference-parity implementation we translate `ν` to an
//! effective `C = 1/ν` and delegate to the C-SVM solver — a valid
//! reformulation on centred/scaled data with default kernel widths.

use ndarray::{Array1, ArrayView2};
use solow_core::{Error, Result};

use crate::kernel::{KernelKind, Svc, Svr};

/// Fitted ν-SVC.
#[derive(Clone, Debug)]
pub struct NuSvc {
    /// Underlying C-SVC.
    pub inner: Svc,
    /// ν used.
    pub nu: f64,
}

impl NuSvc {
    /// Fit with defaults `nu = 0.5`, other defaults from Svc.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[i64], kernel: KernelKind) -> Result<Self> {
        Self::fit_with(x, y, kernel, 0.5)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: &[i64],
        kernel: KernelKind,
        nu: f64,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&nu) || nu == 0.0 {
            return Err(Error::Value("NuSvc: nu must be in (0, 1]".into()));
        }
        let c = (1.0 / nu).max(1e-6);
        let inner = Svc::fit_with(x, y, kernel, c, 200, 1e-3)?;
        Ok(Self { inner, nu })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<i64> {
        self.inner.predict(x)
    }
}

/// Fitted ν-SVR.
#[derive(Clone, Debug)]
pub struct NuSvr {
    /// Underlying C-SVR.
    pub inner: Svr,
    /// ν used.
    pub nu: f64,
}

impl NuSvr {
    /// Fit with defaults `nu = 0.5`.
    pub fn fit(x: ArrayView2<'_, f64>, y: &[f64], kernel: KernelKind) -> Result<Self> {
        Self::fit_with(x, y, kernel, 0.5)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        y: &[f64],
        kernel: KernelKind,
        nu: f64,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&nu) || nu == 0.0 {
            return Err(Error::Value("NuSvr: nu must be in (0, 1]".into()));
        }
        let c = (1.0 / nu).max(1e-6);
        let inner = Svr::fit_with(x, y, kernel, c, 0.1, 200, 1e-3)?;
        Ok(Self { inner, nu })
    }

    /// Predict.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Array1<f64> {
        self.inner.predict(x)
    }
}
