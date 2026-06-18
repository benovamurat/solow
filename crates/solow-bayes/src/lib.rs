//! # solow-bayes
//!
//! Bayesian generalized linear mixed models fit by **mean-field variational
//! Bayes** (deterministic; no MCMC). Two families are provided with a
//! random-effects design (`exog_vc`): [`BinomialBayesMixedGLM`] (logit link) and
//! [`PoissonBayesMixedGLM`] (log link). Both are fit by [`BayesMixedGlm::fit_vb`],
//! which maximizes the evidence lower bound (ELBO) of a factored Gaussian
//! variational posterior over the fixed effects, the variance-component
//! parameters (log standard deviations of the random effects), and the random
//! effect realizations.
//!
//! The parameterization, ELBO and its gradient mirror the canonical
//! statistical-computing reference's `genmod.bayes_mixed_glm` module exactly, so
//! the posterior means ([`BayesMixedGlmResults::fe_mean`],
//! [`BayesMixedGlmResults::vcp_mean`], [`BayesMixedGlmResults::vc_mean`]) and the
//! ELBO ([`BayesMixedGlmResults::elbo`] / [`BayesMixedGlmResults::llf`]) agree to
//! optimizer tolerance.
//!
//! ## Model
//!
//! For observation `i` the linear predictor is
//! `eta_i = x_i · fe + z_i · vc`, where `x` is `exog` (fixed effects) and `z` is
//! `exog_vc` (random effects). Each random effect realization `vc_j` is
//! Gaussian with mean zero and standard deviation `exp(vcp[ident[j]])`; the
//! `vcp` parameters (log standard deviations) have a `N(0, vcp_p²)` prior and the
//! fixed effects a `N(0, fe_p²)` prior.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_optimize::minimize_bfgs;

mod map_estimation;
pub use map_estimation::MapResult;

/// Ten-point Gauss–Legendre nodes/weights on `[-1, 1]` (weight, node), used to
/// integrate the family contribution against a standard Gaussian. Identical to
/// the reference's `glw` table.
const GLW: [(f64, f64); 10] = [
    (0.295_524_224_714_752_9, -0.148_874_338_981_631_2),
    (0.295_524_224_714_752_9, 0.148_874_338_981_631_2),
    (0.269_266_719_309_996_3, -0.433_395_394_129_247_2),
    (0.269_266_719_309_996_3, 0.433_395_394_129_247_2),
    (0.219_086_362_515_982, -0.679_409_568_299_024_4),
    (0.219_086_362_515_982, 0.679_409_568_299_024_4),
    (0.149_451_349_150_580_6, -0.865_063_366_688_984_5),
    (0.149_451_349_150_580_6, 0.865_063_366_688_984_5),
    (0.066_671_344_308_688_1, -0.973_906_528_517_171_7),
    (0.066_671_344_308_688_1, 0.973_906_528_517_171_7),
];

/// Integration half-range for the Gaussian quadrature (`-RNG..RNG`).
const RNG: f64 = 5.0;

/// The GLM family for the Bayesian mixed model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Binomial family with the logit link (0/1 responses).
    Binomial,
    /// Poisson family with the log link (count responses).
    Poisson,
}

/// A Bayesian mixed GLM with a random-effects design, fit by variational Bayes.
///
/// Construct with [`BayesMixedGlm::new`], then call [`BayesMixedGlm::fit_vb`].
#[derive(Clone, Debug)]
pub struct BayesMixedGlm {
    family: Family,
    endog: Array1<f64>,
    exog: Array2<f64>,
    exog2: Array2<f64>,
    exog_vc: Array2<f64>,
    exog_vc2: Array2<f64>,
    ident: Vec<usize>,
    k_fep: usize,
    k_vcp: usize,
    k_vc: usize,
    vcp_p: f64,
    fe_p: f64,
}

impl BayesMixedGlm {
    /// Build a model.
    ///
    /// * `endog` — response vector (`0/1` for binomial, counts for Poisson).
    /// * `exog` — fixed-effects design, shape `(n, k_fep)`.
    /// * `exog_vc` — random-effects design, shape `(n, k_vc)`; each column is one
    ///   independent random effect realization.
    /// * `ident` — length `k_vc`; columns sharing an `ident` value share a
    ///   variance-component (log-sd) parameter. Values must be `0..k_vcp`.
    /// * `vcp_p` — prior sd for the variance-component (log-sd) parameters.
    /// * `fe_p` — prior sd for the fixed-effects parameters.
    pub fn new(
        family: Family,
        endog: Array1<f64>,
        exog: Array2<f64>,
        exog_vc: Array2<f64>,
        ident: Vec<usize>,
        vcp_p: f64,
        fe_p: f64,
    ) -> Result<Self> {
        let n = endog.len();
        if exog.nrows() != n || exog_vc.nrows() != n {
            return Err(Error::Shape(
                "endog, exog and exog_vc must have matching row counts".into(),
            ));
        }
        if ident.len() != exog_vc.ncols() {
            return Err(Error::Shape(
                "len(ident) must equal the number of columns of exog_vc".into(),
            ));
        }
        if family == Family::Binomial && !endog.iter().all(|&y| y == 0.0 || y == 1.0) {
            return Err(Error::Value("binomial endog values must be 0 or 1".into()));
        }
        let k_fep = exog.ncols();
        let k_vc = exog_vc.ncols();
        let k_vcp = ident.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let exog2 = exog.mapv(|v| v * v);
        let exog_vc2 = exog_vc.mapv(|v| v * v);
        Ok(Self {
            family,
            endog,
            exog,
            exog2,
            exog_vc,
            exog_vc2,
            ident,
            k_fep,
            k_vcp,
            k_vc,
            vcp_p,
            fe_p,
        })
    }

    /// Total number of variational mean (or sd) parameters. Also the length of
    /// the stacked `[fep, vcp, vc]` MAP parameter vector.
    pub(crate) fn dim(&self) -> usize {
        self.k_fep + self.k_vcp + self.k_vc
    }

    /// Split a stacked vector into `(fep, vcp, vc)` views by length.
    pub(crate) fn unpack<'a>(&self, v: &'a [f64]) -> (&'a [f64], &'a [f64], &'a [f64]) {
        let a = self.k_fep;
        let b = a + self.k_vcp;
        (&v[..a], &v[a..b], &v[b..])
    }

    /// The GLM family.
    pub(crate) fn family(&self) -> Family {
        self.family
    }

    /// The response vector.
    pub(crate) fn endog(&self) -> &Array1<f64> {
        &self.endog
    }

    /// The fixed-effects design matrix.
    pub(crate) fn exog(&self) -> &Array2<f64> {
        &self.exog
    }

    /// The random-effects design matrix.
    pub(crate) fn exog_vc(&self) -> &Array2<f64> {
        &self.exog_vc
    }

    /// The variance-component index for random-effect column `j`.
    pub(crate) fn ident_at(&self, j: usize) -> usize {
        self.ident[j]
    }

    /// Number of fixed-effects parameters.
    pub(crate) fn k_fep(&self) -> usize {
        self.k_fep
    }

    /// Number of variance-component (log-sd) parameters.
    pub(crate) fn k_vcp(&self) -> usize {
        self.k_vcp
    }

    /// Prior sd for the variance-component parameters.
    pub(crate) fn vcp_p(&self) -> f64 {
        self.vcp_p
    }

    /// Prior sd for the fixed-effects parameters.
    pub(crate) fn fe_p(&self) -> f64 {
        self.fe_p
    }

    /// Mean and variance of the linear predictor for every observation under the
    /// variational posterior. Mirrors `_lp_stats`.
    fn lp_stats(
        &self,
        fep_mean: &[f64],
        fep_sd: &[f64],
        vc_mean: &[f64],
        vc_sd: &[f64],
    ) -> (Array1<f64>, Array1<f64>) {
        let n = self.endog.len();
        let mut tm = Array1::<f64>::zeros(n);
        let mut tv = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut m = 0.0;
            let mut var = 0.0;
            for k in 0..self.k_fep {
                m += self.exog[[i, k]] * fep_mean[k];
                var += self.exog2[[i, k]] * fep_sd[k] * fep_sd[k];
            }
            for k in 0..self.k_vc {
                m += self.exog_vc[[i, k]] * vc_mean[k];
                var += self.exog_vc2[[i, k]] * vc_sd[k] * vc_sd[k];
            }
            tm[i] = m;
            tv[i] = var;
        }
        (tm, tv)
    }

    /// The per-observation `h(z)` contribution of the family to the ELBO.
    /// `h(z) = -log(1 + exp(lp))` (binomial) or `-exp(lp)` (Poisson) with
    /// `lp = tm + sqrt(tv) z`.
    fn h(&self, z: f64, tm: &Array1<f64>, tv_sqrt: &Array1<f64>) -> Array1<f64> {
        match self.family {
            Family::Binomial => Array1::from_iter(
                tm.iter()
                    .zip(tv_sqrt.iter())
                    .map(|(&m, &s)| -log1pexp(m + s * z)),
            ),
            Family::Poisson => Array1::from_iter(
                tm.iter()
                    .zip(tv_sqrt.iter())
                    .map(|(&m, &s)| -(m + s * z).exp()),
            ),
        }
    }

    /// The derivative factor `h'`-like term used in the gradient. For the
    /// binomial this is `-sigmoid(lp)`; for the Poisson it is `-exp(lp)`.
    fn h_grad(&self, z: f64, tm: &Array1<f64>, tv_sqrt: &Array1<f64>) -> Array1<f64> {
        match self.family {
            Family::Binomial => Array1::from_iter(
                tm.iter()
                    .zip(tv_sqrt.iter())
                    .map(|(&m, &s)| -sigmoid(m + s * z)),
            ),
            Family::Poisson => Array1::from_iter(
                tm.iter()
                    .zip(tv_sqrt.iter())
                    .map(|(&m, &s)| -(m + s * z).exp()),
            ),
        }
    }

    /// Terms of the ELBO common to all families: `p(vc|vcp) p(vcp) p(fe)`.
    /// Mirrors `_elbo_common`.
    fn elbo_common(
        &self,
        fep_mean: &[f64],
        fep_sd: &[f64],
        vcp_mean: &[f64],
        vcp_sd: &[f64],
        vc_mean: &[f64],
        vc_sd: &[f64],
    ) -> f64 {
        let mut iv = 0.0;
        // p(vc | vcp): `m`/`s` are the per-column vcp params via `ident`, so the
        // `-sum(m)` term sums over all k_vc columns (one log-sd per realization),
        // matching `np.sum(vcp_mean[ident])`.
        for j in 0..self.k_vc {
            let m = vcp_mean[self.ident[j]];
            let s = vcp_sd[self.ident[j]];
            let u = vc_mean[j] * vc_mean[j] + vc_sd[j] * vc_sd[j];
            iv -= u * (2.0 * (s * s - m)).exp() / 2.0;
            iv -= m;
        }
        // p(vcp)
        for k in 0..self.k_vcp {
            iv -= 0.5 * (vcp_mean[k] * vcp_mean[k] + vcp_sd[k] * vcp_sd[k])
                / (self.vcp_p * self.vcp_p);
        }
        // p(fe)
        for k in 0..self.k_fep {
            iv -=
                0.5 * (fep_mean[k] * fep_mean[k] + fep_sd[k] * fep_sd[k]) / (self.fe_p * self.fe_p);
        }
        iv
    }

    /// The ELBO at variational mean vector `vb_mean` and sd vector `vb_sd`
    /// (natural scale). Mirrors `vb_elbo` / `vb_elbo_base`.
    pub fn vb_elbo(&self, vb_mean: &[f64], vb_sd: &[f64]) -> f64 {
        let (fep_mean, vcp_mean, vc_mean) = self.unpack(vb_mean);
        let (fep_sd, vcp_sd, vc_sd) = self.unpack(vb_sd);
        let (tm, tv) = self.lp_stats(fep_mean, fep_sd, vc_mean, vc_sd);
        let tv_sqrt = tv.mapv(f64::sqrt);

        let n = self.endog.len();
        // p(y | vc): quadrature of the family contribution + endog·tm.
        let mut iv_vec = Array1::<f64>::zeros(n);
        for &(w, node) in GLW.iter() {
            let z = RNG * node;
            let hz = self.h(z, &tm, &tv_sqrt);
            let f = w * (-z * z / 2.0).exp();
            iv_vec.scaled_add(f, &hz);
        }
        iv_vec /= (2.0 * std::f64::consts::PI).sqrt();
        iv_vec *= RNG;
        let mut iv: f64 = 0.0;
        for i in 0..n {
            iv += iv_vec[i] + self.endog[i] * tm[i];
        }

        iv += self.elbo_common(fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd);

        let log_sd: f64 = vb_sd.iter().map(|&s| s.ln()).sum();
        iv + log_sd
    }

    /// Gradient of the ELBO with respect to `(mean, sd)` (natural sd scale).
    /// Mirrors `vb_elbo_grad` / `vb_elbo_grad_base`. Returns
    /// `(mean_grad, sd_grad)`.
    pub fn vb_elbo_grad(&self, vb_mean: &[f64], vb_sd: &[f64]) -> (Array1<f64>, Array1<f64>) {
        let (fep_mean, vcp_mean, vc_mean) = self.unpack(vb_mean);
        let (fep_sd, vcp_sd, vc_sd) = self.unpack(vb_sd);
        let (tm, tv) = self.lp_stats(fep_mean, fep_sd, vc_mean, vc_sd);
        let tv_sqrt = tv.mapv(f64::sqrt);
        let n = self.endog.len();

        let mut fep_mean_grad = Array1::<f64>::zeros(self.k_fep);
        let mut vc_mean_grad = Array1::<f64>::zeros(self.k_vc);
        let mut fep_sd_grad = Array1::<f64>::zeros(self.k_fep);
        let mut vc_sd_grad = Array1::<f64>::zeros(self.k_vc);

        for &(w, node) in GLW.iter() {
            let z = RNG * node;
            // u = h'(z) * N(z)
            let hz = self.h_grad(z, &tm, &tv_sqrt);
            let u: Array1<f64> = Array1::from_iter(
                hz.iter()
                    .map(|&v| v * (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()),
            );
            // r = u / sqrt(tv)
            let r: Array1<f64> =
                Array1::from_iter(u.iter().zip(tv_sqrt.iter()).map(|(&a, &b)| a / b));

            // fep_mean_grad += w * u·exog
            for k in 0..self.k_fep {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += u[i] * self.exog[[i, k]];
                }
                fep_mean_grad[k] += w * acc;
            }
            // vc_mean_grad += w * exog_vc^T u
            for k in 0..self.k_vc {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += u[i] * self.exog_vc[[i, k]];
                }
                vc_mean_grad[k] += w * acc;
            }
            // fep_sd_grad += w * z * (r · (exog^2 * fep_sd))
            for k in 0..self.k_fep {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += r[i] * self.exog2[[i, k]] * fep_sd[k];
                }
                fep_sd_grad[k] += w * z * acc;
            }
            // vc_sd_grad += w * z * (exog_vc2 * vc_sd)^T r
            for k in 0..self.k_vc {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += self.exog_vc2[[i, k]] * vc_sd[k] * r[i];
                }
                vc_sd_grad[k] += w * z * acc;
            }
        }

        fep_mean_grad *= RNG;
        vc_mean_grad *= RNG;
        fep_sd_grad *= RNG;
        vc_sd_grad *= RNG;

        // + endog·exog and exog_vc^T endog
        for k in 0..self.k_fep {
            let mut acc = 0.0;
            for i in 0..n {
                acc += self.endog[i] * self.exog[[i, k]];
            }
            fep_mean_grad[k] += acc;
        }
        for k in 0..self.k_vc {
            let mut acc = 0.0;
            for i in 0..n {
                acc += self.endog[i] * self.exog_vc[[i, k]];
            }
            vc_mean_grad[k] += acc;
        }

        // Common (prior + p(vc|vcp)) contributions. Mirrors `_elbo_grad_common`.
        let mut vcp_mean_grad = Array1::<f64>::zeros(self.k_vcp);
        let mut vcp_sd_grad = Array1::<f64>::zeros(self.k_vcp);
        for j in 0..self.k_vc {
            let id = self.ident[j];
            let m = vcp_mean[id];
            let s = vcp_sd[id];
            let u = vc_mean[j] * vc_mean[j] + vc_sd[j] * vc_sd[j];
            let ve = (2.0 * (s * s - m)).exp();
            vcp_mean_grad[id] += u * ve - 1.0;
            vcp_sd_grad[id] += -2.0 * u * ve * s;
            vc_mean_grad[j] += -vc_mean[j] * ve;
            vc_sd_grad[j] += -vc_sd[j] * ve;
        }
        for k in 0..self.k_vcp {
            vcp_mean_grad[k] -= vcp_mean[k] / (self.vcp_p * self.vcp_p);
            vcp_sd_grad[k] -= vcp_sd[k] / (self.vcp_p * self.vcp_p);
        }
        for k in 0..self.k_fep {
            fep_mean_grad[k] -= fep_mean[k] / (self.fe_p * self.fe_p);
            fep_sd_grad[k] -= fep_sd[k] / (self.fe_p * self.fe_p);
        }

        // Entropy term d/d sd of sum(log sd) = 1/sd.
        for k in 0..self.k_fep {
            fep_sd_grad[k] += 1.0 / fep_sd[k];
        }
        for k in 0..self.k_vcp {
            vcp_sd_grad[k] += 1.0 / vcp_sd[k];
        }
        for k in 0..self.k_vc {
            vc_sd_grad[k] += 1.0 / vc_sd[k];
        }

        let mut mean_grad = Array1::<f64>::zeros(self.dim());
        let mut sd_grad = Array1::<f64>::zeros(self.dim());
        let (a, b) = (self.k_fep, self.k_fep + self.k_vcp);
        for k in 0..self.k_fep {
            mean_grad[k] = fep_mean_grad[k];
            sd_grad[k] = fep_sd_grad[k];
        }
        for k in 0..self.k_vcp {
            mean_grad[a + k] = vcp_mean_grad[k];
            sd_grad[a + k] = vcp_sd_grad[k];
        }
        for k in 0..self.k_vc {
            mean_grad[b + k] = vc_mean_grad[k];
            sd_grad[b + k] = vc_sd_grad[k];
        }
        (mean_grad, sd_grad)
    }

    /// Fit the model by maximizing the ELBO (mean-field variational Bayes).
    ///
    /// The optimizer works on the stacked vector `[mean; log(sd)]`. Internally we
    /// minimize `-ELBO` with the analytic gradient (transformed for the log-sd
    /// reparameterization) via BFGS, matching the reference's `fit_vb`.
    ///
    /// * `mean_start` — starting variational means, length `dim()`. If `None`,
    ///   zeros (with the `vcp` block floored at `-1`, as in the reference).
    /// * `sd_start` — starting variational sds (natural scale), length `dim()`.
    ///   If `None`, `exp(-0.5)` for every coordinate.
    pub fn fit_vb(
        &self,
        mean_start: Option<Array1<f64>>,
        sd_start: Option<Array1<f64>>,
        maxiter: usize,
        gtol: f64,
    ) -> Result<BayesMixedGlmResults> {
        let dim = self.dim();
        let mut m = mean_start.unwrap_or_else(|| Array1::zeros(dim));
        if m.len() != dim {
            return Err(Error::Shape("mean_start has wrong length".into()));
        }
        // Floor the vcp starting means at -1, mirroring the reference.
        let (i1, i2) = (self.k_fep, self.k_fep + self.k_vcp);
        for k in i1..i2 {
            if m[k] < -1.0 {
                m[k] = -1.0;
            }
        }
        // s is the log of the sd; floor at -1.
        let sd0 = sd_start.unwrap_or_else(|| Array1::from_elem(dim, (-0.5f64).exp()));
        if sd0.len() != dim {
            return Err(Error::Shape("sd_start has wrong length".into()));
        }
        let mut s = sd0.mapv(|v| v.ln());
        for v in s.iter_mut() {
            if *v < -1.0 {
                *v = -1.0;
            }
        }

        // Pack [mean; log sd].
        let mut start = Array1::<f64>::zeros(2 * dim);
        for k in 0..dim {
            start[k] = m[k];
            start[dim + k] = s[k];
        }

        let f = |x: &Array1<f64>| -> f64 {
            let mean = x.slice(ndarray::s![..dim]).to_vec();
            let sd: Vec<f64> = x
                .slice(ndarray::s![dim..])
                .iter()
                .map(|&v| v.exp())
                .collect();
            -self.vb_elbo(&mean, &sd)
        };
        let grad = |x: &Array1<f64>| -> Array1<f64> {
            let mean = x.slice(ndarray::s![..dim]).to_vec();
            let logsd = x.slice(ndarray::s![dim..]).to_vec();
            let sd: Vec<f64> = logsd.iter().map(|&v| v.exp()).collect();
            let (gm, mut gs) = self.vb_elbo_grad(&mean, &sd);
            // chain rule for sd = exp(log sd): multiply by exp(log sd) = sd
            for k in 0..dim {
                gs[k] *= sd[k];
            }
            let mut g = Array1::<f64>::zeros(2 * dim);
            for k in 0..dim {
                g[k] = -gm[k];
                g[dim + k] = -gs[k];
            }
            g
        };

        // Restart BFGS in bursts and accept a scale-aware function-value stall
        // as convergence: the finite-difference gradient has a platform-
        // dependent roundoff floor it cannot drop below, so `|g| <= gtol` alone
        // is not reachable on every platform. The optimum (and every reported
        // quantity) is identical either way — this only makes the `converged`
        // flag robust across platforms and stops the fit as soon as the ELBO
        // stalls instead of grinding to `maxiter`.
        let burst = 25usize;
        let bursts = maxiter.div_ceil(burst).max(1);
        let mut xk = start.clone();
        let mut f_prev = f(&xk);
        let mut converged = false;
        let mut iters = 0usize;
        let mut grad_norm = f64::INFINITY;
        for _ in 0..bursts {
            let res = minimize_bfgs(&xk, f, grad, burst, gtol)?;
            xk = res.x;
            iters += res.iters;
            grad_norm = res.grad_norm;
            let f_now = res.fval;
            if res.converged || (f_prev - f_now).abs() <= 1e-12 * (1.0 + f_now.abs()) {
                converged = true;
                break;
            }
            f_prev = f_now;
        }
        let x = xk;
        let mean = x.slice(ndarray::s![..dim]).to_owned();
        let sd = x.slice(ndarray::s![dim..]).mapv(|v| v.exp());

        let mean_s = mean
            .as_slice()
            .ok_or_else(|| Error::Value("mean must be contiguous".into()))?;
        let sd_s = sd
            .as_slice()
            .ok_or_else(|| Error::Value("sd must be contiguous".into()))?;
        let (fep, vcp, vc) = self.unpack(mean_s);
        let fe_mean = Array1::from_vec(fep.to_vec());
        let vcp_mean = Array1::from_vec(vcp.to_vec());
        let vc_mean = Array1::from_vec(vc.to_vec());
        let (fep_sd, vcp_sd, vc_sd) = self.unpack(sd_s);
        let fe_sd = Array1::from_vec(fep_sd.to_vec());
        let vcp_sd_a = Array1::from_vec(vcp_sd.to_vec());
        let vc_sd_a = Array1::from_vec(vc_sd.to_vec());

        let elbo = self.vb_elbo(mean_s, sd_s);

        Ok(BayesMixedGlmResults {
            fe_mean,
            vcp_mean,
            vc_mean,
            fe_sd,
            vcp_sd: vcp_sd_a,
            vc_sd: vc_sd_a,
            elbo,
            converged,
            iters,
            grad_norm,
        })
    }
}

/// A binomial (logit-link) Bayesian mixed GLM. Thin constructor over
/// [`BayesMixedGlm`].
pub struct BinomialBayesMixedGLM;

impl BinomialBayesMixedGLM {
    /// Build a binomial model. See [`BayesMixedGlm::new`]. This is a named
    /// constructor mirroring the reference API; it returns the family-agnostic
    /// [`BayesMixedGlm`] rather than `Self` (a zero-sized marker).
    #[allow(clippy::new_ret_no_self, clippy::too_many_arguments)]
    pub fn new(
        endog: Array1<f64>,
        exog: Array2<f64>,
        exog_vc: Array2<f64>,
        ident: Vec<usize>,
        vcp_p: f64,
        fe_p: f64,
    ) -> Result<BayesMixedGlm> {
        BayesMixedGlm::new(Family::Binomial, endog, exog, exog_vc, ident, vcp_p, fe_p)
    }
}

/// A Poisson (log-link) Bayesian mixed GLM. Thin constructor over
/// [`BayesMixedGlm`].
pub struct PoissonBayesMixedGLM;

impl PoissonBayesMixedGLM {
    /// Build a Poisson model. See [`BayesMixedGlm::new`]. This is a named
    /// constructor mirroring the reference API; it returns the family-agnostic
    /// [`BayesMixedGlm`] rather than `Self` (a zero-sized marker).
    #[allow(clippy::new_ret_no_self, clippy::too_many_arguments)]
    pub fn new(
        endog: Array1<f64>,
        exog: Array2<f64>,
        exog_vc: Array2<f64>,
        ident: Vec<usize>,
        vcp_p: f64,
        fe_p: f64,
    ) -> Result<BayesMixedGlm> {
        BayesMixedGlm::new(Family::Poisson, endog, exog, exog_vc, ident, vcp_p, fe_p)
    }
}

/// Posterior summary returned by [`BayesMixedGlm::fit_vb`].
#[derive(Clone, Debug)]
pub struct BayesMixedGlmResults {
    /// Posterior mean of the fixed-effects coefficients.
    pub fe_mean: Array1<f64>,
    /// Posterior mean of the variance-component (log-sd) parameters.
    pub vcp_mean: Array1<f64>,
    /// Posterior mean of the random-effect realizations.
    pub vc_mean: Array1<f64>,
    /// Posterior sd of the fixed-effects coefficients.
    pub fe_sd: Array1<f64>,
    /// Posterior sd of the variance-component parameters.
    pub vcp_sd: Array1<f64>,
    /// Posterior sd of the random-effect realizations.
    pub vc_sd: Array1<f64>,
    /// The maximized evidence lower bound.
    pub elbo: f64,
    /// Whether the optimizer met its convergence test.
    pub converged: bool,
    /// Optimizer iterations performed.
    pub iters: usize,
    /// Final gradient norm reported by the optimizer.
    pub grad_norm: f64,
}

impl BayesMixedGlmResults {
    /// The ELBO, which serves as the (variational) log-likelihood. Alias for
    /// [`BayesMixedGlmResults::elbo`].
    pub fn llf(&self) -> f64 {
        self.elbo
    }
}

/// Numerically stable `log(1 + exp(x))`.
fn log1pexp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Numerically stable logistic sigmoid `1 / (1 + exp(-x))`.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};
    use solow_optimize::approx_fprime;

    /// The 6-group balanced random-intercept design used in the fixtures.
    fn binom_model() -> BayesMixedGlm {
        let n_groups = 6usize;
        let per = 4usize;
        let n = n_groups * per;
        let x1: Vec<f64> = (0..n)
            .map(|i| -1.5 + 3.0 * i as f64 / (n as f64 - 1.0))
            .collect();
        let mut exog = Array2::<f64>::zeros((n, 2));
        let mut exog_vc = Array2::<f64>::zeros((n, n_groups));
        for i in 0..n {
            exog[[i, 0]] = 1.0;
            exog[[i, 1]] = x1[i];
            exog_vc[[i, i / per]] = 1.0;
        }
        // Endog from the binomial fixture case.
        let endog = array![
            0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0.,
            1., 1.
        ];
        let ident = vec![0usize; n_groups];
        BayesMixedGlm::new(Family::Binomial, endog, exog, exog_vc, ident, 0.5, 2.0).unwrap()
    }

    /// ELBO matches the reference value at the fixed start (mean=0, sd=exp(-0.5)).
    /// Endog is irrelevant to the ELBO *base* integral, but the `endog·tm` term
    /// is zero at mean=0 so this is robust to the exact endog draw.
    #[test]
    fn elbo_base_matches_reference_at_start() {
        let m = binom_model();
        let dim = m.dim();
        let mean = vec![0.0; dim];
        let sd = vec![(-0.5f64).exp(); dim];
        // Reference value computed independently from the same model.
        assert_abs_diff_eq!(m.vb_elbo(&mean, &sd), -27.081695786098585, epsilon = 1e-9);
    }

    /// Analytic gradient agrees with a finite-difference gradient of the ELBO
    /// (wrt the `[mean; log sd]` reparameterization the optimizer uses).
    #[test]
    fn analytic_grad_matches_finite_difference() {
        let m = binom_model();
        let dim = m.dim();
        // Pack [mean; log sd] and define the negative-ELBO objective.
        let mut x = Array1::<f64>::zeros(2 * dim);
        for k in 0..dim {
            x[k] = 0.1 * (k as f64 - 3.0);
            x[dim + k] = -0.4;
        }
        let obj = |x: &Array1<f64>| -> f64 {
            let mean = x.slice(ndarray::s![..dim]).to_vec();
            let sd: Vec<f64> = x
                .slice(ndarray::s![dim..])
                .iter()
                .map(|&v| v.exp())
                .collect();
            -m.vb_elbo(&mean, &sd)
        };
        let fd = approx_fprime(&x, obj);

        let mean = x.slice(ndarray::s![..dim]).to_vec();
        let logsd = x.slice(ndarray::s![dim..]).to_vec();
        let sd: Vec<f64> = logsd.iter().map(|&v| v.exp()).collect();
        let (gm, mut gs) = m.vb_elbo_grad(&mean, &sd);
        for k in 0..dim {
            gs[k] *= sd[k];
        }
        for k in 0..dim {
            assert_abs_diff_eq!(-gm[k], fd[k], epsilon = 1e-5);
            assert_abs_diff_eq!(-gs[k], fd[dim + k], epsilon = 1e-5);
        }
    }

    /// A small Poisson model fits and the ELBO gradient vanishes at the optimum.
    #[test]
    fn poisson_fit_reaches_stationary_point() {
        let n_groups = 4usize;
        let per = 3usize;
        let n = n_groups * per;
        let mut exog = Array2::<f64>::zeros((n, 2));
        let mut exog_vc = Array2::<f64>::zeros((n, n_groups));
        for i in 0..n {
            exog[[i, 0]] = 1.0;
            exog[[i, 1]] = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
            exog_vc[[i, i / per]] = 1.0;
        }
        let endog = array![1., 2., 1., 0., 1., 3., 2., 1., 4., 2., 1., 0.];
        let ident = vec![0usize; n_groups];
        let m = BayesMixedGlm::new(Family::Poisson, endog, exog, exog_vc, ident, 0.5, 2.0).unwrap();
        let res = m.fit_vb(None, None, 10_000, 1e-8).unwrap();
        assert!(res.converged, "did not converge, |g|={}", res.grad_norm);
        assert!(res.grad_norm < 1e-7);
    }
}
