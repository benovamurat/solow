//! Marginal GEE regression for **categorical** (nominal / ordinal) responses.
//!
//! A multinomial outcome with `K` distinct levels is analyzed by expanding
//! every observation into `ncut = K − 1` binary indicator rows and fitting a
//! GEE to the expanded data.  Two response geometries are supported:
//!
//! * [`NominalGee`] — *unordered* categories.  Each cut `j` (a category other
//!   than the reference / largest level) gets its **own** coefficient block, so
//!   the expanded design is block-diagonal (`kron(eⱼ, xᵢ)`) and the cut
//!   probabilities are coupled through the multinomial-logit link
//!   `μ_j = e^{η_j} / (1 + Σ_k e^{η_k})`.  The indicator is `I(yᵢ = cutⱼ)`.
//!
//! * [`OrdinalGee`] — *ordered* categories.  A single covariate-effect vector
//!   is shared across cuts, augmented by one intercept per cut, and the
//!   ordinary logit link `μ = 1/(1+e^{−η})` is applied per row.  The indicator
//!   is `I(yᵢ > cutⱼ)` (a proportional-odds cumulative model).
//!
//! Within a single original observation the `ncut` indicators are
//! (deterministically) correlated; between observations the working
//! association is either [`CategoricalCov::Independence`] (zero) or
//! [`CategoricalCov::GlobalOddsRatio`] (the Heagerty–Zeger / Lumley global
//! odds-ratio structure, estimated by iteratively matching pooled `2×2`
//! cut-point tables).  Inference uses the cluster-robust sandwich covariance.
//!
//! Validated against an authoritative reference (`NominalGEE` / `OrdinalGEE`
//! with `GlobalOddsRatio`).

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_sf;
use solow_glm::{Family, Glm, Link};
use solow_linalg::{inv, solve};

/// Between-observation working association for categorical GEE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CategoricalCov {
    /// Indicators from *different* original observations are uncorrelated.
    /// There is no association parameter to estimate.
    Independence,
    /// The global odds-ratio structure of Heagerty–Zeger (ordinal) and Lumley:
    /// a single odds ratio governs the joint distribution of every
    /// between-observation indicator pair, estimated by matching pooled
    /// cut-point `2×2` tables.
    GlobalOddsRatio,
}

/// Whether the categorical response is nominal (unordered) or ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Nominal,
    Ordinal,
}

/// A nominal-response (unordered multinomial) marginal GEE awaiting estimation.
#[derive(Clone, Debug)]
pub struct NominalGee {
    inner: CategoricalGee,
}

/// An ordinal-response (proportional-odds cumulative) marginal GEE awaiting
/// estimation.
#[derive(Clone, Debug)]
pub struct OrdinalGee {
    inner: CategoricalGee,
}

impl NominalGee {
    /// Build a nominal GEE.
    ///
    /// `exog` is the *original* design (typically including an intercept
    /// column); `group_labels` assigns each observation to a cluster.  The
    /// response `endog` must take a small number of distinct values; the
    /// largest is treated as the reference category.
    pub fn new(
        endog: Array1<f64>,
        exog: Array2<f64>,
        group_labels: &[i64],
        cov: CategoricalCov,
    ) -> Result<Self> {
        Ok(NominalGee {
            inner: CategoricalGee::new(Kind::Nominal, endog, exog, group_labels, cov)?,
        })
    }

    /// Set the maximum number of outer (Fisher-scoring) iterations.
    pub fn maxiter(mut self, m: usize) -> Self {
        self.inner.maxiter = m;
        self
    }

    /// Set the convergence tolerance on the score-equation norm.
    pub fn ctol(mut self, t: f64) -> Self {
        self.inner.ctol = t;
        self
    }

    /// Fit the model.
    pub fn fit(&self) -> Result<CategoricalGeeResults> {
        self.inner.fit()
    }
}

impl OrdinalGee {
    /// Build an ordinal GEE.
    ///
    /// `exog` is the *original* design and **must not** contain an intercept
    /// column — one intercept per cut point is appended automatically (an
    /// intercept in `exog` would make the augmented design rank-deficient).
    pub fn new(
        endog: Array1<f64>,
        exog: Array2<f64>,
        group_labels: &[i64],
        cov: CategoricalCov,
    ) -> Result<Self> {
        Ok(OrdinalGee {
            inner: CategoricalGee::new(Kind::Ordinal, endog, exog, group_labels, cov)?,
        })
    }

    /// Set the maximum number of outer (Fisher-scoring) iterations.
    pub fn maxiter(mut self, m: usize) -> Self {
        self.inner.maxiter = m;
        self
    }

    /// Set the convergence tolerance on the score-equation norm.
    pub fn ctol(mut self, t: f64) -> Self {
        self.inner.ctol = t;
        self
    }

    /// Fit the model.
    pub fn fit(&self) -> Result<CategoricalGeeResults> {
        self.inner.fit()
    }
}

/// Shared machinery for nominal / ordinal categorical GEE.
#[derive(Clone, Debug)]
struct CategoricalGee {
    kind: Kind,
    cov: CategoricalCov,
    /// Number of cut points, `K − 1`.
    ncut: usize,
    /// Width of the *expanded* design (number of mean parameters).
    nparam: usize,
    /// Expanded design, one block of `ncut` rows per original observation, in
    /// cluster (sorted group-label) order.
    exog: Array2<f64>,
    /// Expanded binary indicators aligned with `exog`.
    endog: Array1<f64>,
    /// For each cluster, the row indices into `exog`/`endog` (already in
    /// expanded form, so a length-`m` cluster has `m·ncut` rows).
    groups: Vec<Vec<usize>>,
    /// For each cluster, the number of *original* observations in it.
    group_nobs: Vec<usize>,
    /// Total number of expanded rows.
    nrows: usize,
    maxiter: usize,
    ctol: f64,
}

impl CategoricalGee {
    fn new(
        kind: Kind,
        endog: Array1<f64>,
        exog: Array2<f64>,
        group_labels: &[i64],
        cov: CategoricalCov,
    ) -> Result<Self> {
        let n = endog.len();
        if n != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if group_labels.len() != n {
            return Err(Error::Shape("group_labels length != endog length".into()));
        }

        // Distinct outcome levels in ascending order; cuts drop the largest.
        let mut levels: Vec<f64> = endog.iter().copied().collect();
        levels.sort_by(|a, b| a.total_cmp(b));
        levels.dedup();
        if levels.len() < 2 {
            return Err(Error::Shape("endog must have at least two levels".into()));
        }
        let ncut = levels.len() - 1;
        let cuts = &levels[..ncut];
        let p = exog.ncols();

        let nparam = match kind {
            Kind::Nominal => ncut * p,
            Kind::Ordinal => ncut + p,
        };

        // Clusters in sorted group-label order (matching the reference's
        // `np.unique` grouping), preserving original row order within a group.
        let mut order: Vec<i64> = group_labels.to_vec();
        order.sort_unstable();
        order.dedup();
        let mut orig_groups: Vec<Vec<usize>> = vec![Vec::new(); order.len()];
        for (i, &lab) in group_labels.iter().enumerate() {
            let pos = order.binary_search(&lab).unwrap();
            orig_groups[pos].push(i);
        }

        // Build the expanded design row-block by row-block, walking clusters in
        // order so that expanded rows for one cluster are contiguous.
        let width = match kind {
            Kind::Nominal => ncut * p,
            Kind::Ordinal => ncut + p,
        };
        let nrows = ncut * n;
        let mut exog_out = Array2::<f64>::zeros((nrows, width));
        let mut endog_out = Array1::<f64>::zeros(nrows);
        let mut groups: Vec<Vec<usize>> = Vec::with_capacity(order.len());
        let mut group_nobs: Vec<usize> = Vec::with_capacity(order.len());

        let mut jrow = 0usize;
        for og in &orig_groups {
            let mut rows: Vec<usize> = Vec::with_capacity(og.len() * ncut);
            for &i in og {
                let yval = endog[i];
                for (cix, &cut) in cuts.iter().enumerate() {
                    match kind {
                        Kind::Ordinal => {
                            // Per-cut intercepts then the original covariates.
                            exog_out[[jrow, cix]] = 1.0;
                            for c in 0..p {
                                exog_out[[jrow, ncut + c]] = exog[[i, c]];
                            }
                            endog_out[jrow] = if yval > cut { 1.0 } else { 0.0 };
                        }
                        Kind::Nominal => {
                            // kron(e_cix, x_i): block `cix` of width p.
                            let base = cix * p;
                            for c in 0..p {
                                exog_out[[jrow, base + c]] = exog[[i, c]];
                            }
                            endog_out[jrow] = if yval == cut { 1.0 } else { 0.0 };
                        }
                    }
                    rows.push(jrow);
                    jrow += 1;
                }
            }
            group_nobs.push(og.len());
            groups.push(rows);
        }

        Ok(CategoricalGee {
            kind,
            cov,
            ncut,
            nparam,
            exog: exog_out,
            endog: endog_out,
            groups,
            group_nobs,
            nrows,
            maxiter: 300,
            ctol: 1e-10,
        })
    }

    /// Linear predictor for a cluster's expanded rows.
    fn lin_pred(&self, idx: &[usize], params: &Array1<f64>) -> Array1<f64> {
        let mut lpr = Array1::<f64>::zeros(idx.len());
        for (k, &r) in idx.iter().enumerate() {
            let mut s = 0.0;
            for j in 0..self.nparam {
                s += self.exog[[r, j]] * params[j];
            }
            lpr[k] = s;
        }
        lpr
    }

    /// Mean (expected indicator) for a cluster given its linear predictor.
    ///
    /// For ordinal models the logit link is applied per row; for nominal
    /// models the `ncut` indicators of each original observation are coupled
    /// through the shared multinomial normalizer.
    fn mean(&self, lpr: &Array1<f64>) -> Array1<f64> {
        match self.kind {
            Kind::Ordinal => lpr.mapv(|e| 1.0 / (1.0 + (-e).exp())),
            Kind::Nominal => {
                let mut mu = Array1::<f64>::zeros(lpr.len());
                let nobs = lpr.len() / self.ncut;
                for o in 0..nobs {
                    let base = o * self.ncut;
                    let mut denom = 1.0;
                    for k in 0..self.ncut {
                        denom += lpr[base + k].exp();
                    }
                    for k in 0..self.ncut {
                        mu[base + k] = lpr[base + k].exp() / denom;
                    }
                }
                mu
            }
        }
    }

    /// Mean-structure derivative `D = ∂μ/∂β` for a cluster's expanded rows.
    ///
    /// Both geometries use the same *row-wise* derivative
    /// `D[r,j] = μ_r (1 − μ_r) · exog[r,j]`.  For ordinal models this is the
    /// logit inverse-link derivative; for nominal models the reference applies
    /// the identical row-wise form (the multinomial coupling enters only the
    /// mean `μ` and the working covariance, not this Jacobian), so we match it.
    fn mean_deriv(&self, idx: &[usize], mu: &Array1<f64>) -> Array2<f64> {
        let m = idx.len();
        let mut d = Array2::<f64>::zeros((m, self.nparam));
        for (k, &r) in idx.iter().enumerate() {
            let idl = mu[k] * (1.0 - mu[k]);
            for j in 0..self.nparam {
                d[[k, j]] = self.exog[[r, j]] * idl;
            }
        }
        d
    }

    /// The working covariance matrix `V` for a cluster, given the expected
    /// indicators `mu` and the current global odds ratio `dep`.
    ///
    /// `V` is block structured by original observation: the within-observation
    /// block is deterministic, while between-observation blocks are zero for
    /// [`CategoricalCov::Independence`] or filled from the global odds-ratio
    /// joint-probability formula otherwise.
    fn working_cov(&self, gi: usize, mu: &Array1<f64>, dep: f64) -> Array2<f64> {
        let m = mu.len();
        let nobs = self.group_nobs[gi];
        let mut v = Array2::<f64>::zeros((m, m));

        if self.cov == CategoricalCov::GlobalOddsRatio {
            // Full E[YY'] from the global odds ratio, then subtract the mean
            // outer product; the within-observation blocks are overwritten
            // below with their deterministic values.
            let eyy = self.get_eyy(mu, dep);
            for a in 0..m {
                for b in 0..m {
                    v[[a, b]] = eyy[[a, b]] - mu[a] * mu[b];
                }
            }
        }

        // Within-observation blocks (size ncut), deterministic for both covs.
        for o in 0..nobs {
            let base = o * self.ncut;
            for a in 0..self.ncut {
                for b in 0..self.ncut {
                    let ea = mu[base + a];
                    let eb = mu[base + b];
                    let val = match self.kind {
                        Kind::Ordinal => ea.min(eb) - ea * eb,
                        Kind::Nominal => {
                            if a == b {
                                ea - ea * ea
                            } else {
                                -ea * eb
                            }
                        }
                    };
                    v[[base + a, base + b]] = val;
                }
            }
        }
        v
    }

    /// `E[YY']` under the global odds-ratio model for a cluster, before any
    /// within-observation correction (handled by [`Self::working_cov`]).
    fn get_eyy(&self, mu: &Array1<f64>, dep: f64) -> Array2<f64> {
        let m = mu.len();
        let mut eyy = Array2::<f64>::zeros((m, m));
        if dep == 1.0 {
            for a in 0..m {
                for b in 0..m {
                    eyy[[a, b]] = mu[a] * mu[b];
                }
            }
            return eyy;
        }
        let or = dep;
        for a in 0..m {
            for b in 0..m {
                let psum = mu[a] + mu[b];
                let pprod = mu[a] * mu[b];
                let pfac =
                    ((1.0 + psum * (or - 1.0)).powi(2) + 4.0 * or * (1.0 - or) * pprod).sqrt();
                eyy[[a, b]] = (1.0 + psum * (or - 1.0) - pfac) / (2.0 * (or - 1.0));
            }
        }
        eyy
    }

    /// One Fisher-scoring update of the mean parameters; returns the update and
    /// the current score (before the update) for the convergence test.
    fn update_mean_params(
        &self,
        params: &Array1<f64>,
        dep: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let (bmat, _, score) = self.accumulate(params, dep)?;
        let update = solve(&bmat, &score)?;
        Ok((update, score))
    }

    /// Accumulate the bread `B = Σ DᵀV⁻¹D`, the sandwich center
    /// `C = Σ (DᵀV⁻¹r)(DᵀV⁻¹r)ᵀ`, and the score `Σ DᵀV⁻¹r`.
    fn accumulate(
        &self,
        params: &Array1<f64>,
        dep: f64,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>)> {
        let p = self.nparam;
        let mut bmat = Array2::<f64>::zeros((p, p));
        let mut cmat = Array2::<f64>::zeros((p, p));
        let mut score = Array1::<f64>::zeros(p);

        for (gi, idx) in self.groups.iter().enumerate() {
            if idx.is_empty() {
                continue;
            }
            let lpr = self.lin_pred(idx, params);
            let mu = self.mean(&lpr);
            let resid: Array1<f64> = idx
                .iter()
                .zip(mu.iter())
                .map(|(&r, m)| self.endog[r] - m)
                .collect();
            let dmat = self.mean_deriv(idx, &mu);
            let vmat = self.working_cov(gi, &mu, dep);

            let vinv_d = solve_mat(&vmat, &dmat)?;
            let vinv_r = solve(&vmat, &resid)?;

            bmat += &dmat.t().dot(&vinv_d);
            let dvinv_resid = dmat.t().dot(&vinv_r);
            score += &dvinv_resid;
            for a in 0..p {
                for b in 0..p {
                    cmat[[a, b]] += dvinv_resid[a] * dvinv_resid[b];
                }
            }
        }
        Ok((bmat, cmat, score))
    }

    /// Crude (marginal) global odds ratio: pool every between-observation
    /// cut-point pair into `2×2` tables of *observed* indicators and take the
    /// inverse-variance-weighted (pooled) odds ratio.
    fn observed_crude_oddsratio(&self) -> f64 {
        // tables[(k2,k1)] for 0 <= k2 <= k1 < ncut.
        let mut tables = self.empty_tables();
        for (gi, idx) in self.groups.iter().enumerate() {
            let nobs = self.group_nobs[gi];
            let y: Array1<f64> = idx.iter().map(|&r| self.endog[r]).collect();
            self.accumulate_tables(&mut tables, &y, &y, nobs);
        }
        pooled_odds_ratio(&tables)
    }

    /// Allocate the per-cut-pair `2×2` contingency tables (lower triangle).
    fn empty_tables(&self) -> Vec<[[f64; 2]; 2]> {
        let mut n = 0;
        for k1 in 0..self.ncut {
            n += k1 + 1;
        }
        vec![[[0.0; 2]; 2]; n]
    }

    /// Linear index of cut-pair `(k2, k1)` with `k2 <= k1` in the lower-triangle
    /// table list (matching the construction order in [`Self::empty_tables`]).
    fn pair_index(&self, k2: usize, k1: usize) -> usize {
        // pairs ordered by k1 ascending, then k2 in 0..=k1.
        let mut base = 0;
        for k in 0..k1 {
            base += k + 1;
        }
        base + k2
    }

    /// Add a cluster's between-observation contributions to the pooled tables.
    ///
    /// `eyy11[a,b]` is the joint probability (or observed product) that both
    /// indicators are 1; `ey_a`/`ey_b` are the marginal expectations used to
    /// derive the 10/01/00 cells.  For the *observed* crude ratio pass both
    /// `eyy` and the marginals are the realized 0/1 indicators.
    fn accumulate_tables(
        &self,
        tables: &mut [[[f64; 2]; 2]],
        ya: &Array1<f64>,
        yb: &Array1<f64>,
        nobs: usize,
    ) {
        // Between-subject lower-triangle pairs (i1 > i2).
        for i1 in 0..nobs {
            for i2 in 0..i1 {
                for k1 in 0..self.ncut {
                    for k2 in 0..=k1 {
                        let a = i1 * self.ncut + k1;
                        let b = i2 * self.ncut + k2;
                        let p11 = ya[a] * yb[b];
                        let p10 = ya[a] * (1.0 - yb[b]);
                        let p01 = (1.0 - ya[a]) * yb[b];
                        let p00 = (1.0 - ya[a]) * (1.0 - yb[b]);
                        let t = &mut tables[self.pair_index(k2, k1)];
                        t[1][1] += p11;
                        t[1][0] += p10;
                        t[0][1] += p01;
                        t[0][0] += p00;
                    }
                }
            }
        }
    }

    /// One global-odds-ratio update: rebuild the pooled tables from the current
    /// model-implied joint probabilities and rescale `dep` by
    /// `crude_or / expected_or`.
    fn update_dep(&self, params: &Array1<f64>, dep: f64, crude_or: f64) -> f64 {
        // No between-observation pairs anywhere => nothing to update.
        if self.group_nobs.iter().all(|&m| m <= 1) {
            return dep;
        }
        let mut tables = self.empty_tables();
        for (gi, idx) in self.groups.iter().enumerate() {
            let nobs = self.group_nobs[gi];
            if nobs <= 1 {
                continue;
            }
            let lpr = self.lin_pred(idx, params);
            let mu = self.mean(&lpr);
            let eyy = self.get_eyy(&mu, dep);
            // Build expectation-based 2x2 cells directly.
            for i1 in 0..nobs {
                for i2 in 0..i1 {
                    for k1 in 0..self.ncut {
                        for k2 in 0..=k1 {
                            let a = i1 * self.ncut + k1;
                            let b = i2 * self.ncut + k2;
                            let e11 = eyy[[a, b]];
                            let e10 = mu[a] - e11;
                            let e01 = mu[b] - e11;
                            let e00 = 1.0 - (e11 + e10 + e01);
                            let t = &mut tables[self.pair_index(k2, k1)];
                            t[1][1] += e11;
                            t[1][0] += e10;
                            t[0][1] += e01;
                            t[0][0] += e00;
                        }
                    }
                }
            }
        }
        let cor_expval = pooled_odds_ratio(&tables);
        let new_dep = dep * crude_or / cor_expval;
        if new_dep.is_finite() {
            new_dep
        } else {
            1.0
        }
    }

    /// Starting parameters: the GLM (binomial-logit) fit of the expanded
    /// indicators on the expanded design.  This coincides with the reference's
    /// Independence-GEE starting fit for the mean parameters.
    fn starting_params(&self) -> Result<Array1<f64>> {
        let glm = Glm::with_link(
            self.endog.clone(),
            self.exog.clone(),
            Family::Binomial,
            Link::Logit,
        )?
        .fit()?;
        Ok(glm.params)
    }

    /// Fit the model.
    fn fit(&self) -> Result<CategoricalGeeResults> {
        let mut params = self.starting_params()?;

        let update_dep =
            self.cov == CategoricalCov::GlobalOddsRatio && self.group_nobs.iter().any(|&m| m > 1);
        // The crude OR is fixed across iterations and seeds `dep`.
        let crude_or = if update_dep {
            self.observed_crude_oddsratio()
        } else {
            1.0
        };
        let mut dep = if update_dep { crude_or } else { 1.0 };

        let mut score_norm = f64::INFINITY;
        let mut num_assoc_updates = 0usize;
        let mut converged = false;

        for _ in 0..self.maxiter {
            let (update, score) = self.update_mean_params(&params, dep)?;
            params = &params + &update;
            score_norm = score.iter().map(|s| s * s).sum::<f64>().sqrt();

            if score_norm < self.ctol && (num_assoc_updates > 0 || !update_dep) {
                converged = true;
                break;
            }

            if update_dep {
                dep = self.update_dep(&params, dep, crude_or);
                num_assoc_updates += 1;
            } else {
                converged = score_norm < self.ctol;
                if converged {
                    break;
                }
            }
        }

        // Covariances.  Note categorical GEE has unit scale (binary variance),
        // so `cov_naive = B⁻¹` and `cov_robust = B⁻¹ C B⁻¹`.
        let (bmat, cmat, _) = self.accumulate(&params, dep)?;
        let bmati = inv(&bmat)?;
        let cov_naive = bmati.clone();
        let cov_robust = bmati.dot(&cmat).dot(&bmati);

        let p = self.nparam;
        let bse: Array1<f64> = (0..p).map(|j| cov_robust[[j, j]].sqrt()).collect();
        let bse_naive: Array1<f64> = (0..p).map(|j| cov_naive[[j, j]].sqrt()).collect();
        let tvalues: Array1<f64> = params.iter().zip(bse.iter()).map(|(b, s)| b / s).collect();
        let pvalues: Array1<f64> = tvalues.mapv(|t| 2.0 * norm_sf(t.abs()));

        // Fitted values in expanded-row order (cluster order, as stored).
        let mut fitted = Array1::<f64>::zeros(self.nrows);
        for idx in &self.groups {
            let lpr = self.lin_pred(idx, &params);
            let mu = self.mean(&lpr);
            for (k, &r) in idx.iter().enumerate() {
                fitted[r] = mu[k];
            }
        }

        Ok(CategoricalGeeResults {
            params,
            bse,
            bse_naive,
            tvalues,
            pvalues,
            cov_robust,
            cov_naive,
            dep_params: if update_dep { dep } else { 0.0 },
            scale: 1.0,
            fittedvalues: fitted,
            ncut: self.ncut,
            score_norm,
            converged,
        })
    }
}

/// Fitted results of a categorical ([`NominalGee`] / [`OrdinalGee`]) GEE.
#[derive(Clone, Debug)]
pub struct CategoricalGeeResults {
    /// Estimated mean-structure parameters.  For nominal models the layout is
    /// block-by-cut (`[cut₀ coefs, cut₁ coefs, …]`); for ordinal models it is
    /// `[intercept₀, …, interceptₙ₋₁, covariate coefs]`.
    pub params: Array1<f64>,
    /// Robust (sandwich) standard errors.
    pub bse: Array1<f64>,
    /// Naive (model-based) standard errors.
    pub bse_naive: Array1<f64>,
    /// `params / bse` (robust).
    pub tvalues: Array1<f64>,
    /// Two-sided normal p-values from the robust z-statistics.
    pub pvalues: Array1<f64>,
    /// Robust sandwich covariance of `params`.
    pub cov_robust: Array2<f64>,
    /// Naive model-based covariance of `params`.
    pub cov_naive: Array2<f64>,
    /// Estimated global odds ratio; `0` for the independence working
    /// association.
    pub dep_params: f64,
    /// Dispersion/scale (always `1` for the binary indicator model).
    pub scale: f64,
    /// Fitted indicator means in expanded-row order.
    pub fittedvalues: Array1<f64>,
    /// Number of cut points (`K − 1`).
    pub ncut: usize,
    /// L2 norm of the score equations at convergence.
    pub score_norm: f64,
    /// Whether the score-norm tolerance was met.
    pub converged: bool,
}

/// Inverse-variance-weighted pooled odds ratio of a list of `2×2` tables.
fn pooled_odds_ratio(tables: &[[[f64; 2]; 2]]) -> f64 {
    if tables.is_empty() {
        return 1.0;
    }
    let mut log_or = Vec::with_capacity(tables.len());
    let mut var = Vec::with_capacity(tables.len());
    for t in tables {
        let lor = t[1][1].ln() + t[0][0].ln() - t[0][1].ln() - t[1][0].ln();
        log_or.push(lor);
        var.push(1.0 / t[1][1] + 1.0 / t[0][0] + 1.0 / t[0][1] + 1.0 / t[1][0]);
    }
    let wts: Vec<f64> = var.iter().map(|v| 1.0 / v).collect();
    let wtsum: f64 = wts.iter().sum();
    let log_pooled: f64 = wts
        .iter()
        .zip(log_or.iter())
        .map(|(w, e)| (w / wtsum) * e)
        .sum();
    log_pooled.exp()
}

/// Solve `A X = B` column by column for a matrix right-hand side.
fn solve_mat(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, k) = b.dim();
    let mut out = Array2::<f64>::zeros((m, k));
    for j in 0..k {
        let col = b.column(j).to_owned();
        let sol = solve(a, &col)?;
        for i in 0..m {
            out[[i, j]] = sol[i];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ordinal_expands_indicators() {
        // 3 levels {0,1,2} => ncut=2, indicators I(y>0), I(y>1).
        let y = array![0.0, 1.0, 2.0];
        let x = array![[0.5], [1.0], [-0.5]];
        let groups = [0i64, 0, 1];
        let m = CategoricalGee::new(Kind::Ordinal, y, x, &groups, CategoricalCov::Independence)
            .unwrap();
        assert_eq!(m.ncut, 2);
        // y=0 => (0,0); y=1 => (1,0); y=2 => (1,1).
        assert_eq!(m.endog.to_vec(), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        // Intercept columns are the first ncut columns.
        assert_eq!(m.nparam, 2 + 1);
    }

    #[test]
    fn nominal_expands_indicators() {
        // 3 levels => ncut=2, indicators I(y==0), I(y==1).
        let y = array![0.0, 1.0, 2.0];
        let x = array![[1.0, 0.5], [1.0, 1.0], [1.0, -0.5]];
        let groups = [0i64, 0, 1];
        let m = CategoricalGee::new(Kind::Nominal, y, x, &groups, CategoricalCov::Independence)
            .unwrap();
        assert_eq!(m.ncut, 2);
        assert_eq!(m.nparam, 2 * 2);
        // y=0 => (1,0); y=1 => (0,1); y=2 => (0,0).
        assert_eq!(m.endog.to_vec(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn nominal_mean_matches_softmax() {
        // The multinomial-logit mean couples the ncut indicators of one
        // observation: μ_k = e^{η_k} / (1 + Σ_j e^{η_j}), summing to < 1.
        let y = array![0.0, 1.0, 2.0];
        let x = array![[1.0], [1.0], [1.0]];
        let groups = [0i64, 0, 0];
        let m = CategoricalGee::new(Kind::Nominal, y, x, &groups, CategoricalCov::Independence)
            .unwrap();
        // One observation, ncut=2 rows; η = (0.5, -0.3).
        let lpr = array![0.5_f64, -0.3];
        let mu = m.mean(&lpr);
        let denom = 1.0 + 0.5_f64.exp() + (-0.3_f64).exp();
        assert!((mu[0] - 0.5_f64.exp() / denom).abs() < 1e-12);
        assert!((mu[1] - (-0.3_f64).exp() / denom).abs() < 1e-12);
        assert!(mu[0] + mu[1] < 1.0);
    }

    #[test]
    fn ordinal_mean_is_logit() {
        let y = array![0.0, 1.0, 2.0];
        let x = array![[0.5], [1.0], [-0.5]];
        let groups = [0i64, 0, 1];
        let m = CategoricalGee::new(Kind::Ordinal, y, x, &groups, CategoricalCov::Independence)
            .unwrap();
        let lpr = array![0.7_f64, -1.2];
        let mu = m.mean(&lpr);
        assert!((mu[0] - 1.0 / (1.0 + (-0.7_f64).exp())).abs() < 1e-12);
        assert!((mu[1] - 1.0 / (1.0 + 1.2_f64.exp())).abs() < 1e-12);
    }
}
