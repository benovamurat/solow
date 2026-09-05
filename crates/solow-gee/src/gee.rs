//! Generalized estimating equations (GEE).
//!
//! GEE extends generalized linear models to clustered / longitudinal data by
//! positing a *working* correlation structure within each cluster.  The mean
//! parameters are estimated by Fisher scoring on the estimating equations,
//! while the correlation (association) parameter is re-estimated between mean
//! updates.  Inference uses the robust *sandwich* covariance, which remains
//! valid even when the working correlation is misspecified; a model-based
//! ("naive") covariance is also reported.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_distributions::norm_sf;
use solow_glm::{Family, Glm, Link};
use solow_linalg::{inv, solve};

/// The within-cluster working correlation structure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CovStruct {
    /// Observations within a cluster are treated as uncorrelated (the working
    /// correlation is the identity).  GEE then reduces to a GLM for the point
    /// estimates, but inference still uses the cluster-robust sandwich.
    Independence,
    /// A single common correlation `ρ` between every pair of observations in a
    /// cluster (compound symmetry).
    Exchangeable,
}

/// A GEE model awaiting estimation.
#[derive(Clone, Debug)]
pub struct Gee {
    endog: Array1<f64>,
    exog: Array2<f64>,
    /// Row indices for each cluster, in order of first appearance of the group
    /// label in the input data.
    groups: Vec<Vec<usize>>,
    family: Family,
    link: Link,
    cov_struct: CovStruct,
    /// Degrees of freedom subtracted when normalizing the scale (defaults to
    /// the number of mean parameters, matching the reference).
    ddof_scale: f64,
    maxiter: usize,
    /// Convergence tolerance on the L2 norm of the score equations.
    ctol: f64,
}

impl Gee {
    /// Build a GEE with the family's canonical link.
    ///
    /// `group_labels` assigns each observation to a cluster; rows that share a
    /// label form one cluster.  Clusters are ordered by first appearance.
    pub fn new(
        endog: Array1<f64>,
        exog: Array2<f64>,
        group_labels: &[i64],
        family: Family,
        cov_struct: CovStruct,
    ) -> Result<Self> {
        let link = family.default_link();
        Self::with_link(endog, exog, group_labels, family, link, cov_struct)
    }

    /// Build a GEE with an explicit link.
    pub fn with_link(
        endog: Array1<f64>,
        exog: Array2<f64>,
        group_labels: &[i64],
        family: Family,
        link: Link,
        cov_struct: CovStruct,
    ) -> Result<Self> {
        let n = endog.len();
        if n != exog.nrows() {
            return Err(Error::Shape("endog length != exog rows".into()));
        }
        if group_labels.len() != n {
            return Err(Error::Shape("group_labels length != endog length".into()));
        }
        let groups = group_indices(group_labels);
        let p = exog.ncols();
        Ok(Gee {
            endog,
            exog,
            groups,
            family,
            link,
            cov_struct,
            ddof_scale: p as f64,
            maxiter: 300,
            ctol: 1e-10,
        })
    }

    /// Set the maximum number of Fisher-scoring iterations.
    pub fn maxiter(mut self, m: usize) -> Self {
        self.maxiter = m;
        self
    }

    /// Set the convergence tolerance on the score-equation norm.
    pub fn ctol(mut self, t: f64) -> Self {
        self.ctol = t;
        self
    }

    /// Number of observations.
    fn nobs(&self) -> usize {
        self.endog.len()
    }

    /// `dμ/dη` at the given linear predictor (the inverse-link derivative).
    fn inverse_deriv(&self, eta: f64) -> f64 {
        let mu = self.link.inverse(eta);
        1.0 / self.link.deriv(mu)
    }

    /// Group-wise expected values `μ` and linear predictors `η` for `params`.
    fn cached_means(&self, params: &Array1<f64>) -> Vec<(Array1<f64>, Array1<f64>)> {
        self.groups
            .iter()
            .map(|idx| {
                let m = idx.len();
                let mut eta = Array1::<f64>::zeros(m);
                let mut mu = Array1::<f64>::zeros(m);
                for (k, &i) in idx.iter().enumerate() {
                    let mut lp = 0.0;
                    for j in 0..self.exog.ncols() {
                        lp += self.exog[[i, j]] * params[j];
                    }
                    eta[k] = lp;
                    mu[k] = self.link.inverse(lp);
                }
                (mu, eta)
            })
            .collect()
    }

    /// The mean-structure derivative `D = ∂μ/∂β` for a cluster:
    /// row `k` is `exog[k] · (dμ/dη)`.
    fn mean_deriv(&self, idx: &[usize], eta: &Array1<f64>) -> Array2<f64> {
        let p = self.exog.ncols();
        let mut dmat = Array2::<f64>::zeros((idx.len(), p));
        for (k, &i) in idx.iter().enumerate() {
            let idl = self.inverse_deriv(eta[k]);
            for j in 0..p {
                dmat[[k, j]] = self.exog[[i, j]] * idl;
            }
        }
        dmat
    }

    /// Working covariance `V = S · R · S` for a cluster, where `S = diag(sdev)`
    /// and `R` is the working correlation matrix implied by `cov_struct`.
    fn working_cov(&self, sdev: &Array1<f64>, dep: f64) -> Array2<f64> {
        let m = sdev.len();
        let mut v = Array2::<f64>::zeros((m, m));
        for a in 0..m {
            for b in 0..m {
                let r = if a == b {
                    1.0
                } else {
                    match self.cov_struct {
                        CovStruct::Independence => 0.0,
                        CovStruct::Exchangeable => dep,
                    }
                };
                v[[a, b]] = r * sdev[a] * sdev[b];
            }
        }
        v
    }

    /// One Fisher-scoring update of the mean parameters.
    ///
    /// Returns `(update, score)` where `params + update` is the next iterate and
    /// `score = Σ Dᵀ V⁻¹ (y − μ)` is the estimating-equation value *before* the
    /// update (used for the convergence test).
    fn update_mean_params(
        &self,
        cached: &[(Array1<f64>, Array1<f64>)],
        dep: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let p = self.exog.ncols();
        let mut bmat = Array2::<f64>::zeros((p, p));
        let mut score = Array1::<f64>::zeros(p);
        for (gi, idx) in self.groups.iter().enumerate() {
            let (mu, eta) = &cached[gi];
            let resid: Array1<f64> = self
                .endog_group(idx)
                .iter()
                .zip(mu.iter())
                .map(|(y, m)| y - m)
                .collect();
            let dmat = self.mean_deriv(idx, eta);
            let sdev = mu.mapv(|m| self.family.variance(m).sqrt());
            let vmat = self.working_cov(&sdev, dep);

            // V⁻¹ D and V⁻¹ r via a single linear solve.
            let vinv_d = solve_spd(&vmat, &dmat)?;
            let vinv_r = solve(&vmat, &resid)?;

            bmat += &dmat.t().dot(&vinv_d);
            score += &dmat.t().dot(&vinv_r);
        }
        let update = solve(&bmat, &score)?;
        Ok((update, score))
    }

    /// Endog values for a cluster.
    fn endog_group(&self, idx: &[usize]) -> Array1<f64> {
        idx.iter().map(|&i| self.endog[i]).collect()
    }

    /// Update the exchangeable correlation parameter (compound symmetry) from
    /// the current standardized residuals, matching the reference normalization.
    fn update_dep(&self, cached: &[(Array1<f64>, Array1<f64>)]) -> f64 {
        if self.cov_struct == CovStruct::Independence {
            return 0.0;
        }
        let nobs = self.nobs() as f64;
        let ddof = self.ddof_scale;
        let mut residsq_sum = 0.0;
        let mut scale = 0.0;
        let mut fsum1 = 0.0;
        let mut fsum2 = 0.0;
        let mut n_pairs = 0.0;
        for (gi, idx) in self.groups.iter().enumerate() {
            let (mu, _) = &cached[gi];
            let y = self.endog_group(idx);
            let resid: Array1<f64> = y
                .iter()
                .zip(mu.iter())
                .map(|(yy, m)| (yy - m) / self.family.variance(*m).sqrt())
                .collect();
            let ssr: f64 = resid.iter().map(|r| r * r).sum();
            scale += ssr;
            fsum1 += idx.len() as f64;
            let rsum: f64 = resid.sum();
            residsq_sum += (rsum * rsum - ssr) / 2.0;
            let ngrp = resid.len() as f64;
            let npr = 0.5 * ngrp * (ngrp - 1.0);
            fsum2 += npr;
            n_pairs += npr;
        }
        if n_pairs == 0.0 {
            // No within-cluster pairs (all singletons): association undefined.
            return 0.0;
        }
        scale /= fsum1 * (nobs - ddof) / nobs;
        residsq_sum /= scale;
        residsq_sum / (fsum2 * (n_pairs - ddof) / n_pairs)
    }

    /// Estimate the dispersion/scale. Fixed at 1 for Poisson/Binomial.
    fn estimate_scale(&self, cached: &[(Array1<f64>, Array1<f64>)]) -> f64 {
        if self.family.fixed_scale() {
            return 1.0;
        }
        let nobs = self.nobs() as f64;
        let ddof = self.ddof_scale;
        let mut scale = 0.0;
        let mut fsum = 0.0;
        for (gi, idx) in self.groups.iter().enumerate() {
            let (mu, _) = &cached[gi];
            let y = self.endog_group(idx);
            for (yy, m) in y.iter().zip(mu.iter()) {
                let r = (yy - m) / self.family.variance(*m).sqrt();
                scale += r * r;
            }
            fsum += idx.len() as f64;
        }
        scale /= fsum * (nobs - ddof) / nobs;
        scale
    }

    /// Robust (sandwich) and naive (model-based) covariance matrices, and the
    /// center matrix of the sandwich.
    fn covmat(
        &self,
        cached: &[(Array1<f64>, Array1<f64>)],
        dep: f64,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let p = self.exog.ncols();
        let mut bmat = Array2::<f64>::zeros((p, p));
        let mut cmat = Array2::<f64>::zeros((p, p));
        for (gi, idx) in self.groups.iter().enumerate() {
            let (mu, eta) = &cached[gi];
            let resid: Array1<f64> = self
                .endog_group(idx)
                .iter()
                .zip(mu.iter())
                .map(|(y, m)| y - m)
                .collect();
            let dmat = self.mean_deriv(idx, eta);
            let sdev = mu.mapv(|m| self.family.variance(m).sqrt());
            let vmat = self.working_cov(&sdev, dep);

            let vinv_d = solve_spd(&vmat, &dmat)?;
            let vinv_r = solve(&vmat, &resid)?;

            bmat += &dmat.t().dot(&vinv_d);
            let dvinv_resid = dmat.t().dot(&vinv_r);
            // Outer product of the per-cluster score contribution.
            for a in 0..p {
                for b in 0..p {
                    cmat[[a, b]] += dvinv_resid[a] * dvinv_resid[b];
                }
            }
        }
        let scale = self.estimate_scale(cached);
        let bmati = inv(&bmat)?;
        let cov_naive = &bmati * scale;
        let cov_robust = bmati.dot(&cmat).dot(&bmati);
        Ok((cov_robust, cov_naive))
    }

    /// Fit the model.
    pub fn fit(&self) -> Result<GeeResults> {
        let p = self.exog.ncols();

        // Starting values from a plain GLM fit (matching the reference).
        let glm = Glm::with_link(
            self.endog.clone(),
            self.exog.clone(),
            self.family,
            self.link,
        )?
        .fit()?;
        let mut params = glm.params.clone();

        let mut cached = self.cached_means(&params);
        let mut dep = 0.0;
        let mut score_norm = f64::INFINITY;
        let mut num_assoc_updates = 0usize;
        let mut converged = false;

        for _ in 0..self.maxiter {
            let (update, score) = self.update_mean_params(&cached, dep)?;
            params = &params + &update;
            cached = self.cached_means(&params);

            score_norm = score.iter().map(|s| s * s).sum::<f64>().sqrt();

            let update_dep = self.cov_struct != CovStruct::Independence;
            if score_norm < self.ctol && (num_assoc_updates > 0 || !update_dep) {
                converged = true;
                break;
            }

            if update_dep {
                dep = self.update_dep(&cached);
                num_assoc_updates += 1;
            } else {
                converged = score_norm < self.ctol;
                if converged {
                    break;
                }
            }
        }

        let (cov_robust, cov_naive) = self.covmat(&cached, dep)?;
        let scale = self.estimate_scale(&cached);

        let bse: Array1<f64> = (0..p).map(|j| cov_robust[[j, j]].sqrt()).collect();
        let bse_naive: Array1<f64> = (0..p).map(|j| cov_naive[[j, j]].sqrt()).collect();

        let tvalues: Array1<f64> = params.iter().zip(bse.iter()).map(|(b, s)| b / s).collect();
        let pvalues: Array1<f64> = tvalues.mapv(|t| 2.0 * norm_sf(t.abs()));

        let fitted: Array1<f64> = {
            let mut f = Array1::<f64>::zeros(self.nobs());
            for (gi, idx) in self.groups.iter().enumerate() {
                let (mu, _) = &cached[gi];
                for (k, &i) in idx.iter().enumerate() {
                    f[i] = mu[k];
                }
            }
            f
        };

        Ok(GeeResults {
            params,
            bse,
            bse_naive,
            tvalues,
            pvalues,
            cov_robust,
            cov_naive,
            dep_params: dep,
            scale,
            fittedvalues: fitted,
            score_norm,
            converged,
        })
    }
}

/// Fitted GEE results.
#[derive(Clone, Debug)]
pub struct GeeResults {
    /// Estimated mean-structure parameters `β`.
    pub params: Array1<f64>,
    /// Robust (sandwich) standard errors.
    pub bse: Array1<f64>,
    /// Naive (model-based) standard errors.
    pub bse_naive: Array1<f64>,
    /// `params / bse` (robust).
    pub tvalues: Array1<f64>,
    /// Two-sided normal p-values from the robust z-statistics.
    pub pvalues: Array1<f64>,
    /// Robust sandwich covariance matrix of `params`.
    pub cov_robust: Array2<f64>,
    /// Naive model-based covariance matrix of `params`.
    pub cov_naive: Array2<f64>,
    /// Estimated working-correlation (association) parameter; `0` for
    /// independence.
    pub dep_params: f64,
    /// Estimated dispersion/scale (`1` for Poisson/Binomial).
    pub scale: f64,
    /// Fitted means `μ` in input row order.
    pub fittedvalues: Array1<f64>,
    /// L2 norm of the score equations at convergence.
    pub score_norm: f64,
    /// Whether the score-norm tolerance was met.
    pub converged: bool,
}

/// Group input rows by label, preserving first-appearance order of labels.
fn group_indices(labels: &[i64]) -> Vec<Vec<usize>> {
    let mut order: Vec<i64> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, &lab) in labels.iter().enumerate() {
        match order.iter().position(|&l| l == lab) {
            Some(pos) => groups[pos].push(i),
            None => {
                order.push(lab);
                groups.push(vec![i]);
            }
        }
    }
    groups
}

/// Solve `A X = B` for a matrix right-hand side `B`, column by column.
fn solve_spd(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
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
    fn group_indices_preserves_order() {
        let g = group_indices(&[5, 5, 2, 2, 5]);
        assert_eq!(g, vec![vec![0, 1, 4], vec![2, 3]]);
    }

    #[test]
    fn independence_poisson_matches_glm_params() {
        // With Independence working correlation, GEE point estimates equal the
        // GLM (Poisson) MLE.
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
        ];
        let y = array![1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let groups = [0i64, 0, 1, 1, 2, 2];
        let gee = Gee::new(
            y.clone(),
            x.clone(),
            &groups,
            Family::Poisson,
            CovStruct::Independence,
        )
        .unwrap();
        let res = gee.fit().unwrap();
        let glm = Glm::new(y, x, Family::Poisson).unwrap().fit().unwrap();
        for j in 0..2 {
            assert!((res.params[j] - glm.params[j]).abs() < 1e-8);
        }
        assert!(res.converged);
        assert_eq!(res.dep_params, 0.0);
    }

    #[test]
    fn exchangeable_reduces_to_independence_when_no_within_corr() {
        // When every cluster is a singleton, the exchangeable association
        // parameter is undefined (no pairs) and falls back to 0, so the
        // exchangeable fit coincides with the independence fit.
        let x = array![
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.2],
            [1.0, -0.2],
        ];
        let y = array![3.0, 1.0, 4.0, 1.0, 5.0, 2.0];
        let groups = [0i64, 1, 2, 3, 4, 5]; // all singletons
        let exch = Gee::new(
            y.clone(),
            x.clone(),
            &groups,
            Family::Poisson,
            CovStruct::Exchangeable,
        )
        .unwrap()
        .fit()
        .unwrap();
        let indep = Gee::new(y, x, &groups, Family::Poisson, CovStruct::Independence)
            .unwrap()
            .fit()
            .unwrap();
        assert!(exch.converged && indep.converged);
        for j in 0..exch.params.len() {
            assert!((exch.params[j] - indep.params[j]).abs() < 1e-9);
        }
    }
}
