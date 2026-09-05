//! Deterministic linear mediation effects.
//!
//! [`mediation_effects`] computes the point estimates of a causal mediation
//! analysis for the linear-outcome / linear-mediator case with no interaction
//! involving the mediator. These are exactly the deterministic limit of the
//! reference `mediation.Mediation.fit` Monte-Carlo procedure: replacing each
//! simulated parameter draw by its point estimate and each simulated potential
//! mediator by its conditional mean (valid because the linear outcome predictor
//! is affine in the mediator) collapses the average causal mediated effect
//! (ACME), average direct effect (ADE), total effect and proportion mediated to
//! closed-form quantities.
//!
//! The reference also reports simulation-based confidence intervals and
//! p-values; those are RNG-driven and out of scope here (see crate notes).

use ndarray::{Array1, Array2};
use solow_regression::LinearModel;

/// Deterministic point estimates from a linear mediation analysis.
///
/// Field names follow the reference `MediationResults`: `_ctrl` and `_tx` are
/// the effects with the *other* treatment arm held at control (0) or treated
/// (1); `_avg` averages the two. In the linear no-interaction case the control
/// and treated variants coincide.
#[derive(Debug, Clone, Copy)]
pub struct MediationResults {
    /// Average causal mediated (indirect) effect, control arm.
    pub acme_ctrl: f64,
    /// Average causal mediated (indirect) effect, treated arm.
    pub acme_tx: f64,
    /// Average direct effect, control arm.
    pub ade_ctrl: f64,
    /// Average direct effect, treated arm.
    pub ade_tx: f64,
    /// Total effect, `(ACME_ctrl + ACME_tx + ADE_ctrl + ADE_tx) / 2`.
    pub total_effect: f64,
    /// Proportion mediated, control arm.
    pub prop_med_ctrl: f64,
    /// Proportion mediated, treated arm.
    pub prop_med_tx: f64,
    /// Average proportion mediated.
    pub prop_med_avg: f64,
    /// Average causal mediated (indirect) effect.
    pub acme_avg: f64,
    /// Average direct effect.
    pub ade_avg: f64,
}

/// Specification of a linear mediation analysis (no formulas).
///
/// `outcome_endog` / `outcome_exog` define the outcome regression `Y ~ X_o`
/// (which must include both the mediator and the exposure as columns), and
/// `mediator_endog` / `mediator_exog` the mediator regression `M ~ X_m`. The
/// `*_pos` fields give the column positions of the exposure in each design and
/// of the mediator in the outcome design, exactly as in the reference's
/// positional API (`Mediation(outcome, mediator, [exp_pos_outcome,
/// exp_pos_mediator], med_pos_outcome)`).
#[derive(Debug, Clone)]
pub struct Mediation {
    /// Outcome model response vector.
    pub outcome_endog: Array1<f64>,
    /// Outcome model design matrix (includes exposure and mediator columns).
    pub outcome_exog: Array2<f64>,
    /// Mediator model response vector.
    pub mediator_endog: Array1<f64>,
    /// Mediator model design matrix (includes the exposure column).
    pub mediator_exog: Array2<f64>,
    /// Column position of the exposure in the outcome design.
    pub exp_pos_outcome: usize,
    /// Column position of the exposure in the mediator design.
    pub exp_pos_mediator: usize,
    /// Column position of the mediator in the outcome design.
    pub med_pos_outcome: usize,
}

impl Mediation {
    /// Mediator design with the exposure column set to `exposure`.
    fn mediator_exog_at(&self, exposure: f64) -> Array2<f64> {
        let mut m = self.mediator_exog.clone();
        m.column_mut(self.exp_pos_mediator).fill(exposure);
        m
    }

    /// Outcome design with exposure set to `exposure` and the mediator column
    /// set to the per-observation `mediator` values.
    fn outcome_exog_at(&self, exposure: f64, mediator: &Array1<f64>) -> Array2<f64> {
        let mut o = self.outcome_exog.clone();
        o.column_mut(self.exp_pos_outcome).fill(exposure);
        o.column_mut(self.med_pos_outcome).assign(mediator);
        o
    }

    /// Fit both linear models and return the deterministic mediation point
    /// estimates. Mirrors the deterministic limit of the reference
    /// `Mediation.fit` for linear outcome and mediator models.
    pub fn fit(&self) -> MediationResults {
        let beta_o = LinearModel::ols(self.outcome_endog.clone(), self.outcome_exog.clone())
            .expect("outcome OLS")
            .fit()
            .expect("outcome fit")
            .params;
        let beta_m = LinearModel::ols(self.mediator_endog.clone(), self.mediator_exog.clone())
            .expect("mediator OLS")
            .fit()
            .expect("mediator fit")
            .params;

        // potential_mediator[tm] = E[M | exposure = tm] (conditional mean).
        let pm: [Array1<f64>; 2] = [
            self.mediator_exog_at(0.0).dot(&beta_m),
            self.mediator_exog_at(1.0).dot(&beta_m),
        ];

        // predicted_outcomes[tm][te] = E[Y | mediator = pm[tm], exposure = te].
        let predict = |tm: usize, te: f64| self.outcome_exog_at(te, &pm[tm]).dot(&beta_o);
        let po: [[Array1<f64>; 2]; 2] = [
            [predict(0, 0.0), predict(0, 1.0)],
            [predict(1, 0.0), predict(1, 1.0)],
        ];

        let mean = |a: &Array1<f64>| a.sum() / a.len() as f64;

        // indirect_effects[t] = po[1][t] - po[0][t]; direct[t] = po[t][1] - po[t][0].
        let acme_ctrl = mean(&(&po[1][0] - &po[0][0]));
        let acme_tx = mean(&(&po[1][1] - &po[0][1]));
        let ade_ctrl = mean(&(&po[0][1] - &po[0][0]));
        let ade_tx = mean(&(&po[1][1] - &po[1][0]));

        let total_effect = (acme_ctrl + acme_tx + ade_ctrl + ade_tx) / 2.0;
        let prop_med_ctrl = acme_ctrl / total_effect;
        let prop_med_tx = acme_tx / total_effect;
        let prop_med_avg = (prop_med_ctrl + prop_med_tx) / 2.0;
        let acme_avg = (acme_ctrl + acme_tx) / 2.0;
        let ade_avg = (ade_ctrl + ade_tx) / 2.0;

        MediationResults {
            acme_ctrl,
            acme_tx,
            ade_ctrl,
            ade_tx,
            total_effect,
            prop_med_ctrl,
            prop_med_tx,
            prop_med_avg,
            acme_avg,
            ade_avg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn matches_baron_kenny() {
        // M is NOT collinear with [const, T, Z] (it carries an extra component
        // `noise`), so the outcome design is full rank and OLS recovers the
        // exact coefficients from the noiseless outcome. Then ACME = a*b and
        // ADE = c' exactly.
        let t = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let z = array![10.0, 12.0, 14.0, 9.0, 11.0, 13.0, 8.0, 15.0, 7.0, 16.0];
        let extra = array![0.3, -0.5, 0.1, 0.8, -0.2, 0.6, -0.7, 0.4, 0.2, -0.9];
        let n = t.len();
        let m: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + 0.8 * t[i] + 0.05 * z[i] + extra[i]));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 1.0 + 1.5 * m[i] + 0.7 * t[i] + 0.02 * z[i]));

        // outcome exog: [const, M, T, Z]; mediator exog: [const, T, Z].
        let ones = Array1::ones(n);
        let mut oe = Array2::zeros((n, 4));
        oe.column_mut(0).assign(&ones);
        oe.column_mut(1).assign(&m);
        oe.column_mut(2).assign(&t);
        oe.column_mut(3).assign(&z);
        let mut me = Array2::zeros((n, 3));
        me.column_mut(0).assign(&ones);
        me.column_mut(1).assign(&t);
        me.column_mut(2).assign(&z);

        // Recover the fitted mediator T-coefficient (a) to verify the
        // Baron-Kenny identity ACME = a * b with b = 1.5 (exactly recovered
        // since Y is noiseless in [const, M, T, Z]).
        let a_fit = LinearModel::ols(m.clone(), me.clone())
            .unwrap()
            .fit()
            .unwrap()
            .params[1];

        let med = Mediation {
            outcome_endog: y,
            outcome_exog: oe,
            mediator_endog: m,
            mediator_exog: me,
            exp_pos_outcome: 2,
            exp_pos_mediator: 1,
            med_pos_outcome: 1,
        };
        let r = med.fit();
        // ACME = a * b with b = 1.5; ADE = c' = 0.7; control == treated (linear).
        assert!(
            (r.acme_avg - a_fit * 1.5).abs() < 1e-8,
            "acme {}",
            r.acme_avg
        );
        assert!((r.ade_avg - 0.7).abs() < 1e-8, "ade {}", r.ade_avg);
        assert!((r.acme_ctrl - r.acme_tx).abs() < 1e-10);
        assert!((r.total_effect - (r.acme_avg + r.ade_avg)).abs() < 1e-8);
    }
}
