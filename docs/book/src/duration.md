# Duration and hazard models

The `solow-duration` crate covers right-censored survival (duration) analysis.
It provides the Kaplan–Meier estimator of a survival curve (`SurvfuncRight`),
Cox proportional-hazards regression by partial likelihood (`PHReg`, and
`PHRegTies` with selectable tie handling), the weighted log-rank family of
group-comparison tests (`survdiff`), and cumulative-incidence estimation for
competing risks (`CumIncidenceRight`). Every estimator targets the
single-stratum, no-left-truncation, no-offset case and is cross-validated
against frozen golden values from the reference.

> Status is the universal censoring flag: `1.0` marks an observed event
> (failure) and `0.0` marks a right-censored observation. For competing risks
> the status carries the cause label instead (`0` censored, `1..=J` causes).

## Background

Let \\( T \\) be a non-negative failure time. Survival analysis describes its
distribution through the **survival function** \\( S(t) = P(T > t) \\), the
**hazard** \\( \lambda(t) \\) (the instantaneous failure rate among survivors),
and the **cumulative hazard** \\( \Lambda(t) = \int_0^t \lambda(u)\,du \\), which
are linked by

\\[ S(t) = \exp\\{-\Lambda(t)\\}, \qquad \lambda(t) = -\frac{d}{dt}\log S(t). \\]

### Kaplan–Meier estimator

With right-censored data, order the distinct failure times
\\( t_1 < t_2 < \dots \\). Let \\( d_i \\) be the number of failures at \\( t_i \\)
and \\( n_i \\) the size of the risk set just before \\( t_i \\). The
product-limit (Kaplan–Meier) estimator is

\\[ \hat S(t) = \prod_{t_i \le t} \left( 1 - \frac{d_i}{n_i} \right), \\]

and Greenwood's formula gives its variance,

\\[ \widehat{\operatorname{Var}}\\,\hat S(t) = \hat S(t)^2
   \sum_{t_i \le t} \frac{d_i}{n_i\,(n_i - d_i)}. \\]

`SurvfuncRight` reports the estimator at the **distinct event times only**;
censored-only times never appear on their own, and the Greenwood standard error
is `NaN` where the risk set is exhausted (\\( n_i = d_i \\)).

### Cox proportional hazards

The Cox model leaves the baseline hazard unspecified and writes the hazard of
subject \\( i \\) with covariates \\( x_i \\) as

\\[ \lambda(t \mid x_i) = \lambda_0(t)\,\exp(x_i^\top \beta), \\]

so \\( \exp(\beta_j) \\) is the hazard ratio for a unit change in covariate
\\( j \\). The coefficients are estimated by maximizing the **Breslow partial
log-likelihood**, which conditions on the risk set \\( R(t_i) \\) at each failure
time and so cancels \\( \lambda_0 \\):

\\[ \log L(\beta) = \sum_i \left[ x_i^\top \beta
   - \log\\!\\!\sum_{j \in R(t_i)} \exp(x_j^\top \beta) \right]. \\]

When several subjects fail at the same time, the **Efron** approximation
progressively deflates the tied failures' contribution to the denominator and is
more accurate than Breslow; both are available through `PHRegTies` via the
`Ties` enum. The model is fit with a Newton step that drives the gradient (score)
to zero; standard errors come from the inverse observed information
\\( \big({-}\nabla^2 \log L\big)^{-1} \\). Because the baseline hazard absorbs the
intercept, you supply **no constant column** in the covariate matrix.

### Log-rank tests

To compare \\( g \\) groups, `survdiff` accumulates, at each event time, the
observed group event counts \\( O \\) and their expectations \\( E \\) under the
null of equal hazards, with a weight \\( w_k \\) per event time. The statistic

\\[ \chi^2 = (O - E)^\top V^{-1} (O - E) \\]

is compared to a \\( \chi^2_{g-1} \\) distribution. Weights \\( w_k = 1 \\) give
the unweighted log-rank (Mantel–Cox) test; the Gehan–Breslow, Tarone–Ware and
Fleming–Harrington families weight early or late differences differently.

### Cumulative incidence (competing risks)

With competing causes \\( j = 1, \dots, J \\), the cause-specific cumulative
incidence \\( I_j(t) = P(T \le t,\, J = j) \\) is estimated from the pooled
all-cause survival \\( \hat S \\) as

\\[ \hat I_j(t) = \sum_{t_i \le t} \hat S(t_{i-1})\,\frac{d_{j,i}}{n_i}, \\]

where \\( d_{j,i} \\) counts cause-\\( j \\) events at \\( t_i \\). This is what
`CumIncidenceRight` computes, with an Aalen-type variance for the standard error.

## Example

A Cox proportional-hazards fit. Covariates go in `exog` (one row per subject, no
intercept column); `time` and `status` are parallel slices.

```rust
use ndarray::{array, Array2};
use solow_duration::PHReg;

// Seven subjects: event/censoring times, a single covariate, and status
// (1 = event observed, 0 = right-censored).
let time = [4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0];
let status = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
let exog: Array2<f64> = array![
    [0.5], [1.2], [-0.3], [0.8], [0.1], [-1.0], [0.4],
];

let model = PHReg::new(&time, &exog, &status).unwrap();
let res = model.fit().unwrap();

assert!(res.converged);
println!("coef (log hazard ratio) = {:?}", res.params);
println!("std err                 = {:?}", res.bse);
println!("z                       = {:?}", res.tvalues);
println!("p-value                 = {:?}", res.pvalues);
println!("partial log-likelihood  = {:.4}", res.llf);
```

`PHReg::fit` returns a `PHRegResults` exposing `params`, `bse`, `tvalues`,
`pvalues`, `cov_params`, `llf`, and `converged`. A positive coefficient means
the covariate raises the hazard (shortens survival); the hazard ratio is
`res.params.mapv(f64::exp)`. With this small, untied data set the fit converges
in a handful of Newton iterations and the printed gradient at the optimum is
numerically zero — the exact coefficient value depends on the data and is not
reproduced here.

To handle tied event times with the Efron correction, use `PHRegTies`:

```rust
use ndarray::{array, Array2};
use solow_duration::{PHRegTies, Ties};

let time = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0];   // tied failure times
let status = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0];
let exog: Array2<f64> = array![
    [0.2], [-0.5], [1.0], [0.3], [-0.8], [0.6],
];

let res = PHRegTies::new(&time, &exog, &status, Ties::Efron)
    .unwrap()
    .fit()
    .unwrap();
println!("efron coef = {:?}", res.params);
println!("ties       = {:?}", res.ties);
```

A Kaplan–Meier survival curve from right-censored times:

```rust
use solow_duration::SurvfuncRight;

let time = [1.0, 2.0, 3.0, 4.0, 5.0];
let status = [1.0, 0.0, 1.0, 1.0, 1.0];   // second observation censored

let sf = SurvfuncRight::new(&time, &status).unwrap();
for k in 0..sf.surv_times.len() {
    println!(
        "t = {:>3}  S = {:.4}  se = {:.4}  n_risk = {}  d = {}",
        sf.surv_times[k], sf.surv_prob[k], sf.surv_prob_se[k],
        sf.n_risk[k], sf.n_events[k],
    );
}
```

`SurvfuncRight` exposes `surv_times`, `surv_prob`, `surv_prob_se` (Greenwood,
`NaN` once the risk set is exhausted), `n_risk`, and `n_events`. The censored
observation contributes to the risk set but produces no row of its own.

Comparing two groups with the log-rank test:

```rust
use solow_duration::{survdiff, WeightType};

let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let status = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
let group = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

let res = survdiff(&time, &status, &group, WeightType::LogRank).unwrap();
println!("chisq = {:.4}  df = {}  p = {:.4}", res.chisq, res.df, res.pvalue);
```

`survdiff` returns a `SurvDiffResult` with `chisq`, `pvalue`, and `df`
(`number of groups − 1`). Swapping `WeightType::LogRank` for `GehanBreslow`,
`TaroneWare`, or `FlemingHarrington(p)` re-weights the same machinery.

## Module reference

**Models**

| Name | Description |
| --- | --- |
| `SurvfuncRight` | Kaplan–Meier (product-limit) survival estimator with Greenwood standard errors. |
| `PHReg` | Cox proportional-hazards regression using the Breslow partial likelihood. |
| `PHRegTies` | Cox PH regression with a selectable Breslow or Efron tie-handling method. |
| `CumIncidenceRight` | Cause-specific cumulative incidence functions for competing-risks data. |

**Results**

| Name | Description |
| --- | --- |
| `PHRegResults` | Fitted `PHReg`: `params`, `bse`, `tvalues`, `pvalues`, `cov_params`, `llf`, `converged`. |
| `PHRegTiesResults` | Fitted `PHRegTies`: as above, plus the `ties` method used. |
| `SurvDiffResult` | Log-rank outcome: `chisq`, `pvalue`, `df`. |

**Functions**

| Name | Description |
| --- | --- |
| `survdiff` | Weighted log-rank test of equality of two or more survival distributions. |

**Enums**

| Name | Description |
| --- | --- |
| `Ties` | Tie-handling for the Cox partial likelihood: `Breslow` or `Efron`. |
| `WeightType` | Log-rank weight family: `LogRank`, `GehanBreslow`, `TaroneWare`, `FlemingHarrington(p)`. |

Selected methods worth noting: `PHReg::breslow_loglike`, `breslow_gradient`,
and `breslow_hessian` expose the partial-likelihood objective and its
derivatives directly; `PHRegTies` adds the `efron_*` counterparts plus
method-dispatching `loglike` / `gradient` / `hessian`.

Full API: see the generated rustdoc for `solow-duration`.

## References

- Kaplan, E. L., and Meier, P. (1958). "Nonparametric Estimation from
  Incomplete Observations." *Journal of the American Statistical Association*,
  53(282), 457–481.
- Cox, D. R. (1972). "Regression Models and Life-Tables." *Journal of the
  Royal Statistical Society, Series B*, 34(2), 187–220.
- Efron, B. (1977). "The Efficiency of Cox's Likelihood Function for Censored
  Data." *Journal of the American Statistical Association*, 72(359), 557–565.
- Kalbfleisch, J. D., and Prentice, R. L. (2002). *The Statistical Analysis of
  Failure Time Data*, 2nd ed. Wiley.
