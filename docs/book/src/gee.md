# Generalized estimating equations

The `solow-gee` crate fits *marginal* (population-averaged) regression models
for correlated responses — repeated measures, longitudinal panels, or otherwise
clustered data — by generalized estimating equations (GEE). The main entry
point is the `Gee` model: you supply a response, a design, a per-observation
group label, a GLM `Family`, and a working within-cluster correlation
(`CovStruct`); `.fit()` returns a `GeeResults` carrying the mean coefficients
together with both robust (sandwich) and naive (model-based) inference. For
categorical responses the crate also provides `NominalGee` and `OrdinalGee`.

## Background

GEE models the marginal mean of each response without committing to a full
joint likelihood for the correlated observations. For cluster \\( i \\) with
\\( n_i \\) observations, the marginal mean is linked to the covariates exactly
as in a GLM,

\\[
g(\mu_{ij}) = x_{ij}^{\top} \beta , \qquad
\operatorname{Var}(y_{ij}) = \phi\, V(\mu_{ij}),
\\]

where \\( g \\) is the link, \\( V \\) the family variance function, and
\\( \phi \\) a dispersion (scale) parameter. The within-cluster dependence is
captured by a **working** correlation matrix \\( R(\alpha) \\), giving the
working covariance

\\[
V_i = \phi\, A_i^{1/2}\, R(\alpha)\, A_i^{1/2},
\qquad A_i = \operatorname{diag}\!\big(V(\mu_{i1}), \dots, V(\mu_{i n_i})\big).
\\]

The coefficients solve the estimating equations

\\[
U(\beta) \;=\; \sum_{i} D_i^{\top} V_i^{-1} \big(y_i - \mu_i\big) \;=\; 0,
\qquad D_i = \frac{\partial \mu_i}{\partial \beta},
\\]

which `solow-gee` drives to zero by Fisher scoring. Each scoring step uses the
update \\( \big(\sum_i D_i^{\top} V_i^{-1} D_i\big)^{-1} U(\beta) \\); for the
exchangeable structure the association parameter \\( \alpha \\) is re-estimated
from standardized residuals between mean updates.

The decisive property of GEE is that \\( \hat\beta \\) stays consistent even
when \\( R(\alpha) \\) is misspecified, provided the marginal mean is correct.
Valid standard errors then come from the **sandwich** (robust) covariance

\\[
\operatorname{Cov}(\hat\beta) \;=\; B^{-1}\, M\, B^{-1},
\qquad
B = \sum_i D_i^{\top} V_i^{-1} D_i,
\quad
M = \sum_i D_i^{\top} V_i^{-1} \big(y_i - \mu_i\big)\big(y_i - \mu_i\big)^{\top} V_i^{-1} D_i .
\\]

The "bread" \\( B^{-1} \\) (scaled by \\( \phi \\)) is reported as the naive
model-based covariance; the full sandwich is the robust covariance, with the
per-cluster outer product \\( M \\) as its meat.

> **Scope.** `solow-gee` implements the `Independence` and `Exchangeable`
> working correlations with the Gaussian, Poisson, and Binomial families. It is
> deliberately narrower than the reference's GEE module: AR(1) and other
> time-dependent correlation structures, and GLM-style QIC model selection, are
> not currently provided.

## Example

The example below fits a Poisson GEE with an exchangeable working correlation
to three clusters of two observations each. The first design column is the
intercept (as elsewhere in Solow, you supply it yourself), and `groups` assigns
each row to a cluster.

```rust
use ndarray::{array, Array1, Array2};
use solow_gee::{CovStruct, Gee, GeeResults};
use solow_glm::Family;

// Three clusters (labels 0, 1, 2), two observations each.
let x: Array2<f64> = array![
    [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
    [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
];
let y: Array1<f64> = array![1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
let groups = [0i64, 0, 1, 1, 2, 2];

let res: GeeResults = Gee::new(y, x, &groups, Family::Poisson, CovStruct::Exchangeable)
    .unwrap()
    .fit()
    .unwrap();

assert!(res.converged);
println!("params      = {:?}", res.params);     // mean coefficients β
println!("robust bse  = {:?}", res.bse);         // sandwich standard errors
println!("naive  bse  = {:?}", res.bse_naive);   // model-based standard errors
println!("dep_params  = {:.4}", res.dep_params); // exchangeable correlation α
println!("scale       = {:.4}", res.scale);      // dispersion (1 for Poisson)
```

`Gee::new` uses the family's canonical link (here `Link::Log`); use
`Gee::with_link` to pair a family with a non-default link. The builder methods
`.maxiter(m)` and `.ctol(t)` adjust the scoring iteration cap and the
convergence tolerance on the score-equation norm.

The printed `params` are the population-averaged log-rate coefficients. Because
this is a Poisson model, `scale` is fixed at `1.0` and `dep_params` reports the
fitted exchangeable correlation \\( \alpha \\). The `bse` field holds the
cluster-robust (sandwich) standard errors while `bse_naive` holds the
model-based ones; the two diverge when the working correlation is misspecified,
which is precisely when you should trust the robust column. `res.tvalues`,
`res.pvalues`, and the fitted means `res.fittedvalues` round out the inference.
(Exact numeric values depend on the fit and are not reproduced here.)

Two structural facts are worth noting, both exercised by the crate's tests.
With `CovStruct::Independence`, the GEE point estimates coincide with the
ordinary GLM MLE (only the inference differs, via the robust sandwich). And when
every cluster is a singleton, the exchangeable association is undefined and
falls back to zero, so the exchangeable and independence fits agree.

For an unordered multinomial response you would instead build a `NominalGee`,
and for an ordered (proportional-odds) response an `OrdinalGee`; both take a
`CategoricalCov` working association (`Independence` or the Heagerty–Zeger /
Lumley `GlobalOddsRatio`) and return a `CategoricalGeeResults`.

## Module reference

**Models**

| Name | Description |
| --- | --- |
| `Gee` | Marginal GEE for continuous/count/binary responses; `new`, `with_link`, builder `maxiter`/`ctol`, and `fit`. |
| `NominalGee` | Marginal GEE for an unordered (multinomial-logit) categorical response. |
| `OrdinalGee` | Marginal GEE for an ordered (proportional-odds cumulative) categorical response. |

**Results**

| Name | Description |
| --- | --- |
| `GeeResults` | Fit of `Gee`: `params`, `bse`, `bse_naive`, `tvalues`, `pvalues`, `cov_robust`, `cov_naive`, `dep_params`, `scale`, `fittedvalues`, `score_norm`, `converged`. |
| `CategoricalGeeResults` | Fit of `NominalGee`/`OrdinalGee`: the same inference fields plus `ncut` (number of cut points). |

**Enums**

| Name | Description |
| --- | --- |
| `CovStruct` | Working correlation for `Gee`: `Independence` or `Exchangeable` (compound symmetry). |
| `CategoricalCov` | Working association for categorical GEE: `Independence` or `GlobalOddsRatio`. |

The family/link types `Family` and `Link` are re-used from the `solow-glm`
crate. Full API: see the generated rustdoc for `solow-gee`.

## References

- Liang, K.-Y. and Zeger, S. L. (1986). Longitudinal data analysis using
  generalized linear models. *Biometrika*, 73(1), 13–22.
- Zeger, S. L. and Liang, K.-Y. (1986). Longitudinal data analysis for discrete
  and continuous outcomes. *Biometrics*, 42(1), 121–130.
- Heagerty, P. J. and Zeger, S. L. (1996). Marginal regression models for
  clustered ordinal measurements. *Journal of the American Statistical
  Association*, 91(435), 1024–1036.
- Diggle, P. J., Heagerty, P., Liang, K.-Y., and Zeger, S. L. (2002).
  *Analysis of Longitudinal Data* (2nd ed.). Oxford University Press.
