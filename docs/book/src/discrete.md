# Discrete choice & counts

The [`solow-discrete`] crate fits discrete-choice and count models by maximum
likelihood. The core estimators — [`Logit`], [`Probit`], and [`Poisson`] — use
a full Newton step with analytic log-likelihood, score, and Hessian, converging
to the true optimum so results match the reference to machine precision.

Each model is constructed with `new(endog, exog)` and fit with `.fit()`,
returning a [`DiscreteResults`]. As elsewhere in Solow, you provide the
intercept column yourself.

## Logit and Probit

```rust
use ndarray::{array, Array2};
use solow_discrete::{Logit, Probit};

let mut x = Array2::<f64>::ones((6, 2));
x.column_mut(1)
    .assign(&array![-1.1, -0.4, 0.1, 0.7, 1.2, 1.9]);
let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

let logit = Logit::new(y.clone(), x.clone()).unwrap().fit().unwrap();
assert!(logit.converged);
println!("logit params = {:?}", logit.params);
println!("logit llf    = {:.4}", logit.llf);

let probit = Probit::new(y, x).unwrap().fit().unwrap();
println!("probit params = {:?}", probit.params);
```

`DiscreteResults` carries `params`, `bse`, `tvalues`, `pvalues`, `llf`, `aic`,
`bic`, `converged`, and `conf_int(alpha)`. It also reports the model-fit
statistics relative to the intercept-only model: `llnull`, the likelihood-ratio
statistic `llr` with its p-value `llr_pvalue`, and McFadden's pseudo-R²
`prsquared`.

```rust
use ndarray::{array, Array2};
use solow_discrete::Logit;

let mut x = Array2::<f64>::ones((6, 2));
x.column_mut(1).assign(&array![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

let res = Logit::new(y, x).unwrap().fit().unwrap();
println!("McFadden pseudo R^2 = {:.4}", res.prsquared);
println!("LR test: stat = {:.4}, p = {:.4}", res.llr, res.llr_pvalue);
```

## Poisson count regression

```rust
use ndarray::{array, Array2};
use solow_discrete::Poisson;

let mut x = Array2::<f64>::ones((5, 2));
x.column_mut(1).assign(&array![0.0, 1.0, 2.0, 3.0, 4.0]);
let y = array![1.0, 2.0, 4.0, 7.0, 12.0];

let res = Poisson::new(y, x).unwrap().fit().unwrap();
println!("Poisson params = {:?}", res.params);
println!("AIC = {:.2}", res.aic);
```

## Confidence intervals

```rust
use ndarray::{array, Array2};
use solow_discrete::Logit;

let mut x = Array2::<f64>::ones((6, 2));
x.column_mut(1).assign(&array![-1.1, -0.4, 0.1, 0.7, 1.2, 1.9]);
let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

let res = Logit::new(y, x).unwrap().fit().unwrap();
let ci = res.conf_int(0.05); // (k, 2): [lower, upper] per coefficient
println!("95% CI =\n{:?}", ci);
```

## More count and choice models

`solow-discrete` also provides:

- **Multinomial logit** — [`MNLogit`] for an unordered categorical response.
- **Negative binomial** — [`NegativeBinomial`] for overdispersed counts.
- **Ordered models** — [`OrderedModel`] (logit/probit) for ordinal responses,
  selectable via [`Distr`].
- **Generalized Poisson** — [`GeneralizedPoisson`].
- **Zero-inflated and hurdle counts** — [`HurdleCountModel`] and the
  zero-inflated estimators.
- **Conditional / fixed-effects** logit and Poisson — [`ConditionalLogit`],
  [`ConditionalPoisson`].
- **Left-truncated Poisson** — [`TruncatedLFPoisson`].

[`solow-discrete`]: https://github.com/solow-rs/solow
[`Logit`]: ./discrete.md
[`Probit`]: ./discrete.md
[`Poisson`]: ./discrete.md
[`DiscreteResults`]: ./discrete.md
[`MNLogit`]: ./discrete.md
[`NegativeBinomial`]: ./discrete.md
[`OrderedModel`]: ./discrete.md
[`Distr`]: ./discrete.md
[`GeneralizedPoisson`]: ./discrete.md
[`HurdleCountModel`]: ./discrete.md
[`ConditionalLogit`]: ./discrete.md
[`ConditionalPoisson`]: ./discrete.md
[`TruncatedLFPoisson`]: ./discrete.md
