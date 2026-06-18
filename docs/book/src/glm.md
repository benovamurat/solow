# Generalized linear models

The [`solow-glm`] crate fits generalized linear models by iteratively
reweighted least squares (IRLS). A GLM is defined by a [`Family`] (the
exponential-dispersion distribution of the response) and a [`Link`] (the
function relating the linear predictor to the mean). Each family has a canonical
default link, which you can override.

> As with linear regression, you supply the intercept column yourself.

## Families and links

`Family` variants:

| Family | Variance `V(μ)` | Default link |
| --- | --- | --- |
| `Family::Gaussian` | `1` | `Link::Identity` |
| `Family::Binomial` | `μ(1 − μ)` | `Link::Logit` |
| `Family::Poisson` | `μ` | `Link::Log` |
| `Family::Gamma` | `μ²` | `Link::InversePower` |
| `Family::InverseGaussian` | `μ³` | `Link::InverseSquared` |
| `Family::NegativeBinomial { alpha }` | `μ + α μ²` | `Link::Log` |

`Link` variants: `Identity`, `Log`, `Logit`, `Probit`, `CLogLog`,
`InversePower`, `InverseSquared`, `Sqrt`.

## Poisson regression

```rust
use ndarray::{array, Array1};
use solow_glm::{Family, Glm};

// Counts increasing with the predictor; first column is the intercept.
let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
let y: Array1<f64> = array![1.0, 2.0, 4.0, 7.0, 12.0];

let res = Glm::new(y, x, Family::Poisson).unwrap().fit().unwrap();

assert!(res.converged);
println!("params   = {:?}", res.params);
println!("deviance = {:.4}", res.deviance);
println!("AIC      = {:.2}", res.aic);
```

`Glm::new` uses the family's canonical link. The fitted [`GlmResults`] exposes
`params`, `bse`, `tvalues`, `pvalues`, `deviance`, `llf`, `aic`, `bic`,
`scale`, `converged`, plus `conf_int(alpha)` and `pseudo_rsquared()`.

## Logistic regression (Binomial family)

```rust
use ndarray::{array, Array1};
use solow_glm::{Family, Glm};

let x = array![
    [1.0, -1.2], [1.0, -0.4], [1.0, 0.3], [1.0, 0.9], [1.0, 1.8],
];
let y: Array1<f64> = array![0.0, 0.0, 1.0, 1.0, 1.0];

let res = Glm::new(y, x, Family::Binomial).unwrap().fit().unwrap();
println!("logit coefficients = {:?}", res.params);
```

## Choosing a non-canonical link

Use `Glm::with_link` to pair a family with a non-default link — for instance, a
binomial model with the complementary log-log link:

```rust
use ndarray::{array, Array1};
use solow_glm::{Family, Glm, Link};

let x = array![
    [1.0, -1.2], [1.0, -0.4], [1.0, 0.3], [1.0, 0.9], [1.0, 1.8],
];
let y: Array1<f64> = array![0.0, 0.0, 1.0, 1.0, 1.0];

let res = Glm::with_link(y, x, Family::Binomial, Link::CLogLog)
    .unwrap()
    .fit()
    .unwrap();
println!("cloglog coefficients = {:?}", res.params);
```

## Gamma regression

```rust
use ndarray::{array, Array1};
use solow_glm::{Family, Glm};

let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
let y: Array1<f64> = array![2.0, 3.5, 5.0, 8.0, 13.0];

let res = Glm::new(y, x, Family::Gamma).unwrap().fit().unwrap();
println!("scale (dispersion) = {:.4}", res.scale);
```

## Negative-binomial and Tweedie

For overdispersed counts, the negative-binomial family takes a dispersion
parameter `alpha`:

```rust
use ndarray::{array, Array1};
use solow_glm::{Family, Glm};

let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
let y: Array1<f64> = array![2.0, 1.0, 5.0, 9.0, 14.0];

let res = Glm::new(y, x, Family::NegativeBinomial { alpha: 1.0 })
    .unwrap()
    .fit()
    .unwrap();
println!("converged = {}", res.converged);
```

The [`TweedieGlm`] estimator (also in `solow-glm`) covers Tweedie responses
with a configurable power parameter.

[`solow-glm`]: https://github.com/solow-rs/solow
[`Family`]: ./glm.md
[`Link`]: ./glm.md
[`GlmResults`]: ./glm.md
[`TweedieGlm`]: ./glm.md
