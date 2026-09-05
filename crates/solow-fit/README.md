# solow-fit

Ergonomic, formula-driven model fitting for the Solow statistical stack —
the one-call bridge from an R/patsy-style formula string plus named data to a
fully fitted model whose coefficients are *labeled* with the design column
names. This is the from-scratch equivalent of the reference library's
`from_formula` constructors: instead of hand-assembling a design matrix and
threading column names through by hand, you write

```text
ols("y ~ x1 + C(g)", &df)
```

and get back a [`NamedFit`] that pairs the estimator's results with the
ordered column names, ready to [`summary`](NamedFit::summary).

The formula layer ([`solow_formula`]) already emits the `Intercept` column
when the formula carries one, so these functions pass the design straight
through to the estimator without re-adding a constant — the formula path and
the manual `add_constant` + estimator path therefore produce *identical*
coefficients and standard errors (see the crate's tests, which assert
agreement to `≤ 1e-12`).

## Models

* [`ols`] / [`wls`] / [`gls`] — linear regression ([`LinearResults`]).
* [`glm`] — generalized linear models with a [`Family`] and optional
  [`Link`] ([`GlmResults`]).
* [`logit`] / [`probit`] / [`poisson`] — discrete-choice and count models
  ([`DiscreteResults`]).

## One import

[`DataFrame`] is re-exported here, so a typical user needs a single `use`:

```
use solow_fit::{ols, DataFrame};

let mut df = DataFrame::new();
df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
df.add_numeric("x", vec![0.0, 1.0, 2.0, 3.0, 4.0]);
let fit = ols("y ~ x", &df).unwrap();
assert_eq!(fit.names, vec!["Intercept".to_string(), "x".to_string()]);
// The fitted line is y = 1 + x, recovered exactly.
assert!((fit.results.params[0] - 1.0).abs() < 1e-10);
assert!((fit.results.params[1] - 1.0).abs() < 1e-10);
println!("{}", fit.summary());
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-fit) · License: BSD-3-Clause
