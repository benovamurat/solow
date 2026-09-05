# solow-duration

Survival / duration analysis for the Solow statistical-computing stack.

* [`SurvfuncRight`] — the Kaplan–Meier (product-limit) estimator of a
  right-censored survival function, with Greenwood standard errors.
* [`PHReg`] — Cox proportional-hazards regression estimated by maximizing
  the Breslow partial log-likelihood, exposing coefficients, standard
  errors, z-statistics, p-values and the partial log-likelihood.

Both estimators are cross-validated against golden reference values frozen
in `tests/fixtures/duration.json`.

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-duration) · License: BSD-3-Clause
