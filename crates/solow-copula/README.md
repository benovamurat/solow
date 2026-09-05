# solow-copula

Bivariate copulas for the Solow statistical library.

Two families are provided:

* **Archimedean** copulas in closed form: [`ClaytonCopula`],
  [`FrankCopula`], and [`GumbelCopula`]. Each exposes `cdf(u, v)`,
  `pdf(u, v)`, and the Kendall's-tau mapping `tau()`.
* **Elliptical** copulas: [`GaussianCopula`] (with closed-form `pdf`
  via the normal-quantile transform, an analytic `tau`/`spearmans_rho`,
  and a `cdf` built on a bivariate-normal CDF) and an optional
  [`StudentTCopula`] `pdf`.

The free functions [`kendalls_tau`] and [`spearmans_rho`] compute the
sample rank correlations for paired data.

All quantities are cross-validated against the canonical Python
reference (`distributions.copula.api`) in `tests/reference.rs`.

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-copula) · License: BSD-3-Clause
