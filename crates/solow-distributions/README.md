# solow-distributions

Special functions and the continuous distributions used for statistical
inference. Everything is implemented from scratch in pure Rust and validated
against an authoritative reference.

- [`special`] — `lgamma`, `digamma`, incomplete beta/gamma (+ inverses), `erf`
- [`continuous`] — [`Normal`], [`StudentT`], [`FDist`], [`ChiSquared`] and the
  matching free functions (`norm_cdf`, `t_sf`, `f_sf`, `chi2_sf`, …)

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-distributions) · License: BSD-3-Clause
