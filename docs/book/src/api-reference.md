# API reference

This user guide is the *narrative* documentation — concepts, model definitions,
worked examples, and plots. The **complete, per-function API reference** (every
public type, method, field, and free function, with signatures and doc
comments) is the generated `rustdoc`.

- **Online:** once a crate is published, its reference lives at
  `https://docs.rs/<crate>` — for example
  [`docs.rs/solow-regression`](https://docs.rs/solow-regression). The docs.rs
  build is automatic; every release is rendered there.
- **Locally:** build it from the source tree with

  ```text
  cargo doc --no-deps --workspace --open
  ```

  This produces the same reference under `target/doc/` and opens it in your
  browser. The whole workspace documents with zero warnings.

Each guide page ends with a **Module reference** table listing that area's key
public items; the links below take you to the full rustdoc for the crate behind
each area.

## Crates by area

| Area (guide page) | Crate | API reference |
| --- | --- | --- |
| [Linear regression](./regression.md) | `solow-regression` | [docs.rs](https://docs.rs/solow-regression) |
| [Generalized linear models](./glm.md) | `solow-glm` | [docs.rs](https://docs.rs/solow-glm) |
| [Generalized estimating equations](./gee.md) | `solow-gee` | [docs.rs](https://docs.rs/solow-gee) |
| [Generalized additive models](./gam.md) | `solow-gam` | [docs.rs](https://docs.rs/solow-gam) |
| [Robust linear models](./robust.md) | `solow-robust` | [docs.rs](https://docs.rs/solow-robust) |
| [Linear mixed effects](./mixed.md) | `solow-mixed` | [docs.rs](https://docs.rs/solow-mixed) |
| [Discrete & count outcomes](./discrete.md) | `solow-discrete` | [docs.rs](https://docs.rs/solow-discrete) |
| [Time series analysis](./time-series.md) | `solow-tsa` | [docs.rs](https://docs.rs/solow-tsa) |
| [State space methods](./statespace.md) | `solow-statespace` | [docs.rs](https://docs.rs/solow-statespace) |
| [Vector autoregressions](./var.md) | `solow-var` | [docs.rs](https://docs.rs/solow-var) |
| [Markov switching & regimes](./regime.md) | `solow-regime` | [docs.rs](https://docs.rs/solow-regime) |
| [Survival & duration](./duration.md) | `solow-duration` | [docs.rs](https://docs.rs/solow-duration) |
| [Nonparametric methods](./nonparametric.md) | `solow-nonparametric` | [docs.rs](https://docs.rs/solow-nonparametric) |
| [Multivariate analysis](./multivariate.md) | `solow-multivariate` | [docs.rs](https://docs.rs/solow-multivariate) |
| [Empirical likelihood](./emplike.md) | `solow-emplike` | [docs.rs](https://docs.rs/solow-emplike) |
| [Copulas](./copula.md) | `solow-copula` | [docs.rs](https://docs.rs/solow-copula) |
| [Other likelihood models](./othermod.md) | `solow-othermod` | [docs.rs](https://docs.rs/solow-othermod) |
| [Multiple imputation](./imputation.md) | `solow-impute` | [docs.rs](https://docs.rs/solow-impute) |
| [Bayesian methods](./bayes.md) | `solow-bayes` | [docs.rs](https://docs.rs/solow-bayes) |
| [Statistical tests](./stats-tests.md) | `solow-stats` | [docs.rs](https://docs.rs/solow-stats) |
| [Distributions & special functions](./distributions.md) | `solow-distributions` | [docs.rs](https://docs.rs/solow-distributions) |
| [Statistical graphics](./graphics.md) | `solow-graphics` | [docs.rs](https://docs.rs/solow-graphics) |
| [Optimization](./optimization.md) | `solow-optimize` | [docs.rs](https://docs.rs/solow-optimize) |
| [Summary tables](./summary-tables.md) | `solow-summary` | [docs.rs](https://docs.rs/solow-summary) |
| [The formula interface](./formula.md) | `solow-formula`, `solow-fit` | [docs.rs](https://docs.rs/solow-formula) |

## Foundational crates

These underpin everything above and are usually used indirectly:

| Crate | Role | API reference |
| --- | --- | --- |
| `solow` | Umbrella crate re-exporting the stack | [docs.rs](https://docs.rs/solow) |
| `solow-core` | Shared error type and data tools | [docs.rs](https://docs.rs/solow-core) |
| `solow-linalg` | From-scratch linear algebra (SVD, eigh, QR, Cholesky) | [docs.rs](https://docs.rs/solow-linalg) |
| `solow-viz` | Dependency-light SVG plotting backend | [docs.rs](https://docs.rs/solow-viz) |
