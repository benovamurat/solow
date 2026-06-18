# Installation

Solow is a Cargo workspace of focused crates. You can depend on the single
crate you need, or on the umbrella [`solow`] crate that re-exports the whole
public API.

## Requirements

- **Rust** 1.80 or newer (stable). Solow tracks the stable toolchain; the
  repository pins it with a `rust-toolchain.toml`.
- **No system math libraries.** Solow's linear algebra and special functions
  are pure Rust, so there is no LAPACK, BLAS, or C compiler to install.

The only foundational third-party dependency is [`ndarray`], which Solow uses
for its vector and matrix types (`Array1<f64>` / `Array2<f64>`).

## Adding individual crates

Pick the crates that match what you are doing. For example, ordinary least
squares plus the summary renderer:

```toml
[dependencies]
solow-core = "0.1"
solow-regression = "0.1"
solow-summary = "0.1"
ndarray = "0.16"
```

A few common combinations:

| Task | Crates |
| --- | --- |
| OLS / WLS / GLS | `solow-core`, `solow-regression` |
| GLM | `solow-glm` |
| Logit / Probit / Poisson | `solow-discrete` |
| Time-series tools and AutoReg | `solow-tsa` |
| SARIMAX / Kalman | `solow-statespace` |
| VAR / VECM | `solow-var` |
| Formula → design matrix | `solow-formula` |
| Rendered summary table | `solow-summary` |

## Adding the umbrella crate

If you would rather have everything available behind one dependency, use the
umbrella crate:

```toml
[dependencies]
solow = "0.1"
ndarray = "0.16"
```

Each library is then reachable as a module, and a `prelude` collects the most
common imports:

```rust
use solow::prelude::*;       // Array1, Array2, Axis, core error types
use solow::regression::LinearModel;
use solow::glm::{Family, Glm};
use solow::tsa::{acf, AutoReg, Trend};
```

## Building from source

```sh
git clone https://github.com/solow-rs/solow
cd solow
cargo build --workspace
cargo test --workspace
```

Because the numerical core is pure Rust, this builds on Linux, macOS, and
Windows with nothing more than a stable Rust toolchain.

## Python users

Solow can also be imported from Python through the `solow-py` bindings (a native
CPython extension built with PyO3). That crate is its own standalone workspace.
See [Using Solow from Python](./python.md) for the build steps and API.

[`ndarray`]: https://docs.rs/ndarray
[`solow`]: https://github.com/solow-rs/solow
