# solow-kernel-approx

**Kernel approximation via random features for Rust.** Random Fourier features (RBF sampler), Nyström method, and explicit feature maps for chi-squared, skewed chi-squared, and polynomial kernels.

These transformers let linear models emulate kernel methods with predictable memory and time budgets.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-kernel-approx = "0.7"
```

## Quick example

```rust
use solow_kernel_approx::RBFSampler;

let sampler = RBFSampler::new(500).gamma(0.1).seed(42).fit(&x)?;
let x_rff = sampler.transform(&x)?;
// x_rff now has 500 random Fourier features; feed to any linear model.
```

## What is inside

| Transformer | What it does |
|---|---|
| `RBFSampler` | Rahimi-Recht 2007 random Fourier features approximating the RBF kernel. |
| `Nystroem` | Nyström approximation of an arbitrary kernel using a random subset of training rows. |
| `AdditiveChi2Sampler` | Vedaldi-Zisserman explicit feature map for the additive chi-squared kernel. |
| `SkewedChi2Sampler` | Explicit feature map for the skewed chi-squared kernel. |
| `PolynomialCountSketch` | Pham-Pagh tensor sketch for polynomial kernels. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Stochastic transformers deterministic under a caller seed.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
