# solow-datasets

**Dataset generators and toy loaders for Rust.** Synthetic classification and regression generators, geometry benchmarks (moons, circles, Swiss roll), classic loaders (iris, wine, diabetes, breast cancer), and utility helpers for class weights and resampling.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-datasets = "0.7"
```

## Quick example

```rust
use solow_datasets::{load_iris, make_classification};

let iris = load_iris();
let (x, y) = make_classification(500, 20, 3, 42);
```

## What is inside

### Toy loaders

| Function | What it does |
|---|---|
| `load_iris` | 150-sample 4-feature Iris flower classification. |
| `load_wine` | 178-sample 13-feature wine cultivar classification. |
| `load_diabetes` | 442-sample 10-feature diabetes regression. |
| `load_breast_cancer` | 569-sample 30-feature Wisconsin breast cancer classification. |

### Synthetic generators

| Function | What it does |
|---|---|
| `make_classification` | Multi-class informative + redundant + noise features with configurable class separation. |
| `make_regression` | Linear regression targets with configurable noise and coefficient sparsity. |
| `make_blobs` | Isotropic Gaussian blobs in R^n. |
| `make_moons` | Two interleaving half-circles for non-linear boundary testing. |
| `make_circles` | Concentric circles for non-linear kernel benchmarks. |
| `make_swiss_roll` | Classic manifold-learning benchmark. |
| `make_low_rank_matrix` | Low-rank matrix with controllable singular value profile for factorization tests. |

### Utility helpers

| Function | What it does |
|---|---|
| `compute_class_weight` | Balanced class weights from a label array. |
| `compute_sample_weight` | Per-sample weights matching a balanced class weighting. |
| `resample_indices_with_replace`, `resample_indices_no_replace` | Deterministic resampling with a caller seed. |

## Correctness

- Every generator uses a portable MMIX-LCG PRNG. A fixed seed reproduces the dataset bit-for-bit across runs and platforms.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
