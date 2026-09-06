# solow-preprocessing

**Feature preprocessing for Rust.** Scalers, encoders, power and quantile transforms, discretization, polynomial and spline basis, binarizers, target encoder, function transformer, in one memory-safe pure-Rust crate.

Every transformer implements `fit`, `transform`, `fit_transform`, and `inverse_transform` where invertible. Inverse round-trip is exact to machine precision on well-scaled data.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-preprocessing = "0.7"
```

## Quick example

```rust
use solow_preprocessing::StandardScaler;

let mut sc = StandardScaler::new();
sc.fit(&x)?;
let x_std = sc.transform(&x)?;
let x_back = sc.inverse_transform(&x_std)?;
```

## What is inside

### Scalers

| Transformer | What it does |
|---|---|
| `StandardScaler` | Zero-mean, unit-variance per column via one-pass Welford variance. |
| `MinMaxScaler` | Range projection onto arbitrary `[a, b]`. |
| `RobustScaler` | Median / IQR scaling, outlier-immune. |
| `MaxAbsScaler` | Divide each column by its max absolute value. |
| `Normalizer` | Row-wise L1 or L2 unit normalization. |

### Power and quantile transforms

| Transformer | What it does |
|---|---|
| `PowerTransformer` | Yeo-Johnson (works with negatives) or Box-Cox (positive only) with maximum-likelihood λ per column. |
| `QuantileTransformer` | Empirical-CDF Gaussianization or uniform mapping via quantile lookup. |

### Discretization

| Transformer | What it does |
|---|---|
| `KBinsDiscretizer` | Bin numeric features with uniform, quantile, or k-means strategy. Output as ordinal or one-hot. |
| `Binarizer` | Threshold-based binary indicator. |

### Categorical encoders

| Transformer | What it does |
|---|---|
| `OneHotEncoder` | One-hot with `drop_first` and `handle_unknown` options. |
| `OrdinalEncoder` | Integer encoding with per-column category lookup. |
| `LabelEncoder` | Single-column label to integer. |
| `LabelBinarizer` | Binary indicator per unique label. |
| `MultiLabelBinarizer` | Multi-label indicator matrix. |
| `TargetEncoder` | Smoothed target-mean encoding of categorical features. |

### Feature construction

| Transformer | What it does |
|---|---|
| `PolynomialFeatures` | Graded-lex polynomial expansion up to configurable degree with or without interaction-only. |
| `SplineTransformer` | B-spline basis expansion with configurable degree and knots. |
| `FunctionTransformer` | Wrap an arbitrary closure as a pipeline transformer. |

## Correctness

- Every transformer cross-verified against committed golden reference fixtures on every CI run.
- Inverse transforms round-trip exact to machine precision on well-scaled data.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
