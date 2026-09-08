# Preprocessing

`solow-preprocessing` covers every day-one feature transformation: scalers, encoders, power and quantile transforms, discretization, polynomial and spline expansions, binarizers, target encoding, and function transformers.

Every transformer implements `fit`, `transform`, `fit_transform`, and `inverse_transform` where invertible. The inverse round-trip is exact to machine precision on well-scaled data.

## Scalers

| Transformer | What it does |
|---|---|
| `StandardScaler` | Zero-mean, unit-variance per column via one-pass Welford variance. |
| `MinMaxScaler` | Range projection onto arbitrary `[a, b]`. |
| `RobustScaler` | Median / IQR scaling, outlier-immune. |
| `MaxAbsScaler` | Divide each column by its max absolute value. Sparsity-preserving. |
| `Normalizer` | Row-wise L1 or L2 unit normalization. |

## Power and quantile transforms

| Transformer | What it does |
|---|---|
| `PowerTransformer` | Yeo-Johnson (works with negatives) or Box-Cox (positive only) with maximum-likelihood λ per column. |
| `QuantileTransformer` | Empirical-CDF Gaussianization or uniform mapping via quantile lookup. |

## Discretization and binarization

| Transformer | What it does |
|---|---|
| `KBinsDiscretizer` | Bin numeric features with uniform, quantile, or k-means strategy. Output as ordinal or one-hot. |
| `Binarizer` | Threshold-based binary indicator. |

## Categorical encoders

| Transformer | What it does |
|---|---|
| `OneHotEncoder` | One-hot with `drop_first` and `handle_unknown` options. |
| `OrdinalEncoder` | Integer encoding with per-column category lookup. |
| `LabelEncoder` | Single-column label to integer. |
| `LabelBinarizer` | Binary indicator per unique label. |
| `MultiLabelBinarizer` | Multi-label indicator matrix. |
| `TargetEncoder` | Smoothed target-mean encoding of categorical features. |

## Feature construction

| Transformer | What it does |
|---|---|
| `PolynomialFeatures` | Graded-lex polynomial expansion up to configurable degree, with `interaction_only` and `include_bias`. |
| `SplineTransformer` | B-spline basis expansion with configurable degree and knot placement. |
| `FunctionTransformer` | Wrap an arbitrary closure as a pipeline transformer. |

## Example

```rust
use solow_preprocessing::StandardScaler;

let mut sc = StandardScaler::new();
sc.fit(&x)?;
let x_std = sc.transform(&x)?;
let x_back = sc.inverse_transform(&x_std)?;
# Ok::<_, solow_core::Error>(())
```

## Correctness

Every transformer is cross-verified against committed golden reference fixtures on every CI run. Inverse transforms round-trip exact to machine precision on well-scaled data.
