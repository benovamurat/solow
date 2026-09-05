# The formula interface

The [`solow-formula`] crate turns a formula string plus named data into a
numeric design matrix with column names, following
[patsy](https://patsy.readthedocs.io/) semantics. The output is validated
against patsy to `1e-12`, so a design matrix Solow builds is column-for-column
identical to what patsy produces.

You build a [`DataFrame`], add named columns, and call [`build`] with a formula.
The result is a [`DesignOutput`] with the optional response vector `y`, the
`design` matrix, and the column `names`.

## A minimal example

```rust
use solow_formula::{build, DataFrame};

let mut df = DataFrame::new();
df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]);
df.add_numeric("x", vec![0.0, 1.0, 2.0, 3.0]);

let out = build("y ~ x", &df).unwrap();

assert_eq!(out.names, vec!["Intercept".to_string(), "x".to_string()]);
assert!(out.y.is_some());            // the left-hand side became `y`
println!("design =\n{:?}", out.design);
```

An intercept named `Intercept` is included by default. The left-hand side of `~`
populates `out.y`; a formula with no `~` (just a right-hand side) leaves `out.y`
as `None` and returns only the design matrix.

## Right-hand-side syntax

| Syntax | Meaning |
| --- | --- |
| `a + b` | add terms |
| `a - b` | remove a term |
| `- 1` or `+ 0` | drop the intercept |
| `a:b` | interaction |
| `a*b` | full cross, i.e. `a + b + a:b` |
| `a/b` | nesting, i.e. `a + a:b` |
| `(a + b + c)**2` | all interactions up to order 2 |
| `C(g)` | mark `g` categorical (treatment/dummy coding) |
| `C(g, Poly)` / `C(g, Sum)` / `C(g, Helmert)` / `C(g, Diff)` | contrast codings |
| `I(expr)` | identity/arithmetic transform of numeric columns |

## Dropping the intercept

```rust
use solow_formula::{build, DataFrame};

let mut df = DataFrame::new();
df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]);
df.add_numeric("x", vec![0.0, 1.0, 2.0, 3.0]);

let out = build("y ~ x - 1", &df).unwrap();
assert_eq!(out.names, vec!["x".to_string()]); // no Intercept column
```

## Categorical predictors and contrasts

Use `add_categorical` for string-valued columns and wrap them in `C(...)` in the
formula. With an intercept present, a `k`-level factor expands to `k − 1`
contrast columns (treatment coding by default), exactly as patsy does.

```rust
use solow_formula::{build, DataFrame};

let mut df = DataFrame::new();
df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
df.add_categorical("g", vec!["a", "b", "c", "a", "b", "c"]);

let out = build("y ~ C(g)", &df).unwrap();
// Intercept + two treatment contrasts for the 3-level factor.
println!("names = {:?}", out.names);
```

Switch the coding scheme with the second argument, e.g. orthogonal polynomial
contrasts:

```rust
# use solow_formula::{build, DataFrame};
# let mut df = DataFrame::new();
# df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
# df.add_categorical("g", vec!["a", "b", "c", "a", "b", "c"]);
let out = build("y ~ C(g, Poly)", &df).unwrap();
println!("polynomial contrast names = {:?}", out.names);
```

## Interactions and transforms

```rust
use solow_formula::{build, DataFrame};

let mut df = DataFrame::new();
df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
df.add_numeric("x1", vec![0.0, 1.0, 2.0, 3.0, 4.0]);
df.add_numeric("x2", vec![1.0, 1.0, 0.0, 0.0, 1.0]);

// Main effects plus their interaction, and a squared term via I(...).
let out = build("y ~ x1 * x2 + I(x1 ** 2)", &df).unwrap();
println!("names = {:?}", out.names);
```

## Feeding the design into a model

The formula output plugs straight into an estimator. Because `build` already
includes the intercept column, you do **not** call `add_constant` afterwards:

```rust
use solow_formula::{build, DataFrame};
use solow_regression::LinearModel;

let mut df = DataFrame::new();
df.add_numeric("y", vec![2.1, 3.9, 6.2, 7.8, 10.1]);
df.add_numeric("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);

let out = build("y ~ x", &df).unwrap();
let y = out.y.expect("formula has a left-hand side");

let res = LinearModel::ols(y, out.design).unwrap().fit().unwrap();
println!("coefficients for {:?} = {:?}", out.names, res.params);
```

[`solow-formula`]: https://github.com/solow-rs/solow
[`DataFrame`]: ./formula.md
[`build`]: ./formula.md
[`DesignOutput`]: ./formula.md
