# solow-polars

Bridge [Polars](https://www.pola.rs/) `DataFrame`s into the Solow statistical
stack. Polars is the workhorse DataFrame library of the Rust data ecosystem;
Solow provides the estimators. This crate is the glue: pull numeric columns out
of a `DataFrame` as [`ndarray`](https://docs.rs/ndarray) vectors / matrices and
fit a model in a single call.

It is its **own workspace** (note the empty `[workspace]` table in
`Cargo.toml`), so the heavy `polars` dependency never touches the core Solow
build. Build and test it on its own:

```sh
cargo test --manifest-path crates/solow-polars/Cargo.toml
```

## Fit a model straight from a Polars DataFrame

```rust
use polars::prelude::*;
use solow_polars::ols_from_frame;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Any Polars DataFrame — here built inline, but it could come from a CSV,
    // Parquet file, a SQL query, or a lazy pipeline `.collect()`.
    let df = df![
        "sales"   => [12.0_f64, 19.0, 23.0, 28.0, 35.0, 41.0],
        "ad_spend"=> [ 1.0_f64,  2.0,  3.0,  4.0,  5.0,  6.0],
        "price"   => [ 9.0_f64,  9.0,  8.0,  8.0,  7.0,  7.0],
    ]?;

    // Fit `sales ~ 1 + ad_spend + price` (an intercept is prepended for you).
    let res = ols_from_frame(&df, "sales", &["ad_spend", "price"], true)?;

    // `res` is a full `solow_regression::LinearResults`.
    println!("coefficients : {:?}", res.params);     // [intercept, b_ad_spend, b_price]
    println!("std. errors  : {:?}", res.bse);
    println!("R^2          : {:.4}", res.rsquared);
    println!("\n{}", res.summary(Some(&["const", "ad_spend", "price"])));
    Ok(())
}
```

## Generalized linear models

The same path works for GLMs — pass a `solow_glm::Family`:

```rust
use polars::prelude::*;
use solow_glm::Family;
use solow_polars::glm_from_frame;

let df = df![
    "incidents" => [1.0_f64, 3.0, 2.0, 5.0, 7.0, 11.0],
    "exposure"  => [0.0_f64, 1.0, 2.0, 3.0, 4.0,  5.0],
]
.unwrap();

// Poisson regression of a count response.
let res = glm_from_frame(&df, "incidents", &["exposure"], true, Family::Poisson).unwrap();
println!("{:?}", res.params);
```

## Lower-level conversions

If you only want the data, not a fitted model:

| function | from | to |
|---|---|---|
| `series_to_array1(&Series)` | one numeric column | `Array1<f64>` |
| `dataframe_to_array2(&DataFrame, &["a", "b"])` | selected numeric columns | `Array2<f64>` (design matrix) |

Both cast through `Float64`, so integer and boolean columns are accepted; columns
containing nulls or non-numeric data return a descriptive error instead of
silently producing `NaN`s, because the downstream estimators require finite
input.
