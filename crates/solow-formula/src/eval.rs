//! Evaluation of `I(...)` arithmetic expressions to a numeric column.

use crate::ast::Expr;
use crate::data::DataFrame;
use solow_core::{Error, Result};

/// Evaluate `expr` row-wise over `data`, producing one value per observation.
pub(crate) fn eval_column(expr: &Expr, data: &DataFrame, nrows: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(nrows);
    for row in 0..nrows {
        out.push(eval_at(expr, data, row)?);
    }
    Ok(out)
}

fn eval_at(expr: &Expr, data: &DataFrame, row: usize) -> Result<f64> {
    Ok(match expr {
        Expr::Num(n) => *n,
        Expr::Var(name) => {
            let col = data.numeric_col(name).ok_or_else(|| {
                Error::Value(format!(
                    "I(...): `{name}` is not a numeric column in the data"
                ))
            })?;
            col[row]
        }
        Expr::Neg(a) => -eval_at(a, data, row)?,
        Expr::Add(a, b) => eval_at(a, data, row)? + eval_at(b, data, row)?,
        Expr::Sub(a, b) => eval_at(a, data, row)? - eval_at(b, data, row)?,
        Expr::Mul(a, b) => eval_at(a, data, row)? * eval_at(b, data, row)?,
        Expr::Div(a, b) => eval_at(a, data, row)? / eval_at(b, data, row)?,
        Expr::Pow(a, b) => eval_at(a, data, row)?.powf(eval_at(b, data, row)?),
    })
}
