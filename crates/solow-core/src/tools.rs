//! Shared data-handling tools used across models.

use crate::error::{Error, Result};
use ndarray::{concatenate, Array2, ArrayView1, ArrayView2, Axis};

/// Validate that every element of a 1-D array is finite (no `NaN`, `+inf`, or `-inf`).
///
/// Returns `Err(Error::Value)` naming `what` on the first non-finite element,
/// so model entry points can reject bad input cleanly instead of panicking
/// downstream. For all-finite input this is a cheap linear scan and the data
/// is left untouched.
pub fn ensure_all_finite(a: &ArrayView1<f64>, what: &str) -> Result<()> {
    if a.iter().any(|v| !v.is_finite()) {
        return Err(Error::Value(format!(
            "{what} contains non-finite values (NaN or inf)"
        )));
    }
    Ok(())
}

/// Validate that every element of a 2-D array is finite (no `NaN`, `+inf`, or `-inf`).
///
/// The 2-D analogue of [`ensure_all_finite`]; see that function for details.
pub fn ensure_all_finite_2d(a: &ArrayView2<f64>, what: &str) -> Result<()> {
    if a.iter().any(|v| !v.is_finite()) {
        return Err(Error::Value(format!(
            "{what} contains non-finite values (NaN or inf)"
        )));
    }
    Ok(())
}

/// Policy for [`add_constant`] when the data already contains a constant column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HasConstant {
    /// Return the data unchanged (the default behavior).
    #[default]
    Skip,
    /// Add a constant column anyway (produces a rank-deficient design matrix).
    Add,
    /// Return an error.
    Raise,
}

/// Returns `true` if column `j` of `x` is (numerically) constant.
fn column_is_constant(x: &Array2<f64>, j: usize) -> bool {
    let col = x.column(j);
    match col.iter().next() {
        None => false,
        Some(&first) => col.iter().all(|&v| v == first),
    }
}

/// Add a column of ones to an array.
///
/// Mirrors the reference `add_constant`: by default, if the data already contains
/// a constant column, the data is returned unchanged ([`HasConstant::Skip`]).
///
/// * `prepend` — if `true`, the constant column is the first column; otherwise the last.
pub fn add_constant(
    x: &Array2<f64>,
    prepend: bool,
    has_constant: HasConstant,
) -> Result<Array2<f64>> {
    let (n, k) = x.dim();

    let already = (0..k).any(|j| column_is_constant(x, j));
    if already {
        match has_constant {
            HasConstant::Skip => return Ok(x.clone()),
            HasConstant::Raise => {
                return Err(Error::Value(
                    "add_constant: data already contains a constant column".into(),
                ))
            }
            HasConstant::Add => {}
        }
    }

    let ones = Array2::<f64>::ones((n, 1));
    let out = if prepend {
        concatenate(Axis(1), &[ones.view(), x.view()])
    } else {
        concatenate(Axis(1), &[x.view(), ones.view()])
    }
    .map_err(|e| Error::Shape(format!("add_constant: {e}")))?;
    Ok(out)
}

/// Convenience wrapper: prepend a constant column, skipping if one already exists
/// (matches the most common reference default).
pub fn add_constant_default(x: &Array2<f64>) -> Result<Array2<f64>> {
    add_constant(x, true, HasConstant::Skip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn prepends_ones() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let g = add_constant(&x, true, HasConstant::Add).unwrap();
        assert_eq!(g.dim(), (2, 3));
        assert_eq!(g.column(0).to_vec(), vec![1.0, 1.0]);
        assert_eq!(g.column(1).to_vec(), vec![1.0, 3.0]);
    }

    #[test]
    fn appends_ones() {
        let x = array![[2.0], [4.0], [6.0]];
        let g = add_constant(&x, false, HasConstant::Add).unwrap();
        assert_eq!(g.dim(), (3, 2));
        assert_eq!(g.column(1).to_vec(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn skips_existing_constant() {
        let x = array![[1.0, 2.0], [1.0, 4.0]];
        let g = add_constant(&x, true, HasConstant::Skip).unwrap();
        assert_eq!(g.dim(), (2, 2)); // unchanged
    }

    #[test]
    fn finite_check_accepts_finite() {
        let v = array![1.0, -2.0, 3.5, 0.0];
        assert!(ensure_all_finite(&v.view(), "endog").is_ok());
        let m = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(ensure_all_finite_2d(&m.view(), "exog").is_ok());
    }

    #[test]
    fn finite_check_rejects_nan_and_inf() {
        let nan = array![1.0, f64::NAN, 3.0];
        assert!(matches!(
            ensure_all_finite(&nan.view(), "endog"),
            Err(Error::Value(_))
        ));
        let inf = array![1.0, f64::INFINITY];
        assert!(matches!(
            ensure_all_finite(&inf.view(), "weights"),
            Err(Error::Value(_))
        ));
        let neg_inf = array![[1.0, 2.0], [f64::NEG_INFINITY, 4.0]];
        assert!(matches!(
            ensure_all_finite_2d(&neg_inf.view(), "exog"),
            Err(Error::Value(_))
        ));
    }
}
