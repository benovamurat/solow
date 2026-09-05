//! [`ColumnTransformer`] and [`FeatureUnion`] — apply distinct
//! transforms to distinct column subsets and concatenate the outputs.

use ndarray::{Array2, ArrayView2, Axis};
use solow_core::{Error, Result};

/// A named transformer keyed to a column subset.
pub struct ColumnTransformerStep {
    /// Human-readable identifier for error messages.
    pub name: String,
    /// Columns of the input to route into this transformer (any order,
    /// no duplicates within a step; distinct steps may overlap).
    pub columns: Vec<usize>,
    /// The transform itself — takes the subset matrix and returns the
    /// transformed matrix.
    pub transform: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>,
}

impl ColumnTransformerStep {
    /// Build a new step.
    pub fn new<F>(name: impl Into<String>, columns: Vec<usize>, f: F) -> Self
    where
        F: Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>> + 'static,
    {
        Self {
            name: name.into(),
            columns,
            transform: Box::new(f),
        }
    }
}

/// Apply per-column-subset transforms and concatenate the outputs
/// horizontally — the reference `ColumnTransformer` shape.
pub struct ColumnTransformer {
    /// Ordered list of steps.
    pub steps: Vec<ColumnTransformerStep>,
    /// If `true` (default), columns not referenced by any step are
    /// dropped; if `false`, they are appended verbatim at the end
    /// (`remainder='passthrough'`).
    pub drop_remainder: bool,
}

impl ColumnTransformer {
    /// New — empty step list, `drop_remainder = true`.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            drop_remainder: true,
        }
    }

    /// Append a step.
    pub fn with_step(mut self, step: ColumnTransformerStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Set the remainder policy.
    pub fn passthrough(mut self, flag: bool) -> Self {
        self.drop_remainder = !flag;
        self
    }

    /// Apply every step and horizontally concatenate.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if self.steps.is_empty() && self.drop_remainder {
            return Err(Error::Value(
                "ColumnTransformer::transform: no steps and remainder dropped → zero-column output"
                    .into(),
            ));
        }
        let n = x.nrows();
        let mut parts: Vec<Array2<f64>> = Vec::with_capacity(self.steps.len() + 1);
        for step in &self.steps {
            for &c in &step.columns {
                if c >= x.ncols() {
                    return Err(Error::Value(format!(
                        "ColumnTransformer::transform: step '{}' references column {c} out of {}",
                        step.name,
                        x.ncols()
                    )));
                }
            }
            // Materialise the sub-matrix.
            let mut sub = Array2::<f64>::zeros((n, step.columns.len()));
            for (ci, &c) in step.columns.iter().enumerate() {
                for i in 0..n {
                    sub[[i, ci]] = x[[i, c]];
                }
            }
            let out = (step.transform)(sub.view())?;
            if out.nrows() != n {
                return Err(Error::Shape(format!(
                    "ColumnTransformer::transform: step '{}' returned {} rows for {n} inputs",
                    step.name,
                    out.nrows()
                )));
            }
            parts.push(out);
        }
        if !self.drop_remainder {
            // Append every column not referenced by any step, in ascending index order.
            let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for step in &self.steps {
                used.extend(&step.columns);
            }
            let remainder: Vec<usize> = (0..x.ncols()).filter(|c| !used.contains(c)).collect();
            if !remainder.is_empty() {
                let mut rest = Array2::<f64>::zeros((n, remainder.len()));
                for (ci, &c) in remainder.iter().enumerate() {
                    for i in 0..n {
                        rest[[i, ci]] = x[[i, c]];
                    }
                }
                parts.push(rest);
            }
        }
        concatenate_columns(&parts)
    }
}

impl Default for ColumnTransformer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FeatureUnion
// ---------------------------------------------------------------------------

/// One transformer in a [`FeatureUnion`].
pub struct FeatureUnionStep {
    /// Human-readable identifier.
    pub name: String,
    /// The transform.
    pub transform: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>,
}

impl FeatureUnionStep {
    /// Build a new step.
    pub fn new<F>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>> + 'static,
    {
        Self {
            name: name.into(),
            transform: Box::new(f),
        }
    }
}

/// Apply every step to the **entire** input matrix and horizontally
/// concatenate — the reference `FeatureUnion` shape.
pub struct FeatureUnion {
    /// Ordered list of steps.
    pub steps: Vec<FeatureUnionStep>,
}

impl FeatureUnion {
    /// New empty union.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a step.
    pub fn with_step(mut self, step: FeatureUnionStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Apply every step to `x` and concatenate.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if self.steps.is_empty() {
            return Err(Error::Value("FeatureUnion::transform: no steps".into()));
        }
        let n = x.nrows();
        let mut parts: Vec<Array2<f64>> = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let out = (step.transform)(x)?;
            if out.nrows() != n {
                return Err(Error::Shape(format!(
                    "FeatureUnion::transform: step '{}' returned {} rows for {n} inputs",
                    step.name,
                    out.nrows()
                )));
            }
            parts.push(out);
        }
        concatenate_columns(&parts)
    }
}

impl Default for FeatureUnion {
    fn default() -> Self {
        Self::new()
    }
}

fn concatenate_columns(parts: &[Array2<f64>]) -> Result<Array2<f64>> {
    if parts.is_empty() {
        return Err(Error::Value("concatenate_columns: no parts".into()));
    }
    let n = parts[0].nrows();
    let total_cols: usize = parts.iter().map(|p| p.ncols()).sum();
    let mut out = Array2::<f64>::zeros((n, total_cols));
    let mut col_off = 0usize;
    for part in parts {
        for i in 0..n {
            for j in 0..part.ncols() {
                out[[i, col_off + j]] = part[[i, j]];
            }
        }
        col_off += part.ncols();
    }
    // Silence unused-import.
    let _ = Axis(1);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn column_transformer_splits_and_recombines_columns() {
        // Column 0 doubled; column 1 identity.
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let ct = ColumnTransformer::new()
            .with_step(ColumnTransformerStep::new("double_col0", vec![0], |sub| {
                Ok(sub.mapv(|v| v * 2.0))
            }))
            .with_step(ColumnTransformerStep::new(
                "identity_col1",
                vec![1],
                |sub| Ok(sub.to_owned()),
            ));
        let out = ct.transform(x.view()).unwrap();
        assert_eq!(out, array![[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]]);
    }

    #[test]
    fn feature_union_stacks_step_outputs() {
        let x = array![[1.0], [2.0], [3.0]];
        let fu = FeatureUnion::new()
            .with_step(FeatureUnionStep::new("plus_one", |x| {
                Ok(x.mapv(|v| v + 1.0))
            }))
            .with_step(FeatureUnionStep::new("times_two", |x| {
                Ok(x.mapv(|v| v * 2.0))
            }));
        let out = fu.transform(x.view()).unwrap();
        assert_eq!(out, array![[2.0, 2.0], [3.0, 4.0], [4.0, 6.0]]);
    }
}
