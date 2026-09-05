//! Sequential preprocessing / estimator pipeline.
//!
//! Each [`Step`] is a boxed closure that takes the current matrix
//! (rows = samples, cols = features) and returns the next matrix.
//! The final step returns the same matrix shape a pipeline consumer
//! would predict against; downstream scorers close over `y` themselves.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// One pipeline step.
pub struct Step {
    /// Human-readable identifier — surfaces in error messages and
    /// pipeline-cloning diagnostics.
    pub name: String,
    /// The step's transform function.
    pub transform: Box<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>>,
}

impl Step {
    /// Build a new pipeline step.
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

/// Sequential pipeline of steps.
pub struct Pipeline {
    /// Ordered pipeline steps.
    pub steps: Vec<Step>,
}

impl Pipeline {
    /// Empty pipeline — chain steps with [`Pipeline::then`].
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a step.
    pub fn then(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    /// Apply every step in order.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if self.steps.is_empty() {
            return Err(Error::Value("Pipeline: at least one step required".into()));
        }
        let mut current = x.to_owned();
        for step in &self.steps {
            current = (step.transform)(current.view())?;
        }
        Ok(current)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_step_pipeline_composes_transforms() {
        let p = Pipeline::new()
            .then(Step::new("plus_one", |x| Ok(x.mapv(|v| v + 1.0))))
            .then(Step::new("times_two", |x| Ok(x.mapv(|v| v * 2.0))));
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let out = p.transform(x.view()).unwrap();
        // ((1+1)*2, (2+1)*2), ((3+1)*2, (4+1)*2) = (4, 6), (8, 10).
        assert_eq!(out, array![[4.0, 6.0], [8.0, 10.0]]);
    }

    #[test]
    fn empty_pipeline_errors() {
        let p = Pipeline::new();
        let x = array![[1.0]];
        assert!(p.transform(x.view()).is_err());
    }
}
