//! Input data container: named numeric and categorical columns.

use std::collections::HashMap;

/// A lightweight column-oriented data frame holding the variables a formula can
/// refer to.
///
/// Columns are either *numeric* (`Vec<f64>`) or *categorical* (`Vec<String>`).
/// All columns must share the same length; this is checked when the design
/// matrix is built.
#[derive(Debug, Clone, Default)]
pub struct DataFrame {
    pub(crate) numeric: HashMap<String, Vec<f64>>,
    pub(crate) categorical: HashMap<String, Vec<String>>,
    /// Insertion order is irrelevant to output (patsy orders by the formula),
    /// but we keep it for deterministic error messages.
    pub(crate) order: Vec<String>,
}

impl DataFrame {
    /// Create an empty frame.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a numeric column.
    pub fn add_numeric(&mut self, name: &str, values: Vec<f64>) {
        if !self.numeric.contains_key(name) && !self.categorical.contains_key(name) {
            self.order.push(name.to_string());
        }
        self.categorical.remove(name);
        self.numeric.insert(name.to_string(), values);
    }

    /// Insert (or replace) a categorical column.
    pub fn add_categorical<S: Into<String>>(&mut self, name: &str, values: Vec<S>) {
        if !self.numeric.contains_key(name) && !self.categorical.contains_key(name) {
            self.order.push(name.to_string());
        }
        self.numeric.remove(name);
        self.categorical.insert(
            name.to_string(),
            values.into_iter().map(Into::into).collect(),
        );
    }

    /// Number of observations (rows), or `None` if the frame is empty.
    pub fn nrows(&self) -> Option<usize> {
        self.numeric
            .values()
            .map(Vec::len)
            .chain(self.categorical.values().map(Vec::len))
            .next()
    }

    pub(crate) fn numeric_col(&self, name: &str) -> Option<&[f64]> {
        self.numeric.get(name).map(Vec::as_slice)
    }

    pub(crate) fn categorical_col(&self, name: &str) -> Option<&[String]> {
        self.categorical.get(name).map(Vec::as_slice)
    }

    pub(crate) fn is_categorical(&self, name: &str) -> bool {
        self.categorical.contains_key(name)
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.numeric.contains_key(name) || self.categorical.contains_key(name)
    }
}
