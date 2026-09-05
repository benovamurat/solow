//! Design-matrix construction: the patsy coding pipeline.
//!
//! The flow mirrors patsy's `build.py`:
//!
//! 1. parse the formula into terms,
//! 2. bucket terms by their set of *numeric* factors (the empty bucket first,
//!    other buckets in first-appearance order), and within a bucket sort
//!    stably by interaction degree,
//! 3. for each term pick categorical contrast codings that avoid structural
//!    redundancy ([`crate::redundancy::pick_contrasts_for_term`]),
//! 4. expand each subterm into concrete columns with patsy's column ordering
//!    (left-most factor iterates fastest), and
//! 5. evaluate the columns over the data.

use crate::ast::{Expr, Factor};
use crate::contrasts::Coding;
use crate::data::DataFrame;
use crate::eval::eval_column;
use crate::parser::parse;
use crate::redundancy::{pick_contrasts_for_term, UsedSubterms};
use crate::terms::Term;
use ndarray::{Array1, Array2};
use solow_core::{Error, Result};
use std::collections::BTreeSet;

/// The result of [`build`].
#[derive(Debug, Clone)]
pub struct DesignOutput {
    /// The response column, if the formula had a `~` left-hand side.
    pub y: Option<Array1<f64>>,
    /// The design matrix (rows = observations, columns = model terms).
    pub design: Array2<f64>,
    /// Column names, in patsy order.
    pub names: Vec<String>,
}

/// Build a design matrix from a formula string and a [`DataFrame`].
///
/// Returns the (optional) response vector, the numeric design matrix, and the
/// column names — matching patsy's `dmatrices` output exactly.
pub fn build(formula: &str, data: &DataFrame) -> Result<DesignOutput> {
    let parsed = parse(formula)?;
    let nrows = data
        .nrows()
        .ok_or_else(|| Error::Value("data frame has no columns".into()))?;
    check_lengths(data, nrows)?;

    let y = match &parsed.response {
        Some(f) => Some(eval_response(f, data, nrows)?),
        None => None,
    };

    // Full ordered term list (intercept first if present).
    let mut terms: Vec<Term> = Vec::new();
    if parsed.rhs.intercept {
        terms.push(Term::intercept());
    }
    terms.extend(parsed.rhs.terms.iter().cloned());

    for t in &terms {
        for f in &t.factors {
            check_factor(f, data)?;
        }
    }

    let ordered = order_terms(&terms, data);

    let mut columns: Vec<Vec<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    build_all(&ordered, data, nrows, &mut columns, &mut names)?;

    let ncols = columns.len();
    let mut flat = Vec::with_capacity(nrows * ncols);
    for row in 0..nrows {
        for col in &columns {
            flat.push(col[row]);
        }
    }
    let design = Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| Error::Shape(format!("design matrix assembly: {e}")))?;

    Ok(DesignOutput { y, design, names })
}

// --------------------------------------------------------------------------
// Validation
// --------------------------------------------------------------------------

fn check_lengths(data: &DataFrame, nrows: usize) -> Result<()> {
    for name in &data.order {
        let len = if let Some(c) = data.numeric_col(name) {
            c.len()
        } else if let Some(c) = data.categorical_col(name) {
            c.len()
        } else {
            continue;
        };
        if len != nrows {
            return Err(Error::Shape(format!(
                "column `{name}` has length {len}, expected {nrows}"
            )));
        }
    }
    Ok(())
}

fn check_factor(f: &Factor, data: &DataFrame) -> Result<()> {
    match f {
        Factor::Numeric(n) => {
            if data.is_categorical(n) {
                return Err(Error::Value(format!(
                    "`{n}` is categorical; reference it as C({n})"
                )));
            }
            if !data.has(n) {
                return Err(Error::Value(format!("unknown variable `{n}`")));
            }
        }
        Factor::Categorical { var, .. } => {
            if !data.has(var) {
                return Err(Error::Value(format!("unknown variable `{var}`")));
            }
        }
        Factor::Identity { expr, .. } => check_expr_vars(expr, data)?,
    }
    Ok(())
}

fn check_expr_vars(expr: &Expr, data: &DataFrame) -> Result<()> {
    match expr {
        Expr::Num(_) => Ok(()),
        Expr::Var(n) => {
            if data.numeric_col(n).is_none() {
                Err(Error::Value(format!(
                    "I(...): `{n}` is not a numeric column"
                )))
            } else {
                Ok(())
            }
        }
        Expr::Neg(a) => check_expr_vars(a, data),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            check_expr_vars(a, data)?;
            check_expr_vars(b, data)
        }
    }
}

fn eval_response(f: &Factor, data: &DataFrame, nrows: usize) -> Result<Array1<f64>> {
    match f {
        Factor::Numeric(n) => {
            let col = data
                .numeric_col(n)
                .ok_or_else(|| Error::Value(format!("response `{n}` is not numeric")))?;
            Ok(Array1::from(col.to_vec()))
        }
        Factor::Identity { expr, .. } => Ok(Array1::from(eval_column(expr, data, nrows)?)),
        Factor::Categorical { var, .. } => Err(Error::NotImplemented(format!(
            "categorical response C({var}) is not supported"
        ))),
    }
}

// --------------------------------------------------------------------------
// Term ordering (patsy buckets)
// --------------------------------------------------------------------------

fn is_numeric_factor(f: &Factor) -> bool {
    matches!(f, Factor::Numeric(_) | Factor::Identity { .. })
}

fn numeric_key(t: &Term) -> BTreeSet<String> {
    t.factors
        .iter()
        .filter(|f| is_numeric_factor(f))
        .map(|f| f.key())
        .collect()
}

/// patsy term ordering: bucket by the set of numeric factors (empty bucket
/// first, others in first-appearance order), then within a bucket sort stably
/// by interaction degree.
fn order_terms(terms: &[Term], _data: &DataFrame) -> Vec<Term> {
    let mut bucket_order: Vec<BTreeSet<String>> = Vec::new();
    for t in terms {
        let k = numeric_key(t);
        if !bucket_order.contains(&k) {
            bucket_order.push(k);
        }
    }
    let empty: BTreeSet<String> = BTreeSet::new();
    if let Some(pos) = bucket_order.iter().position(|k| k == &empty) {
        let e = bucket_order.remove(pos);
        bucket_order.insert(0, e);
    }

    let mut out = Vec::new();
    for bucket in &bucket_order {
        let mut bterms: Vec<Term> = terms
            .iter()
            .filter(|t| &numeric_key(t) == bucket)
            .cloned()
            .collect();
        bterms.sort_by_key(|t| t.factors.len()); // stable, by degree
        out.extend(bterms);
    }
    out
}

// --------------------------------------------------------------------------
// Column construction
// --------------------------------------------------------------------------

fn build_all(
    ordered: &[Term],
    data: &DataFrame,
    nrows: usize,
    columns: &mut Vec<Vec<f64>>,
    names: &mut Vec<String>,
) -> Result<()> {
    // Walk contiguous buckets (already grouped by `order_terms`); the
    // redundancy state is shared within a bucket.
    let mut i = 0;
    while i < ordered.len() {
        let key = numeric_key(&ordered[i]);
        let start = i;
        while i < ordered.len() && numeric_key(&ordered[i]) == key {
            i += 1;
        }
        let mut used = UsedSubterms::default();
        for term in &ordered[start..i] {
            emit_term(term, data, nrows, &mut used, columns, names)?;
        }
    }
    Ok(())
}

/// Emit all columns of one term, consulting and updating the shared `used` set.
fn emit_term(
    term: &Term,
    data: &DataFrame,
    nrows: usize,
    used: &mut UsedSubterms,
    columns: &mut Vec<Vec<f64>>,
    names: &mut Vec<String>,
) -> Result<()> {
    // Categorical factor keys in term order (used for the redundancy lattice).
    let cat_keys: Vec<String> = term
        .factors
        .iter()
        .filter(|f| matches!(f, Factor::Categorical { .. }))
        .map(|f| f.key())
        .collect();

    // One coding per subterm; each maps cat-key -> full(bool).
    let codings = pick_contrasts_for_term(&cat_keys, used);

    for coding in &codings {
        // Build the subterm by walking the term's factors in order. Numeric
        // factors join every subterm; a categorical factor joins only if the
        // coding includes it (patsy: `elif factor in factor_coding`).
        let mut specs: Vec<FactorSpec> = Vec::new();
        for f in &term.factors {
            match f {
                Factor::Categorical { var, contrast, .. } => {
                    let Some(&full) = coding.get(&f.key()) else {
                        continue;
                    };
                    let col = data.categorical_col(var).ok_or_else(|| {
                        Error::Value(format!("C({var}): `{var}` is not categorical"))
                    })?;
                    let levels = sorted_levels(col);
                    let coding = contrast.coding(&levels, full);
                    specs.push(FactorSpec::Categorical {
                        label: f.label(),
                        rows: col.to_vec(),
                        levels,
                        coding,
                    });
                }
                _ => specs.push(FactorSpec::Numeric {
                    label: f.label(),
                    values: numeric_values(f, data, nrows)?,
                }),
            }
        }
        expand_subterm(&specs, nrows, columns, names);
    }
    Ok(())
}

enum FactorSpec {
    Numeric {
        label: String,
        values: Vec<f64>,
    },
    Categorical {
        label: String,
        rows: Vec<String>,
        levels: Vec<String>,
        coding: Coding,
    },
}

impl FactorSpec {
    fn width(&self) -> usize {
        match self {
            FactorSpec::Numeric { .. } => 1,
            FactorSpec::Categorical { coding, .. } => coding.suffixes.len(),
        }
    }

    /// (column suffix, per-row values) for the `j`-th column of this factor.
    fn column(&self, j: usize, nrows: usize) -> (String, Vec<f64>) {
        match self {
            FactorSpec::Numeric { label, values } => (label.clone(), values.clone()),
            FactorSpec::Categorical {
                label,
                rows,
                levels,
                coding,
            } => {
                // Each row's value is the contrast-matrix entry for that row's
                // level in column `j`; the suffix comes straight from the coding.
                let suffix = format!("{label}{}", coding.suffixes[j]);
                let mut v = vec![0.0; nrows];
                for (i, val) in v.iter_mut().enumerate() {
                    let level_idx = levels.iter().position(|l| l == &rows[i]);
                    if let Some(li) = level_idx {
                        *val = coding.matrix[li][j];
                    }
                }
                (suffix, v)
            }
        }
    }
}

/// Expand a subterm into design columns. Combination order matches patsy/R:
/// the left-most factor iterates fastest.
fn expand_subterm(
    specs: &[FactorSpec],
    nrows: usize,
    columns: &mut Vec<Vec<f64>>,
    names: &mut Vec<String>,
) {
    let widths: Vec<usize> = specs.iter().map(FactorSpec::width).collect();
    if widths.contains(&0) {
        return;
    }
    for combo in column_combinations(&widths) {
        let mut pieces: Vec<String> = Vec::new();
        let mut col = vec![1.0f64; nrows];
        for (fi, &j) in combo.iter().enumerate() {
            let (suffix, values) = specs[fi].column(j, nrows);
            pieces.push(suffix);
            for (c, x) in col.iter_mut().zip(values.iter()) {
                *c *= x;
            }
        }
        names.push(if pieces.is_empty() {
            "Intercept".to_string()
        } else {
            pieces.join(":")
        });
        columns.push(col);
    }
}

/// Index combinations across factors with the left-most factor iterating
/// fastest.
fn column_combinations(widths: &[usize]) -> Vec<Vec<usize>> {
    if widths.is_empty() {
        return vec![vec![]];
    }
    let total: usize = widths.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; widths.len()];
    for _ in 0..total {
        out.push(idx.clone());
        let mut k = 0;
        loop {
            idx[k] += 1;
            if idx[k] < widths[k] {
                break;
            }
            idx[k] = 0;
            k += 1;
            if k == widths.len() {
                break;
            }
        }
    }
    out
}

fn sorted_levels(col: &[String]) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for v in col {
        set.insert(v.clone());
    }
    set.into_iter().collect()
}

fn numeric_values(f: &Factor, data: &DataFrame, nrows: usize) -> Result<Vec<f64>> {
    match f {
        Factor::Numeric(n) => {
            let c = data
                .numeric_col(n)
                .ok_or_else(|| Error::Value(format!("`{n}` is not numeric")))?;
            Ok(c.to_vec())
        }
        Factor::Identity { expr, .. } => eval_column(expr, data, nrows),
        Factor::Categorical { .. } => Err(Error::Value("numeric_values on categorical".into())),
    }
}
