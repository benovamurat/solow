//! Abstract syntax for parsed formulas.

use crate::contrasts::ContrastKind;

/// A single *factor*: the atomic, indivisible piece a term is built from.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Factor {
    /// A bare numeric column reference, e.g. `x1`.
    Numeric(String),
    /// `C(var)` / `C(var, <Code>)`: treat `var` as categorical with the given
    /// contrast coding (treatment coding for the bare `C(var)` form). `arg`
    /// preserves the coding name exactly as written so the column label matches
    /// patsy's spelling (`C(g, Treatment)` keeps its `, Treatment`, but the bare
    /// `C(g)` form has `arg = None`).
    Categorical {
        var: String,
        contrast: ContrastKind,
        arg: Option<String>,
    },
    /// `I(expr)`: an identity transform evaluating an arithmetic expression to a
    /// single numeric column. `code` is the canonical patsy spelling used in the
    /// column name (e.g. `I(x1 ** 2)`).
    Identity { code: String, expr: Expr },
}

impl Factor {
    /// The label patsy prints for this factor inside a term/column name.
    pub(crate) fn label(&self) -> String {
        match self {
            Factor::Numeric(n) => n.clone(),
            Factor::Categorical { var, arg, .. } => match arg {
                None => format!("C({var})"),
                Some(a) => format!("C({var}, {a})"),
            },
            Factor::Identity { code, .. } => code.clone(),
        }
    }

    /// Stable key identifying the factor (used for de-duplication / sorting).
    pub(crate) fn key(&self) -> String {
        self.label()
    }
}

/// Arithmetic expression tree used inside `I(...)`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
}
