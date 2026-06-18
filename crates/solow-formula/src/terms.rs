//! Terms and term lists: the algebra patsy operates on before coding.

use crate::ast::Factor;

/// A *term* is an ordered collection of distinct factors (an interaction).
/// The empty term denotes the intercept.
#[derive(Debug, Clone)]
pub(crate) struct Term {
    pub factors: Vec<Factor>,
}

impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        // Terms are equal when they contain the same *set* of factors,
        // irrespective of order (patsy treats `a:b` and `b:a` as one term).
        if self.factors.len() != other.factors.len() {
            return false;
        }
        let mut a: Vec<String> = self.factors.iter().map(Factor::key).collect();
        let mut b: Vec<String> = other.factors.iter().map(Factor::key).collect();
        a.sort();
        b.sort();
        a == b
    }
}
impl Eq for Term {}

impl Term {
    /// The intercept term (no factors).
    pub fn intercept() -> Self {
        Term { factors: vec![] }
    }

    pub fn from_factor(f: Factor) -> Self {
        Term { factors: vec![f] }
    }

    pub fn is_intercept(&self) -> bool {
        self.factors.is_empty()
    }

    /// Interaction of two terms: the de-duplicated union of their factors, in
    /// `self`-then-`other` appearance order (patsy keeps left-to-right order).
    pub fn interact(&self, other: &Term) -> Term {
        let mut factors = self.factors.clone();
        for f in &other.factors {
            if !factors.iter().any(|x| x.key() == f.key()) {
                factors.push(f.clone());
            }
        }
        Term { factors }
    }
}

/// An ordered list of terms plus the intercept flag.
#[derive(Debug, Clone)]
pub(crate) struct TermList {
    pub intercept: bool,
    /// Non-intercept terms, in formula order.
    pub terms: Vec<Term>,
}

impl TermList {
    pub fn with_intercept() -> Self {
        TermList {
            intercept: true,
            terms: vec![],
        }
    }

    pub fn set_intercept(&mut self, on: bool) {
        self.intercept = on;
    }

    /// Add a term, ignoring the intercept (tracked separately) and duplicates.
    pub fn add(&mut self, t: Term) {
        if t.is_intercept() {
            self.intercept = true;
            return;
        }
        if !self.terms.iter().any(|x| x == &t) {
            self.terms.push(t);
        }
    }

    /// Remove a term (or drop the intercept for the empty term).
    pub fn remove(&mut self, t: &Term) {
        if t.is_intercept() {
            self.intercept = false;
            return;
        }
        self.terms.retain(|x| x != t);
    }
}
