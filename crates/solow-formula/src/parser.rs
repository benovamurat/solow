//! Hand-rolled tokenizer and recursive-descent parser for formulas.
//!
//! Two grammars live here. The *formula* grammar covers the term algebra
//! (`~ + - * :` plus the `C(...)`/`I(...)` factor forms). The *arithmetic*
//! grammar covers the contents of `I(...)` (`+ - * / **` and unary minus).

use crate::ast::{Expr, Factor};
use crate::contrasts::ContrastKind;
use crate::terms::{Term, TermList};
use solow_core::{Error, Result};

/// Parsed formula: an optional response term-set and the right-hand-side terms.
pub(crate) struct ParsedFormula {
    pub response: Option<Factor>,
    pub rhs: TermList,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Plus,
    Minus,
    Star,
    Colon,
    Tilde,
    LParen,
    RParen,
    Slash,
    Pow,
    Comma,
    Ident(String),
    Num(f64),
}

fn err(msg: impl Into<String>) -> Error {
    Error::Value(format!("formula parse error: {}", msg.into()))
}

/// Tokenize a formula string. Whitespace separates tokens but is otherwise
/// insignificant.
fn tokenize(src: &str) -> Result<Vec<Tok>> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                i += 1;
            }
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            ':' => {
                toks.push(Tok::Colon);
                i += 1;
            }
            '~' => {
                toks.push(Tok::Tilde);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            '/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    toks.push(Tok::Pow);
                    i += 2;
                } else {
                    toks.push(Tok::Star);
                    i += 1;
                }
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E'
                        || ((chars[i] == '+' || chars[i] == '-')
                            && i > start
                            && (chars[i - 1] == 'e' || chars[i - 1] == 'E')))
                {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let v: f64 = s.parse().map_err(|_| err(format!("bad number `{s}`")))?;
                toks.push(Tok::Num(v));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                toks.push(Tok::Ident(s));
            }
            other => return Err(err(format!("unexpected character `{other}`"))),
        }
    }
    Ok(toks)
}

struct Cursor {
    toks: Vec<Tok>,
    pos: usize,
}

impl Cursor {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

/// Parse a full formula string into a [`ParsedFormula`].
pub(crate) fn parse(src: &str) -> Result<ParsedFormula> {
    let toks = tokenize(src)?;
    let mut cur = Cursor { toks, pos: 0 };

    // Optional `response ~`. Detect a `~` anywhere at top level.
    let has_tilde = cur.toks.contains(&Tok::Tilde);
    let response = if has_tilde {
        // Parse a single factor as the response (patsy allows expressions, but
        // our scope is a single column / transform on the LHS).
        let f = parse_factor(&mut cur)?;
        if !cur.eat(&Tok::Tilde) {
            return Err(err("expected `~` after response"));
        }
        Some(f)
    } else {
        None
    };

    let rhs = parse_term_list(&mut cur)?;

    if cur.pos != cur.toks.len() {
        return Err(err(format!("trailing tokens after position {}", cur.pos)));
    }
    Ok(ParsedFormula { response, rhs })
}

/// Parse the `+`/`-` separated term list (lowest precedence on the RHS).
///
/// An implicit intercept is present unless explicitly removed (`- 1` / `+ 0`).
fn parse_term_list(cur: &mut Cursor) -> Result<TermList> {
    let mut list = TermList::with_intercept();

    // Leading sign handling: a formula may start with `-1`, `+ x`, `0`, etc.
    let mut sign_add = true;
    if cur.eat(&Tok::Plus) {
        sign_add = true;
    } else if cur.eat(&Tok::Minus) {
        sign_add = false;
    }

    loop {
        // Special numeric tokens `0` and `1` adjust the intercept.
        if let Some(Tok::Num(n)) = cur.peek().cloned() {
            if n == 1.0 || n == 0.0 {
                cur.next();
                apply_intercept(&mut list, n, sign_add);
                if !advance_sep(cur, &mut sign_add)? {
                    break;
                }
                continue;
            }
        }

        let product = parse_product(cur)?;
        if sign_add {
            for t in product {
                list.add(t);
            }
        } else {
            for t in product {
                list.remove(&t);
            }
        }

        if !advance_sep(cur, &mut sign_add)? {
            break;
        }
    }

    Ok(list)
}

fn apply_intercept(list: &mut TermList, n: f64, sign_add: bool) {
    // `+ 1` / `- 0` -> keep intercept; `- 1` / `+ 0` -> drop it.
    let want_intercept = if n == 1.0 { sign_add } else { !sign_add };
    if want_intercept {
        list.set_intercept(true);
    } else {
        list.set_intercept(false);
    }
}

/// Consume a `+`/`-` separator, updating `sign_add`. Returns `false` at the end
/// of the term list (no separator, or end of input).
fn advance_sep(cur: &mut Cursor, sign_add: &mut bool) -> Result<bool> {
    match cur.peek() {
        Some(Tok::Plus) => {
            cur.next();
            *sign_add = true;
            Ok(true)
        }
        Some(Tok::Minus) => {
            cur.next();
            *sign_add = false;
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Parse a `*`/`/`-separated product (precedence 200). `a*b` expands to the full
/// cross `a + b + a:b`; `a/b` nests as `a + (joint of a):b`.
fn parse_product(cur: &mut Cursor) -> Result<Vec<Term>> {
    let mut acc = parse_interaction(cur)?; // Vec<Term> from one operand
    loop {
        if cur.eat(&Tok::Star) {
            let rhs = parse_interaction(cur)?;
            acc = full_cross(&acc, &rhs);
        } else if cur.eat(&Tok::Slash) {
            let rhs = parse_interaction(cur)?;
            acc = nest(&acc, &rhs);
        } else {
            break;
        }
    }
    Ok(acc)
}

/// Full cross of two term-sets: every term of `a`, every term of `b`, and every
/// pairwise interaction, in patsy's canonical order (a-terms, b-terms,
/// interactions). Mirrors patsy `_eval_binary_prod`.
fn full_cross(a: &[Term], b: &[Term]) -> Vec<Term> {
    let mut out: Vec<Term> = Vec::new();
    for t in a {
        push_unique(&mut out, t.clone());
    }
    for t in b {
        push_unique(&mut out, t.clone());
    }
    out.extend(interaction(a, b));
    dedup_terms(out)
}

/// Nesting `a/b` (patsy `_eval_binary_div`): keep `a`'s terms, then interact a
/// *single* combined term holding every factor that appears anywhere in `a`
/// with each term of `b`.
fn nest(a: &[Term], b: &[Term]) -> Vec<Term> {
    let mut out: Vec<Term> = a.to_vec();
    let mut combined = Term::intercept();
    for t in a {
        combined = combined.interact(t);
    }
    out.extend(interaction(std::slice::from_ref(&combined), b));
    dedup_terms(out)
}

/// Cross-product interaction of two term-sets (patsy `_interaction`): for every
/// left term and right term, the union of their factors.
fn interaction(a: &[Term], b: &[Term]) -> Vec<Term> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for la in a {
        for lb in b {
            out.push(la.interact(lb));
        }
    }
    out
}

fn push_unique(v: &mut Vec<Term>, t: Term) {
    if !v.iter().any(|x| x == &t) {
        v.push(t);
    }
}

fn dedup_terms(terms: Vec<Term>) -> Vec<Term> {
    let mut out: Vec<Term> = Vec::with_capacity(terms.len());
    for t in terms {
        push_unique(&mut out, t);
    }
    out
}

/// Parse a `:`-separated interaction (precedence 300). `:` is a term-set
/// operation: it distributes over `+`, so `a:(b + c)` yields `a:b + a:c`.
fn parse_interaction(cur: &mut Cursor) -> Result<Vec<Term>> {
    let mut acc = parse_power(cur)?;
    while cur.eat(&Tok::Colon) {
        let rhs = parse_power(cur)?;
        acc = dedup_terms(interaction(&acc, &rhs));
    }
    Ok(acc)
}

/// Parse the interaction-power operator `**` (precedence 500, tightest binary).
/// `(a + b + c)**n` keeps all interactions of degree up to `n`, exactly as
/// patsy iterates `_interaction(left, big)` `n-1` times.
fn parse_power(cur: &mut Cursor) -> Result<Vec<Term>> {
    let base = parse_term_atom(cur)?;
    if cur.eat(&Tok::Pow) {
        let n = match cur.next() {
            Some(Tok::Num(v)) if v.fract() == 0.0 && v >= 1.0 => v as usize,
            other => {
                return Err(err(format!(
                    "`**` in a term requires a positive integer power, found {other:?}"
                )))
            }
        };
        Ok(power(&base, n))
    } else {
        Ok(base)
    }
}

/// Interaction power: `left ** n`. Patsy caps `n` at `len(left.terms)` and then
/// accumulates `left`, `left:left`, ... up to the n-th self-interaction.
fn power(left: &[Term], n: usize) -> Vec<Term> {
    let cap = n.min(left.len());
    let mut all: Vec<Term> = left.to_vec();
    let mut big: Vec<Term> = left.to_vec();
    for _ in 1..cap {
        big = dedup_terms(interaction(left, &big));
        all.extend(big.iter().cloned());
    }
    dedup_terms(all)
}

/// Parse a term *atom*: a single factor, or a parenthesized sub-expression
/// (a `+`/`-` separated term group). Parenthesized groups let `**`, `/`, and
/// `:` apply to whole sub-formulas, e.g. `(a + b + c)**2`.
fn parse_term_atom(cur: &mut Cursor) -> Result<Vec<Term>> {
    if cur.eat(&Tok::LParen) {
        let terms = parse_group(cur)?;
        expect(cur, &Tok::RParen)?;
        return Ok(terms);
    }
    Ok(vec![Term::from_factor(parse_factor(cur)?)])
}

/// Parse a parenthesized `+`/`-` separated group into a flat term-set. The
/// intercept is not tracked inside a group (patsy folds the group's intercept
/// into the surrounding expression); a leading/embedded `1`/`0` literal is
/// therefore ignored at the term-set level here.
fn parse_group(cur: &mut Cursor) -> Result<Vec<Term>> {
    let mut sign_add = true;
    if cur.eat(&Tok::Plus) {
        sign_add = true;
    } else if cur.eat(&Tok::Minus) {
        sign_add = false;
    }
    let mut acc: Vec<Term> = Vec::new();
    loop {
        if let Some(Tok::Num(v)) = cur.peek().cloned() {
            if v == 0.0 || v == 1.0 {
                // Intercept-adjusting literal inside a group: skip it at the
                // term-set level (the constant direction is managed outside).
                cur.next();
                if !skip_sep(cur, &mut sign_add) {
                    break;
                }
                continue;
            }
        }
        let product = parse_product(cur)?;
        if sign_add {
            for t in product {
                push_unique(&mut acc, t);
            }
        } else {
            for t in &product {
                acc.retain(|x| x != t);
            }
        }
        if !skip_sep(cur, &mut sign_add) {
            break;
        }
    }
    Ok(acc)
}

fn skip_sep(cur: &mut Cursor, sign_add: &mut bool) -> bool {
    match cur.peek() {
        Some(Tok::Plus) => {
            cur.next();
            *sign_add = true;
            true
        }
        Some(Tok::Minus) => {
            cur.next();
            *sign_add = false;
            true
        }
        _ => false,
    }
}

/// Parse a single factor: `C(var)`, `C(var, <Code>)`, `I(expr)`, or a numeric
/// identifier.
fn parse_factor(cur: &mut Cursor) -> Result<Factor> {
    match cur.peek().cloned() {
        Some(Tok::Ident(name)) => {
            cur.next();
            if name == "C" {
                expect(cur, &Tok::LParen)?;
                let inner = match cur.next() {
                    Some(Tok::Ident(v)) => v,
                    _ => return Err(err("C(...) expects a variable name")),
                };
                // Optional `, <ContrastCode>`. `arg` preserves the literal
                // spelling for the column label.
                let (contrast, arg) = if cur.eat(&Tok::Comma) {
                    let code = match cur.next() {
                        Some(Tok::Ident(c)) => c,
                        _ => return Err(err("C(var, ...) expects a contrast name")),
                    };
                    let kind = ContrastKind::from_name(&code).ok_or_else(|| {
                        err(format!("unknown contrast coding `{code}` in C(...)"))
                    })?;
                    (kind, Some(code))
                } else {
                    (ContrastKind::Treatment, None)
                };
                expect(cur, &Tok::RParen)?;
                Ok(Factor::Categorical {
                    var: inner,
                    contrast,
                    arg,
                })
            } else if name == "I" {
                expect(cur, &Tok::LParen)?;
                let expr = parse_expr(cur)?;
                expect(cur, &Tok::RParen)?;
                Ok(Factor::Identity {
                    code: format!("I({})", render_expr(&expr)),
                    expr,
                })
            } else {
                Ok(Factor::Numeric(name))
            }
        }
        other => Err(err(format!("expected a factor, found {other:?}"))),
    }
}

fn expect(cur: &mut Cursor, t: &Tok) -> Result<()> {
    if cur.eat(t) {
        Ok(())
    } else {
        Err(err(format!("expected {t:?}, found {:?}", cur.peek())))
    }
}

// ---------------------------------------------------------------------------
// Arithmetic grammar for `I(...)`.
// expr  := add
// add   := mul (('+'|'-') mul)*
// mul   := unary (('*'|'/') unary)*
// unary := '-' unary | pow
// pow   := atom ('**' unary)?            (right-associative)
// atom  := NUM | IDENT | '(' add ')'
// ---------------------------------------------------------------------------

fn parse_expr(cur: &mut Cursor) -> Result<Expr> {
    parse_add(cur)
}

fn parse_add(cur: &mut Cursor) -> Result<Expr> {
    let mut lhs = parse_mul(cur)?;
    loop {
        if cur.eat(&Tok::Plus) {
            let rhs = parse_mul(cur)?;
            lhs = Expr::Add(Box::new(lhs), Box::new(rhs));
        } else if cur.eat(&Tok::Minus) {
            let rhs = parse_mul(cur)?;
            lhs = Expr::Sub(Box::new(lhs), Box::new(rhs));
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_mul(cur: &mut Cursor) -> Result<Expr> {
    let mut lhs = parse_unary(cur)?;
    loop {
        if cur.eat(&Tok::Star) {
            let rhs = parse_unary(cur)?;
            lhs = Expr::Mul(Box::new(lhs), Box::new(rhs));
        } else if cur.eat(&Tok::Slash) {
            let rhs = parse_unary(cur)?;
            lhs = Expr::Div(Box::new(lhs), Box::new(rhs));
        } else {
            break;
        }
    }
    Ok(lhs)
}

fn parse_unary(cur: &mut Cursor) -> Result<Expr> {
    if cur.eat(&Tok::Minus) {
        let e = parse_unary(cur)?;
        return Ok(Expr::Neg(Box::new(e)));
    }
    if cur.eat(&Tok::Plus) {
        return parse_unary(cur);
    }
    parse_pow(cur)
}

fn parse_pow(cur: &mut Cursor) -> Result<Expr> {
    let base = parse_atom(cur)?;
    if cur.eat(&Tok::Pow) {
        // Right-associative; the exponent may itself be a unary expression.
        let exp = parse_unary(cur)?;
        Ok(Expr::Pow(Box::new(base), Box::new(exp)))
    } else {
        Ok(base)
    }
}

fn parse_atom(cur: &mut Cursor) -> Result<Expr> {
    match cur.next() {
        Some(Tok::Num(n)) => Ok(Expr::Num(n)),
        Some(Tok::Ident(name)) => Ok(Expr::Var(name)),
        Some(Tok::LParen) => {
            let e = parse_add(cur)?;
            expect(cur, &Tok::RParen)?;
            Ok(e)
        }
        other => Err(err(format!("expected an arithmetic atom, found {other:?}"))),
    }
}

/// Render an arithmetic expression in patsy's canonical spelling (operators
/// padded with single spaces, e.g. `x1 ** 2`).
fn render_expr(e: &Expr) -> String {
    match e {
        Expr::Num(n) => render_num(*n),
        Expr::Var(v) => v.clone(),
        Expr::Neg(a) => format!("-{}", render_expr(a)),
        Expr::Add(a, b) => format!("{} + {}", render_expr(a), render_expr(b)),
        Expr::Sub(a, b) => format!("{} - {}", render_expr(a), render_expr(b)),
        Expr::Mul(a, b) => format!("{} * {}", render_expr(a), render_expr(b)),
        Expr::Div(a, b) => format!("{} / {}", render_expr(a), render_expr(b)),
        Expr::Pow(a, b) => format!("{} ** {}", render_expr(a), render_expr(b)),
    }
}

/// Format a numeric literal the way Python prints it inside a formula string:
/// integers without a decimal point, others via the default float repr.
fn render_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}
