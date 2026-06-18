# solow-formula

An R/patsy-style *formula* interface that turns a formula string plus named
data into a numeric design matrix with column names, matching
[patsy](https://patsy.readthedocs.io/) semantics.

```
use solow_formula::{DataFrame, build};

let mut df = DataFrame::new();
df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]);
df.add_numeric("x", vec![0.0, 1.0, 2.0, 3.0]);
let out = build("y ~ x", &df).unwrap();
assert_eq!(out.names, vec!["Intercept".to_string(), "x".to_string()]);
assert!(out.y.is_some());
```

Supported right-hand-side syntax:

* `+` add a term, `-` remove a term (`- 1` / `+ 0` drop the intercept),
* `:` interaction, `*` full cross (`a*b == a + b + a:b`),
* `/` nesting (`a/b == a + a:b`), `**` interaction power
  (`(a + b + c)**2` = all ≤2-way interactions), and parenthesized groups,
* `C(var)` mark a column categorical, optionally with a contrast coding:
  `C(g, Poly)`, `C(g, Sum)`, `C(g, Helmert)`, `C(g, Diff)`
  (orthogonal polynomial, deviation/sum-to-zero, Helmert, and
  backward-difference coding; the bare `C(g)` is treatment/dummy coding),
* `I(expr)` identity / arithmetic transform (`**`, `*`, `+`, `-`, unary `-`)
  over a numeric column and numeric literals.

Categorical factors are coded with structural avoidance of redundancy: with
an intercept (or a lower-order term) spanning the constant direction the
reduced `k-1`-column contrast is used, and where a full-rank encoding is
needed the full `k`-column matrix (carrying the constant/mean column) is
emitted — matching patsy's `code_without_intercept` / `code_with_intercept`.

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-formula) · License: BSD-3-Clause
