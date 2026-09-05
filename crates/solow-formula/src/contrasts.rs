//! Categorical *contrast codings* — the matrices patsy uses to turn a factor's
//! `k` levels into numeric columns.
//!
//! Each coding offers two forms, mirroring patsy's `ContrastMatrix` pair:
//!
//! * **reduced** (`code_without_intercept`): the rank-deficient `k × (k-1)`
//!   contrast used when an intercept (or a lower-order term) already spans the
//!   constant direction, and
//! * **full** (`code_with_intercept`): the full-rank `k × k` matrix that also
//!   carries the constant/mean/intercept column, used where patsy needs full
//!   rank (e.g. `0 + C(g, Sum)`).
//!
//! The four extra codings reproduce R's `contr.poly` / `contr.sum` /
//! `contr.helmert` and patsy's backward-difference coding exactly:
//!
//! | kind        | meaning                                   | R equivalent     |
//! |-------------|-------------------------------------------|------------------|
//! | `Treatment` | dummy coding against a reference level    | `contr.treatment`|
//! | `Poly`      | orthogonal polynomial coding              | `contr.poly`     |
//! | `Sum`       | deviation / sum-to-zero coding            | `contr.sum`      |
//! | `Helmert`   | each level vs. the mean of preceding ones | `contr.helmert`  |
//! | `Diff`      | backward difference coding                | (patsy `Diff`)   |
//!
//! Column *suffixes* (the bracketed bits patsy appends to the factor label) are
//! produced alongside each matrix so the design columns get their exact names.

/// The contrast coding selected for a categorical factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContrastKind {
    /// patsy's default dummy coding (`C(g)` / `C(g, Treatment)`).
    Treatment,
    /// Orthogonal polynomial coding (`C(g, Poly)`).
    Poly,
    /// Deviation / sum-to-zero coding (`C(g, Sum)`).
    Sum,
    /// Helmert coding (`C(g, Helmert)`).
    Helmert,
    /// Backward-difference coding (`C(g, Diff)`).
    Diff,
}

impl ContrastKind {
    /// Parse the coding name as it appears inside `C(var, <name>)`.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "Treatment" => Some(ContrastKind::Treatment),
            "Poly" => Some(ContrastKind::Poly),
            "Sum" => Some(ContrastKind::Sum),
            "Helmert" => Some(ContrastKind::Helmert),
            "Diff" => Some(ContrastKind::Diff),
            _ => None,
        }
    }
}

/// A coding matrix plus the per-column suffixes patsy appends to the factor
/// label. `matrix` is row-major, `levels.len()` rows by `suffixes.len()`
/// columns: row `i` gives the contrast values for the `i`-th level.
pub(crate) struct Coding {
    pub matrix: Vec<Vec<f64>>,
    pub suffixes: Vec<String>,
}

impl ContrastKind {
    /// Build the coding for `levels`. `full` selects the full-rank
    /// (`code_with_intercept`) form; otherwise the reduced `k × (k-1)` contrast.
    pub(crate) fn coding(&self, levels: &[String], full: bool) -> Coding {
        match self {
            ContrastKind::Treatment => treatment(levels, full),
            ContrastKind::Poly => poly(levels, full),
            ContrastKind::Sum => sum(levels, full),
            ContrastKind::Helmert => helmert(levels, full),
            ContrastKind::Diff => diff(levels, full),
        }
    }
}

// ---------------------------------------------------------------------------
// Treatment (dummy) coding — patsy's default.
// ---------------------------------------------------------------------------

fn treatment(levels: &[String], full: bool) -> Coding {
    let k = levels.len();
    if full {
        // Full coding: one indicator per level, suffix `[level]`.
        let mut matrix = vec![vec![0.0; k]; k];
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let suffixes = levels.iter().map(|l| format!("[{l}]")).collect();
        Coding { matrix, suffixes }
    } else {
        // Reduced: drop the first (reference) level, suffix `[T.level]`.
        let mut matrix = vec![vec![0.0; k - 1]; k];
        for (i, row) in matrix.iter_mut().enumerate().skip(1) {
            row[i - 1] = 1.0;
        }
        let suffixes = levels[1..].iter().map(|l| format!("[T.{l}]")).collect();
        Coding { matrix, suffixes }
    }
}

// ---------------------------------------------------------------------------
// Sum (deviation) coding — R's contr.sum.
// ---------------------------------------------------------------------------

// These contrast builders index 2-D matrices by explicit row/column position to
// mirror the reference (numpy) constructions one-for-one; iterator rewrites
// would obscure the diagonal / triangular index patterns.
#[allow(clippy::needless_range_loop)]
fn sum_contrast(k: usize) -> Vec<Vec<f64>> {
    // Omit the last level (patsy's default). `out` is k x (k-1): the identity
    // for the kept levels, and -1 across the omitted (last) row.
    let omit = k - 1;
    let mut out = vec![vec![0.0; k - 1]; k];
    for i in 0..omit {
        out[i][i] = 1.0;
    }
    for c in out[omit].iter_mut() {
        *c = -1.0;
    }
    out
}

fn sum(levels: &[String], full: bool) -> Coding {
    let k = levels.len();
    let contrast = sum_contrast(k);
    // Included levels are all but the omitted (last) one.
    let included = &levels[..k - 1];
    if full {
        let mut matrix = vec![Vec::with_capacity(k); k];
        for (i, row) in matrix.iter_mut().enumerate() {
            row.push(1.0);
            row.extend_from_slice(&contrast[i]);
        }
        let mut suffixes = vec!["[mean]".to_string()];
        suffixes.extend(included.iter().map(|l| format!("[S.{l}]")));
        Coding { matrix, suffixes }
    } else {
        let suffixes = included.iter().map(|l| format!("[S.{l}]")).collect();
        Coding {
            matrix: contrast,
            suffixes,
        }
    }
}

// ---------------------------------------------------------------------------
// Helmert coding — R's contr.helmert.
// ---------------------------------------------------------------------------

#[allow(clippy::needless_range_loop)]
fn helmert_contrast(k: usize) -> Vec<Vec<f64>> {
    // patsy "r-like" construction:
    //   contr = zeros((k, k-1))
    //   contr[1:][diag] = arange(1, k)      # subdiagonal block diagonal
    //   contr[triu_indices(k-1)] = -1       # upper triangle (rows 0..k-1)
    let mut contr = vec![vec![0.0; k - 1]; k];
    // contr[1:][np.diag_indices(k-1)]: for j in 0..k-1, row (j+1), col j = j+1.
    for j in 0..k - 1 {
        contr[j + 1][j] = (j + 1) as f64;
    }
    // contr[np.triu_indices(k-1)] = -1 over the (k-1)x(k-1) top block: every
    // (r, c) with r <= c in rows 0..k-1.
    for r in 0..k - 1 {
        for c in r..k - 1 {
            contr[r][c] = -1.0;
        }
    }
    contr
}

fn helmert(levels: &[String], full: bool) -> Coding {
    let k = levels.len();
    let contrast = helmert_contrast(k);
    if full {
        let mut matrix = vec![Vec::with_capacity(k); k];
        for (i, row) in matrix.iter_mut().enumerate() {
            row.push(1.0);
            row.extend_from_slice(&contrast[i]);
        }
        let mut suffixes = vec!["[H.intercept]".to_string()];
        suffixes.extend(levels[1..].iter().map(|l| format!("[H.{l}]")));
        Coding { matrix, suffixes }
    } else {
        let suffixes = levels[1..].iter().map(|l| format!("[H.{l}]")).collect();
        Coding {
            matrix: contrast,
            suffixes,
        }
    }
}

// ---------------------------------------------------------------------------
// Diff (backward difference) coding.
// ---------------------------------------------------------------------------

#[allow(clippy::needless_range_loop)]
fn diff_contrast(k: usize) -> Vec<Vec<f64>> {
    // Port of patsy `_diff_contrast`. For an n=k level factor:
    //   upper triangle (row <= col) of the top (k-1)x(k-1) block gets
    //     (upper_int - k)/k where upper_int = repeat(1..k, 1..k) read column-wise
    //   lower triangle (row >= col), shifted down one row, gets
    //     lower_int/k where lower_int = repeat(1..k, (1..k) reversed) column-wise
    let n = k;
    let mut contr = vec![vec![0.0; n - 1]; n];

    // int_range = 1..n   (length n-1)
    // upper_int = repeat(int_range, int_range): value v repeated v times.
    // Filled down the columns of triu_indices(n-1).
    let mut upper_int: Vec<f64> = Vec::new();
    for v in 1..n {
        for _ in 0..v {
            upper_int.push(v as f64);
        }
    }
    // Column-wise upper-triangle coordinates (row <= col) of an (n-1)x(n-1)
    // matrix: for each col c, rows 0..=c.
    let mut idx = 0usize;
    for c in 0..n - 1 {
        for r in 0..=c {
            contr[r][c] = (upper_int[idx] - n as f64) / n as f64;
            idx += 1;
        }
    }

    // lower_int = repeat(int_range, reverse(int_range)).
    let int_range: Vec<usize> = (1..n).collect();
    let rev: Vec<usize> = int_range.iter().rev().cloned().collect();
    let mut lower_int: Vec<f64> = Vec::new();
    for (v, &times) in int_range.iter().zip(rev.iter()) {
        for _ in 0..times {
            lower_int.push(*v as f64);
        }
    }
    // Column-wise lower-triangle coordinates (row >= col), then shifted +1 row.
    let mut jdx = 0usize;
    for c in 0..n - 1 {
        for r in c..n - 1 {
            contr[r + 1][c] = lower_int[jdx] / n as f64;
            jdx += 1;
        }
    }
    contr
}

fn diff(levels: &[String], full: bool) -> Coding {
    let k = levels.len();
    let contrast = diff_contrast(k);
    if full {
        let mut matrix = vec![Vec::with_capacity(k); k];
        for (i, row) in matrix.iter_mut().enumerate() {
            row.push(1.0);
            row.extend_from_slice(&contrast[i]);
        }
        let suffixes = levels.iter().map(|l| format!("[D.{l}]")).collect();
        Coding { matrix, suffixes }
    } else {
        let suffixes = levels[..k - 1].iter().map(|l| format!("[D.{l}]")).collect();
        Coding {
            matrix: contrast,
            suffixes,
        }
    }
}

// ---------------------------------------------------------------------------
// Poly (orthogonal polynomial) coding — R's contr.poly.
// ---------------------------------------------------------------------------

#[allow(clippy::needless_range_loop)]
fn poly(levels: &[String], full: bool) -> Coding {
    let n = levels.len();
    // scores = 0..n, centered.
    let mean = (n as f64 - 1.0) / 2.0;
    let scores: Vec<f64> = (0..n).map(|i| i as f64 - mean).collect();

    // Vandermonde: raw_poly[i][j] = scores[i] ** j.
    let mut raw = vec![vec![0.0; n]; n];
    for (i, &s) in scores.iter().enumerate() {
        let mut p = 1.0;
        for j in 0..n {
            raw[i][j] = p;
            p *= s;
        }
    }

    // QR via modified Gram-Schmidt, then canonicalize column signs to match
    // numpy's `q *= sign(diag(r))` (which forces R's diagonal positive, making
    // the decomposition unique). Q columns are already orthonormal, so patsy's
    // per-row renormalization is a no-op and is omitted.
    let (mut q, rdiag) = mgs_qr(&raw, n);
    for j in 0..n {
        let s = if rdiag[j] < 0.0 { -1.0 } else { 1.0 };
        if s < 0.0 {
            for row in q.iter_mut() {
                row[j] = -row[j];
            }
        }
    }
    // The constant column is always all 1's.
    for row in q.iter_mut() {
        row[0] = 1.0;
    }

    let names = poly_suffixes(n);
    if full {
        Coding {
            matrix: q,
            suffixes: names,
        }
    } else {
        // Drop the constant column and its name.
        let matrix = q
            .into_iter()
            .map(|row| row[1..].to_vec())
            .collect::<Vec<_>>();
        Coding {
            matrix,
            suffixes: names[1..].to_vec(),
        }
    }
}

/// patsy's Poly column names: `.Constant`, `.Linear`, `.Quadratic`, `.Cubic`,
/// then `^4`, `^5`, ... for higher orders.
fn poly_suffixes(n: usize) -> Vec<String> {
    let base = [".Constant", ".Linear", ".Quadratic", ".Cubic"];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i < base.len() {
            out.push(base[i].to_string());
        } else {
            out.push(format!("^{i}"));
        }
    }
    out
}

/// Modified Gram-Schmidt QR of an `n × n` matrix. Returns `Q` (row-major) and
/// the diagonal of `R` (sufficient for the sign canonicalization).
fn mgs_qr(a: &[Vec<f64>], n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    // Work on columns: v[j] starts as column j of A.
    let cols: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| a[i][j]).collect::<Vec<f64>>())
        .collect();
    let mut q_cols: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    let mut rdiag = vec![0.0; n];

    for j in 0..n {
        let mut v = cols[j].clone();
        for q_prev in q_cols.iter().take(j) {
            // r_ij = q_i . a_j (project original column, matching patsy/numpy).
            let r_ij: f64 = (0..n).map(|i| q_prev[i] * cols[j][i]).sum();
            for i in 0..n {
                v[i] -= r_ij * q_prev[i];
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        rdiag[j] = norm;
        for i in 0..n {
            q_cols[j][i] = v[i] / norm;
        }
    }

    // Reassemble Q as row-major.
    let mut q = vec![vec![0.0; n]; n];
    for j in 0..n {
        for i in 0..n {
            q[i][j] = q_cols[j][i];
        }
    }
    (q, rdiag)
}
