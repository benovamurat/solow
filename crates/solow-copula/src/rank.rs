//! Sample rank correlations for paired data.
//!
//! [`kendalls_tau`] returns the tau-b statistic (with the tie correction
//! used by `scipy.stats.kendalltau`), and [`spearmans_rho`] returns the
//! Pearson correlation of the (tie-averaged) ranks, matching
//! `scipy.stats.spearmanr`.

/// Kendall's tau-b rank correlation coefficient for paired data.
///
/// Counts concordant and discordant pairs and applies the tau-b tie
/// correction:
///
/// `tau_b = (n_c - n_d) / sqrt((n0 - n1)(n0 - n2))`,
///
/// where `n0 = n(n-1)/2`, `n1 = sum_i t_i(t_i-1)/2` over tie groups in `x`,
/// and `n2 = sum_j u_j(u_j-1)/2` over tie groups in `y`.
///
/// # Panics
/// Panics if the inputs have different lengths.
pub fn kendalls_tau(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "kendalls_tau: length mismatch");
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }

    let mut nc = 0i64; // concordant
    let mut nd = 0i64; // discordant
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let s = dx.signum() * dy.signum();
            if dx != 0.0 && dy != 0.0 {
                if s > 0.0 {
                    nc += 1;
                } else {
                    nd += 1;
                }
            }
        }
    }

    let n0 = (n * (n - 1) / 2) as f64;
    let n1 = tie_term(x);
    let n2 = tie_term(y);

    let num = (nc - nd) as f64;
    let den = ((n0 - n1) * (n0 - n2)).sqrt();
    if den == 0.0 {
        return f64::NAN;
    }
    num / den
}

/// Sum over tie groups of `t(t-1)/2` for the values in `v`.
fn tie_term(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let mut total = 0.0;
    let mut i = 0;
    while i < s.len() {
        let mut j = i + 1;
        while j < s.len() && s[j] == s[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        total += t * (t - 1.0) / 2.0;
        i = j;
    }
    total
}

/// Spearman's rank correlation `rho` for paired data.
///
/// Computes the Pearson correlation of the fractional ranks (ties receive
/// the average of their ranks), matching `scipy.stats.spearmanr`.
///
/// # Panics
/// Panics if the inputs have different lengths.
pub fn spearmans_rho(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "spearmans_rho: length mismatch");
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let rx = ranks(x);
    let ry = ranks(y);
    pearson(&rx, &ry)
}

/// Tie-averaged ranks (1-based) for the values in `v`.
fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // Average rank for the tie group spanning sorted positions [i, j).
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of (i+1)..=j
        for &k in &idx[i..j] {
            r[k] = avg;
        }
        i = j;
    }
    r
}

/// Pearson correlation of two equal-length slices.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    let den = (saa * sbb).sqrt();
    if den == 0.0 {
        return f64::NAN;
    }
    sab / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kendall_perfect_concordance() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [10.0, 20.0, 30.0, 40.0];
        assert!((kendalls_tau(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kendall_perfect_discordance() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [40.0, 30.0, 20.0, 10.0];
        assert!((kendalls_tau(&x, &y) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn spearman_perfect_concordance() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((spearmans_rho(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ranks_handle_ties() {
        // values: 3,1,3,2 -> sorted ranks: 1->1, 2->2, 3->avg(3,4)=3.5
        let r = ranks(&[3.0, 1.0, 3.0, 2.0]);
        assert_eq!(r, vec![3.5, 1.0, 3.5, 2.0]);
    }
}
