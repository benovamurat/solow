//! Cluster evaluation metrics — the reference `metrics.cluster`
//! surface.
//!
//! * [`adjusted_rand_score`] — Hubert-Arabie 1985 adjusted Rand index.
//! * [`normalized_mutual_info_score`] — MI normalised by the geometric
//!   or arithmetic mean of entropies.
//! * [`adjusted_mutual_info_score`] — MI corrected for chance (Vinh 2010).
//! * [`homogeneity_score`], [`completeness_score`], [`v_measure_score`]
//!   — Rosenberg-Hirschberg (2007) entropy-based scores.
//! * [`fowlkes_mallows_score`] — geometric mean of pairwise precision
//!   and recall.
//! * [`silhouette_score`] — Rousseeuw's silhouette coefficient.
//! * [`calinski_harabasz_score`] — variance ratio criterion.
//! * [`davies_bouldin_score`] — within/between separation.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Hubert-Arabie adjusted Rand index (`0` = random, `1` = identical).
pub fn adjusted_rand_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64> {
    let n = labels_true.len();
    if labels_pred.len() != n {
        return Err(Error::Shape("adjusted_rand_score: label vectors differ in length".into()));
    }
    if n < 2 {
        return Ok(0.0);
    }
    let (rows, cols, ct) = contingency_table(labels_true, labels_pred);
    let mut sum_c2 = 0.0_f64;
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            let v = ct[[i, j]];
            sum_c2 += v * (v - 1.0) / 2.0;
        }
    }
    let mut row_sum = vec![0.0_f64; rows.len()];
    let mut col_sum = vec![0.0_f64; cols.len()];
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            row_sum[i] += ct[[i, j]];
            col_sum[j] += ct[[i, j]];
        }
    }
    let sum_a2: f64 = row_sum.iter().map(|v| v * (v - 1.0) / 2.0).sum();
    let sum_b2: f64 = col_sum.iter().map(|v| v * (v - 1.0) / 2.0).sum();
    let n2 = (n as f64) * (n as f64 - 1.0) / 2.0;
    let expected = sum_a2 * sum_b2 / n2;
    let max = 0.5 * (sum_a2 + sum_b2);
    if (max - expected).abs() < 1e-30 {
        return Ok(1.0);
    }
    Ok((sum_c2 - expected) / (max - expected))
}

/// Normalised mutual information — `NMI = MI / mean_entropy`.
///
/// `avg` selects `Arith` (arithmetic mean, the reference default) or `Geom`
/// (geometric mean).
pub fn normalized_mutual_info_score(
    labels_true: &[i64],
    labels_pred: &[i64],
    avg: MiAverage,
) -> Result<f64> {
    let (mi, h_true, h_pred) = mutual_info_and_entropies(labels_true, labels_pred)?;
    let denom = match avg {
        MiAverage::Arith => 0.5 * (h_true + h_pred),
        MiAverage::Geom => (h_true * h_pred).sqrt(),
        MiAverage::Min => h_true.min(h_pred),
        MiAverage::Max => h_true.max(h_pred),
    };
    Ok((mi / denom.max(1e-30)).clamp(0.0, 1.0))
}

/// Adjusted mutual information — chance-corrected NMI (Vinh 2010).
pub fn adjusted_mutual_info_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64> {
    let (mi, h_true, h_pred) = mutual_info_and_entropies(labels_true, labels_pred)?;
    // Expected MI under a hypergeometric null.
    let (rows, cols, ct) = contingency_table(labels_true, labels_pred);
    let n = labels_true.len() as f64;
    let mut row_sum = vec![0.0_f64; rows.len()];
    let mut col_sum = vec![0.0_f64; cols.len()];
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            row_sum[i] += ct[[i, j]];
            col_sum[j] += ct[[i, j]];
        }
    }
    // Approximation via factorial-cancellation — sufficient for n ≤ 1000.
    let mut expected = 0.0_f64;
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            let a = row_sum[i];
            let b = col_sum[j];
            let start = (a + b - n).max(1.0) as usize;
            let end = a.min(b) as usize;
            let mut term_sum = 0.0_f64;
            for nij in start..=end {
                let nij_f = nij as f64;
                let numer = nij_f / n * ((nij_f * n / (a * b)).ln());
                let log_choose = ln_choose(a as usize, nij)
                    + ln_choose((n - a) as usize, (b as usize).saturating_sub(nij))
                    - ln_choose(n as usize, b as usize);
                term_sum += numer * log_choose.exp();
            }
            expected += term_sum;
        }
    }
    let denom = 0.5 * (h_true + h_pred) - expected;
    if denom.abs() < 1e-30 {
        return Ok(0.0);
    }
    Ok((mi - expected) / denom)
}

/// Homogeneity — `1 − H(true | pred) / H(true)`.
pub fn homogeneity_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64> {
    let (mi, h_true, _h_pred) = mutual_info_and_entropies(labels_true, labels_pred)?;
    Ok((mi / h_true.max(1e-30)).clamp(0.0, 1.0))
}

/// Completeness — `1 − H(pred | true) / H(pred)`.
pub fn completeness_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64> {
    let (mi, _h_true, h_pred) = mutual_info_and_entropies(labels_true, labels_pred)?;
    Ok((mi / h_pred.max(1e-30)).clamp(0.0, 1.0))
}

/// V-measure — harmonic mean of homogeneity and completeness.
pub fn v_measure_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64> {
    let h = homogeneity_score(labels_true, labels_pred)?;
    let c = completeness_score(labels_true, labels_pred)?;
    if h + c < 1e-30 {
        return Ok(0.0);
    }
    Ok(2.0 * h * c / (h + c))
}

/// Fowlkes-Mallows index.
pub fn fowlkes_mallows_score(labels_true: &[i64], labels_pred: &[i64]) -> Result<f64> {
    let (rows, cols, ct) = contingency_table(labels_true, labels_pred);
    let n = labels_true.len() as f64;
    let mut sum_c2 = 0.0_f64;
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            sum_c2 += ct[[i, j]] * (ct[[i, j]] - 1.0) / 2.0;
        }
    }
    let mut row_sum = vec![0.0_f64; rows.len()];
    let mut col_sum = vec![0.0_f64; cols.len()];
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            row_sum[i] += ct[[i, j]];
            col_sum[j] += ct[[i, j]];
        }
    }
    let a: f64 = row_sum.iter().map(|v| v * (v - 1.0) / 2.0).sum();
    let b: f64 = col_sum.iter().map(|v| v * (v - 1.0) / 2.0).sum();
    if a * b < 1e-30 || n < 2.0 {
        return Ok(0.0);
    }
    Ok(sum_c2 / (a * b).sqrt())
}

/// Silhouette coefficient on Euclidean distances.
pub fn silhouette_score(x: ArrayView2<'_, f64>, labels: &[i64]) -> Result<f64> {
    let n = x.nrows();
    if labels.len() != n {
        return Err(Error::Shape("silhouette_score: labels length mismatch".into()));
    }
    if n < 2 {
        return Err(Error::Value("silhouette_score: need ≥ 2 samples".into()));
    }
    let mut classes: Vec<i64> = labels.to_vec();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return Err(Error::Value("silhouette_score: need ≥ 2 clusters".into()));
    }
    let d = x.ncols();
    let mut score_sum = 0.0_f64;
    for i in 0..n {
        // Mean intra-cluster distance a(i) and lowest mean inter-cluster distance b(i).
        let mut a_sum = 0.0_f64;
        let mut a_count = 0_usize;
        let mut cluster_sums: std::collections::BTreeMap<i64, (f64, usize)> = Default::default();
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut dist = 0.0_f64;
            for k in 0..d {
                let e = x[[i, k]] - x[[j, k]];
                dist += e * e;
            }
            dist = dist.sqrt();
            if labels[j] == labels[i] {
                a_sum += dist;
                a_count += 1;
            } else {
                let e = cluster_sums.entry(labels[j]).or_insert((0.0, 0));
                e.0 += dist;
                e.1 += 1;
            }
        }
        let a = if a_count > 0 { a_sum / a_count as f64 } else { 0.0 };
        let b = cluster_sums
            .values()
            .filter(|(_, n)| *n > 0)
            .map(|(s, n)| s / *n as f64)
            .fold(f64::INFINITY, f64::min);
        let s = if a_count == 0 {
            0.0
        } else {
            (b - a) / a.max(b).max(1e-30)
        };
        score_sum += s;
    }
    Ok(score_sum / n as f64)
}

/// Calinski-Harabasz score — variance ratio criterion.
pub fn calinski_harabasz_score(x: ArrayView2<'_, f64>, labels: &[i64]) -> Result<f64> {
    let n = x.nrows();
    if labels.len() != n {
        return Err(Error::Shape("calinski_harabasz: labels length mismatch".into()));
    }
    let d = x.ncols();
    let mut classes: Vec<i64> = labels.to_vec();
    classes.sort();
    classes.dedup();
    let k = classes.len();
    if k < 2 || n < k + 1 {
        return Err(Error::Value("calinski_harabasz: need ≥ 2 clusters".into()));
    }
    let mut global_mean = vec![0.0_f64; d];
    for i in 0..n {
        for j in 0..d {
            global_mean[j] += x[[i, j]];
        }
    }
    for v in global_mean.iter_mut() {
        *v /= n as f64;
    }
    let mut cluster_means: std::collections::BTreeMap<i64, (Vec<f64>, usize)> = Default::default();
    for i in 0..n {
        let e = cluster_means.entry(labels[i]).or_insert((vec![0.0; d], 0));
        for j in 0..d {
            e.0[j] += x[[i, j]];
        }
        e.1 += 1;
    }
    let mut ss_between = 0.0_f64;
    for (_, (sum, count)) in &cluster_means {
        let mut diff2 = 0.0_f64;
        for j in 0..d {
            let m = sum[j] / *count as f64;
            let dd = m - global_mean[j];
            diff2 += dd * dd;
        }
        ss_between += *count as f64 * diff2;
    }
    let mut ss_within = 0.0_f64;
    for i in 0..n {
        let (sum, count) = &cluster_means[&labels[i]];
        for j in 0..d {
            let m = sum[j] / *count as f64;
            let dd = x[[i, j]] - m;
            ss_within += dd * dd;
        }
    }
    Ok(ss_between / (k - 1) as f64 / (ss_within / (n - k) as f64).max(1e-30))
}

/// Davies-Bouldin score — average within/between separation ratio.
pub fn davies_bouldin_score(x: ArrayView2<'_, f64>, labels: &[i64]) -> Result<f64> {
    let n = x.nrows();
    if labels.len() != n {
        return Err(Error::Shape("davies_bouldin: labels length mismatch".into()));
    }
    let d = x.ncols();
    let mut cluster_means: std::collections::BTreeMap<i64, (Vec<f64>, usize)> = Default::default();
    for i in 0..n {
        let e = cluster_means.entry(labels[i]).or_insert((vec![0.0; d], 0));
        for j in 0..d {
            e.0[j] += x[[i, j]];
        }
        e.1 += 1;
    }
    let means: Vec<(i64, Vec<f64>)> = cluster_means
        .iter()
        .map(|(k, (sum, count))| (*k, sum.iter().map(|v| v / *count as f64).collect()))
        .collect();
    if means.len() < 2 {
        return Err(Error::Value("davies_bouldin: need ≥ 2 clusters".into()));
    }
    // Within-cluster scatter s_i.
    let mut si = vec![0.0_f64; means.len()];
    for i in 0..n {
        let (_, mean) = means
            .iter()
            .find(|(c, _)| *c == labels[i])
            .unwrap();
        let mut dist = 0.0_f64;
        for j in 0..d {
            let e = x[[i, j]] - mean[j];
            dist += e * e;
        }
        let ci = means.iter().position(|(c, _)| *c == labels[i]).unwrap();
        si[ci] += dist.sqrt();
    }
    for i in 0..means.len() {
        let count = cluster_means[&means[i].0].1 as f64;
        si[i] /= count;
    }
    let mut db = 0.0_f64;
    for i in 0..means.len() {
        let mut best = f64::NEG_INFINITY;
        for j in 0..means.len() {
            if i == j {
                continue;
            }
            let mut m_dist = 0.0_f64;
            for k in 0..d {
                let dd = means[i].1[k] - means[j].1[k];
                m_dist += dd * dd;
            }
            let m_dist = m_dist.sqrt().max(1e-30);
            let r = (si[i] + si[j]) / m_dist;
            if r > best {
                best = r;
            }
        }
        db += best;
    }
    Ok(db / means.len() as f64)
}

/// Averaging strategy for [`normalized_mutual_info_score`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MiAverage {
    /// Arithmetic mean of entropies (the reference default).
    Arith,
    /// Geometric mean of entropies.
    Geom,
    /// Minimum entropy.
    Min,
    /// Maximum entropy.
    Max,
}

fn contingency_table(
    a: &[i64],
    b: &[i64],
) -> (Vec<i64>, Vec<i64>, Array2<f64>) {
    let mut rows: Vec<i64> = a.to_vec();
    rows.sort();
    rows.dedup();
    let mut cols: Vec<i64> = b.to_vec();
    cols.sort();
    cols.dedup();
    let mut ct = Array2::<f64>::zeros((rows.len(), cols.len()));
    for i in 0..a.len() {
        let ri = rows.binary_search(&a[i]).unwrap();
        let ci = cols.binary_search(&b[i]).unwrap();
        ct[[ri, ci]] += 1.0;
    }
    (rows, cols, ct)
}

fn mutual_info_and_entropies(a: &[i64], b: &[i64]) -> Result<(f64, f64, f64)> {
    let n = a.len();
    if b.len() != n {
        return Err(Error::Shape("mutual_info: length mismatch".into()));
    }
    let (rows, cols, ct) = contingency_table(a, b);
    let n_f = n as f64;
    let mut mi = 0.0_f64;
    let mut row_sum = vec![0.0_f64; rows.len()];
    let mut col_sum = vec![0.0_f64; cols.len()];
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            row_sum[i] += ct[[i, j]];
            col_sum[j] += ct[[i, j]];
        }
    }
    for i in 0..rows.len() {
        for j in 0..cols.len() {
            let nij = ct[[i, j]];
            if nij > 0.0 {
                mi += (nij / n_f) * ((nij * n_f) / (row_sum[i] * col_sum[j])).ln();
            }
        }
    }
    let h_a: f64 = row_sum.iter().filter(|&&v| v > 0.0).map(|&v| -(v / n_f) * (v / n_f).ln()).sum();
    let h_b: f64 = col_sum.iter().filter(|&&v| v > 0.0).map(|&v| -(v / n_f) * (v / n_f).ln()).sum();
    Ok((mi, h_a, h_b))
}

fn ln_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    (1..=k).map(|i| ((n - i + 1) as f64).ln() - (i as f64).ln()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn adjusted_rand_score_is_one_for_identical_labels() {
        let y = vec![0_i64, 0, 1, 1, 2, 2];
        let r = adjusted_rand_score(&y, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn homogeneity_completeness_v_measure_are_one_for_identical_labels() {
        let y = vec![0_i64, 0, 1, 1, 2, 2];
        assert!((homogeneity_score(&y, &y).unwrap() - 1.0).abs() < 1e-10);
        assert!((completeness_score(&y, &y).unwrap() - 1.0).abs() < 1e-10);
        assert!((v_measure_score(&y, &y).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn silhouette_score_is_high_on_well_separated_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [0.2, 0.2],
            [10.0, 10.0], [10.1, 10.1], [10.2, 10.2]
        ];
        let labels = vec![0_i64, 0, 0, 1, 1, 1];
        let s = silhouette_score(x.view(), &labels).unwrap();
        assert!(s > 0.9);
    }

    #[test]
    fn calinski_harabasz_is_positive_on_separated_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]
        ];
        let labels = vec![0_i64, 0, 1, 1];
        let s = calinski_harabasz_score(x.view(), &labels).unwrap();
        assert!(s > 0.0);
    }

    #[test]
    fn davies_bouldin_is_low_on_separated_clusters() {
        let x = array![
            [0.0_f64, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]
        ];
        let labels = vec![0_i64, 0, 1, 1];
        let s = davies_bouldin_score(x.view(), &labels).unwrap();
        assert!(s < 1.0);
    }
}
