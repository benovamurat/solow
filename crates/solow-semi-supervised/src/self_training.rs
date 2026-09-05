//! Self-training wrapper — the reference `SelfTrainingClassifier`.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// The trait a base classifier must satisfy to be wrappable by
/// [`SelfTrainingClassifier`].
pub trait BaseClassifier {
    /// Fit on `(x, y)`; both arrays are fully-labelled here.
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[i64]) -> Result<()>;

    /// Return a per-row probability vector over the `n_classes` classes
    /// (columns follow the sorted label order at last `fit`).
    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>>;

    /// The sorted labels seen at the last `fit`.
    fn classes(&self) -> Vec<i64>;
}

/// Selection criterion for adding an unlabelled sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelfTrainingCriterion {
    /// Add the top-`k_best` unlabelled samples per iteration, ranked by
    /// their maximum predicted probability.
    KBest(usize),
    /// Add every unlabelled sample whose max probability exceeds `threshold`.
    Threshold(f64),
}

/// Fitted self-training classifier.
pub struct SelfTrainingClassifier<C: BaseClassifier> {
    /// Wrapped base classifier (trained on the enlarged labelled set).
    pub base: C,
    /// Final labels for the whole training set (`-1` if never enlarged).
    pub labeled_iter: Array1<i64>,
    /// Iterations run.
    pub n_iter: usize,
    /// Selection criterion used.
    pub criterion: SelfTrainingCriterion,
}

impl<C: BaseClassifier> SelfTrainingClassifier<C> {
    /// Fit.
    pub fn fit(
        mut base: C,
        x: ArrayView2<'_, f64>,
        y: &[i64],
        criterion: SelfTrainingCriterion,
        max_iter: usize,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n {
            return Err(Error::Shape(
                "SelfTrainingClassifier: y/x length mismatch".into(),
            ));
        }
        let mut current: Vec<i64> = y.to_vec();
        let mut iters_added = Array1::<i64>::from_elem(n, -1_i64);
        for i in 0..n {
            if current[i] >= 0 {
                iters_added[i] = 0;
            }
        }
        let mut used = 0_usize;
        for it in 1..=max_iter {
            used = it;
            // Fit on the labelled subset.
            let labelled_rows: Vec<usize> =
                (0..n).filter(|&i| current[i] >= 0).collect();
            let unlabelled_rows: Vec<usize> =
                (0..n).filter(|&i| current[i] < 0).collect();
            if labelled_rows.is_empty() {
                return Err(Error::Value(
                    "SelfTrainingClassifier: at least one labelled sample required".into(),
                ));
            }
            if unlabelled_rows.is_empty() {
                break;
            }
            let x_lab = row_subset(x, &labelled_rows);
            let y_lab: Vec<i64> = labelled_rows.iter().map(|&i| current[i]).collect();
            base.fit(x_lab.view(), &y_lab)?;
            let x_un = row_subset(x, &unlabelled_rows);
            let probs = base.predict_proba(x_un.view())?;
            let classes = base.classes();
            let mut per_row_max: Vec<(usize, f64, i64)> =
                Vec::with_capacity(unlabelled_rows.len());
            for (rr, &row) in unlabelled_rows.iter().enumerate() {
                let mut best = 0;
                let mut best_p = probs[[rr, 0]];
                for c in 1..probs.ncols() {
                    if probs[[rr, c]] > best_p {
                        best_p = probs[[rr, c]];
                        best = c;
                    }
                }
                per_row_max.push((row, best_p, classes[best]));
            }
            let to_add: Vec<usize> = match criterion {
                SelfTrainingCriterion::KBest(k) => {
                    let mut sorted = per_row_max.clone();
                    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    sorted.iter().take(k).map(|(row, _, _)| *row).collect()
                }
                SelfTrainingCriterion::Threshold(t) => per_row_max
                    .iter()
                    .filter(|(_, p, _)| *p >= t)
                    .map(|(row, _, _)| *row)
                    .collect(),
            };
            if to_add.is_empty() {
                break;
            }
            for row in to_add {
                if current[row] < 0 {
                    let (_, _, lbl) = per_row_max.iter().find(|(r, _, _)| *r == row).unwrap();
                    current[row] = *lbl;
                    iters_added[row] = it as i64;
                }
            }
        }
        // Final fit on everything we now have.
        let final_rows: Vec<usize> = (0..n).filter(|&i| current[i] >= 0).collect();
        let x_final = row_subset(x, &final_rows);
        let y_final: Vec<i64> = final_rows.iter().map(|&i| current[i]).collect();
        base.fit(x_final.view(), &y_final)?;
        Ok(Self {
            base,
            labeled_iter: iters_added,
            n_iter: used,
            criterion,
        })
    }
}

fn row_subset(x: ArrayView2<'_, f64>, rows: &[usize]) -> Array2<f64> {
    let p = x.ncols();
    let mut out = Array2::<f64>::zeros((rows.len(), p));
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..p {
            out[[r, j]] = x[[i, j]];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Toy 1-D distance-to-centroid classifier used only in tests.
    struct Toy {
        centroids: Vec<(i64, f64)>,
    }

    impl BaseClassifier for Toy {
        fn fit(&mut self, x: ArrayView2<'_, f64>, y: &[i64]) -> Result<()> {
            let mut sums: std::collections::BTreeMap<i64, (f64, usize)> = Default::default();
            for (i, &yi) in y.iter().enumerate() {
                let e = sums.entry(yi).or_insert((0.0, 0));
                e.0 += x[[i, 0]];
                e.1 += 1;
            }
            self.centroids = sums
                .into_iter()
                .map(|(k, (s, n))| (k, s / n as f64))
                .collect();
            Ok(())
        }

        fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
            let k = self.centroids.len();
            let mut out = Array2::<f64>::zeros((x.nrows(), k));
            for i in 0..x.nrows() {
                let mut dists: Vec<f64> = self.centroids.iter()
                    .map(|(_, c)| ((x[[i, 0]] - c).abs()))
                    .collect();
                // Turn distance into probability via -d then softmax.
                for d in dists.iter_mut() {
                    *d = (-*d).exp();
                }
                let z: f64 = dists.iter().sum::<f64>().max(1e-30);
                for c in 0..k {
                    out[[i, c]] = dists[c] / z;
                }
            }
            Ok(out)
        }

        fn classes(&self) -> Vec<i64> {
            self.centroids.iter().map(|(c, _)| *c).collect()
        }
    }

    #[test]
    fn self_training_labels_two_clusters_from_one_seed_each() {
        let x = array![
            [0.0_f64], [0.1], [0.2], [0.3],
            [5.0], [5.1], [5.2], [5.3]
        ];
        let y = vec![0_i64, -1, -1, -1, -1, -1, -1, 1];
        let toy = Toy { centroids: vec![] };
        let m = SelfTrainingClassifier::fit(
            toy,
            x.view(),
            &y,
            SelfTrainingCriterion::Threshold(0.5),
            10,
        ).unwrap();
        // Every unlabelled row should have been added.
        for i in 0..8 {
            assert!(m.labeled_iter[i] >= 0, "row {i} was never added");
        }
    }
}
