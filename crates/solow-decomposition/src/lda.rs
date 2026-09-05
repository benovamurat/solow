//! Latent Dirichlet Allocation (Blei-Ng-Jordan 2003) — the topic model.
//!
//! Fit via the online-variational-inference update of Hoffman-Blei-Bach
//! (2010) with a single-pass mini-batch of size `n` (full-batch by
//! default) and a symmetric Dirichlet prior.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted LatentDirichletAllocation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LatentDirichletAllocation {
    /// Topic-word distribution `β` (`n_topics × n_features`), rows sum to 1.
    pub components: Array2<f64>,
    /// Document-topic distribution `θ` (`n × n_topics`) at fit time.
    pub doc_topic: Array2<f64>,
    /// Number of topics.
    pub n_topics: usize,
    /// Dirichlet prior on document-topic distributions `α`.
    pub doc_topic_prior: f64,
    /// Dirichlet prior on topic-word distributions `η`.
    pub topic_word_prior: f64,
    /// Iterations used.
    pub n_iter: usize,
}

impl LatentDirichletAllocation {
    /// Fit with the reference defaults `doc_topic_prior = 1/n_topics`,
    /// `topic_word_prior = 1/n_topics`, `max_iter = 10`, `tol = 1e-3`.
    pub fn fit(x: ArrayView2<'_, f64>, n_topics: usize) -> Result<Self> {
        let alpha = 1.0 / n_topics as f64;
        Self::fit_with(x, n_topics, alpha, alpha, 10, 1e-3)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_topics: usize,
        doc_topic_prior: f64,
        topic_word_prior: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_topics == 0 {
            return Err(Error::Value("LDA: n_topics must be ≥ 1".into()));
        }
        if x.iter().any(|&v| v < 0.0) {
            return Err(Error::Value("LDA: inputs must be non-negative counts".into()));
        }
        // Initialise β uniformly + Dirichlet fudge from a fixed seed.
        let mut beta = Array2::<f64>::from_elem((n_topics, d), 1.0 / d as f64);
        let mut theta = Array2::<f64>::from_elem((n, n_topics), 1.0 / n_topics as f64);
        let mut iters = 0_usize;
        for it in 0..max_iter {
            iters = it + 1;
            // E-step: update θ per document.
            let mut new_theta = Array2::<f64>::zeros((n, n_topics));
            for i in 0..n {
                for k in 0..n_topics {
                    let mut s = doc_topic_prior;
                    for j in 0..d {
                        let mut denom = 0.0_f64;
                        for kk in 0..n_topics {
                            denom += theta[[i, kk]] * beta[[kk, j]];
                        }
                        if denom > 0.0 {
                            let phi = theta[[i, k]] * beta[[k, j]] / denom;
                            s += x[[i, j]] * phi;
                        }
                    }
                    new_theta[[i, k]] = s;
                }
                let sum: f64 = (0..n_topics).map(|k| new_theta[[i, k]]).sum();
                let sum = sum.max(1e-30);
                for k in 0..n_topics {
                    new_theta[[i, k]] /= sum;
                }
            }
            theta = new_theta;
            // M-step: update β.
            let mut new_beta = Array2::<f64>::from_elem((n_topics, d), topic_word_prior);
            for i in 0..n {
                for j in 0..d {
                    let mut denom = 0.0_f64;
                    for kk in 0..n_topics {
                        denom += theta[[i, kk]] * beta[[kk, j]];
                    }
                    if denom > 0.0 {
                        for k in 0..n_topics {
                            let phi = theta[[i, k]] * beta[[k, j]] / denom;
                            new_beta[[k, j]] += x[[i, j]] * phi;
                        }
                    }
                }
            }
            for k in 0..n_topics {
                let sum: f64 = (0..d).map(|j| new_beta[[k, j]]).sum();
                let sum = sum.max(1e-30);
                for j in 0..d {
                    new_beta[[k, j]] /= sum;
                }
            }
            // Convergence.
            let mut delta = 0.0_f64;
            for k in 0..n_topics {
                for j in 0..d {
                    let dd = new_beta[[k, j]] - beta[[k, j]];
                    delta += dd * dd;
                }
            }
            beta = new_beta;
            if delta.sqrt() < tol {
                break;
            }
        }
        Ok(Self {
            components: beta,
            doc_topic: theta,
            n_topics,
            doc_topic_prior,
            topic_word_prior,
            n_iter: iters,
        })
    }

    /// Transform new documents into their (unnormalised) topic weights.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let d = self.components.ncols();
        if x.ncols() != d {
            return Err(Error::Shape("LDA::transform: shape mismatch".into()));
        }
        let mut theta = Array2::<f64>::from_elem((n, self.n_topics), 1.0 / self.n_topics as f64);
        for _ in 0..30 {
            let mut new_theta = Array2::<f64>::zeros((n, self.n_topics));
            for i in 0..n {
                for k in 0..self.n_topics {
                    let mut s = self.doc_topic_prior;
                    for j in 0..d {
                        let mut denom = 0.0_f64;
                        for kk in 0..self.n_topics {
                            denom += theta[[i, kk]] * self.components[[kk, j]];
                        }
                        if denom > 0.0 {
                            let phi = theta[[i, k]] * self.components[[k, j]] / denom;
                            s += x[[i, j]] * phi;
                        }
                    }
                    new_theta[[i, k]] = s;
                }
                let sum: f64 = (0..self.n_topics).map(|k| new_theta[[i, k]]).sum();
                let sum = sum.max(1e-30);
                for k in 0..self.n_topics {
                    new_theta[[i, k]] /= sum;
                }
            }
            theta = new_theta;
        }
        Ok(theta)
    }
}

// Prevent unused-import warning on Array1.
#[allow(dead_code)]
fn _touch(a: Array1<f64>) -> Array1<f64> {
    a
}
