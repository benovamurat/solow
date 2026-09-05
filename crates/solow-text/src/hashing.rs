//! HashingVectorizer and FeatureHasher — the reference "hash the tokens"
//! feature-extraction shortcuts that avoid keeping a vocabulary.

use ndarray::Array2;
use solow_core::{Error, Result};

/// HashingVectorizer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct HashingVectorizer {
    /// Output dimension.
    pub n_features: usize,
    /// Whether to lowercase before tokenising.
    pub lowercase: bool,
    /// Whether to use a signed hash (Rademacher signs).
    pub alternate_sign: bool,
    /// Optional stop-word set (case-sensitive after `lowercase`).
    pub stop_words: Vec<String>,
}

impl HashingVectorizer {
    /// Build with the reference defaults `n_features = 2²⁰ = 1_048_576`.
    pub fn new(n_features: usize) -> Result<Self> {
        if n_features == 0 {
            return Err(Error::Value("HashingVectorizer: n_features must be ≥ 1".into()));
        }
        Ok(Self {
            n_features,
            lowercase: true,
            alternate_sign: true,
            stop_words: Vec::new(),
        })
    }

    /// Transform a corpus into a dense hash-features matrix.
    pub fn transform(&self, docs: &[&str]) -> Result<Array2<f64>> {
        let n = docs.len();
        let d = self.n_features;
        let mut out = Array2::<f64>::zeros((n, d));
        for (i, doc) in docs.iter().enumerate() {
            let text = if self.lowercase {
                doc.to_lowercase()
            } else {
                doc.to_string()
            };
            for tok in tokenise(&text) {
                if self.stop_words.iter().any(|s| s == &tok) {
                    continue;
                }
                let h = fnv1a_64(tok.as_bytes());
                let bin = (h % (d as u64)) as usize;
                let sign = if self.alternate_sign && (h >> 63) & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                out[[i, bin]] += sign;
            }
        }
        Ok(out)
    }
}

/// FeatureHasher — same as HashingVectorizer but the caller already
/// supplies token strings (or `(token, count)` pairs) rather than raw
/// documents.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureHasher {
    /// Output dimension.
    pub n_features: usize,
    /// Whether to use a signed hash.
    pub alternate_sign: bool,
}

impl FeatureHasher {
    /// Build.
    pub fn new(n_features: usize) -> Result<Self> {
        if n_features == 0 {
            return Err(Error::Value("FeatureHasher: n_features must be ≥ 1".into()));
        }
        Ok(Self {
            n_features,
            alternate_sign: true,
        })
    }

    /// Transform.
    pub fn transform(&self, docs: &[Vec<(String, f64)>]) -> Result<Array2<f64>> {
        let n = docs.len();
        let d = self.n_features;
        let mut out = Array2::<f64>::zeros((n, d));
        for (i, doc) in docs.iter().enumerate() {
            for (tok, count) in doc {
                let h = fnv1a_64(tok.as_bytes());
                let bin = (h % (d as u64)) as usize;
                let sign = if self.alternate_sign && (h >> 63) & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                out[[i, bin]] += sign * count;
            }
        }
        Ok(out)
    }
}

fn tokenise(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_vectorizer_returns_a_finite_matrix() {
        let hv = HashingVectorizer::new(16).unwrap();
        let m = hv.transform(&["hello world hello", "foo bar baz"]).unwrap();
        assert_eq!(m.shape(), &[2, 16]);
        // The row sum should be non-zero for both documents.
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        for j in 0..16 {
            s0 += m[[0, j]].abs();
            s1 += m[[1, j]].abs();
        }
        assert!(s0 > 0.0);
        assert!(s1 > 0.0);
    }

    #[test]
    fn feature_hasher_accepts_token_count_pairs() {
        let fh = FeatureHasher::new(8).unwrap();
        let m = fh.transform(&[
            vec![("apple".to_string(), 2.0), ("pear".to_string(), 1.0)],
            vec![("apple".to_string(), 1.0)],
        ]).unwrap();
        assert_eq!(m.shape(), &[2, 8]);
    }
}
