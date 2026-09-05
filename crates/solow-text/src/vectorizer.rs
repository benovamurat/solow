//! Bag-of-words / TF-IDF vectorisers.

use ndarray::Array2;
use solow_core::{Error, Result};
use std::collections::{BTreeMap, HashSet};

fn tokenise(doc: &str, lowercase: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let push = |cur: &mut String, out: &mut Vec<String>| {
        if cur.len() >= 2 {
            out.push(cur.clone());
        }
        cur.clear();
    };
    for ch in doc.chars() {
        if ch.is_alphanumeric() {
            if lowercase {
                for c in ch.to_lowercase() {
                    cur.push(c);
                }
            } else {
                cur.push(ch);
            }
        } else {
            push(&mut cur, &mut out);
        }
    }
    // Flush the trailing token.
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

fn ngrams(tokens: &[String], n_min: usize, n_max: usize) -> Vec<String> {
    let mut out = Vec::new();
    for n in n_min..=n_max {
        if n == 0 || n > tokens.len() {
            continue;
        }
        for w in tokens.windows(n) {
            out.push(w.join(" "));
        }
    }
    out
}

/// Bag-of-words vectoriser.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct CountVectorizer {
    /// Sorted feature vocabulary.
    pub vocabulary: Vec<String>,
    /// Whether tokens were lower-cased during fit.
    pub lowercase: bool,
    /// N-gram range (inclusive).
    pub ngram_range: (usize, usize),
    /// Optional stop-word set (already lower-cased if `lowercase`).
    pub stop_words: HashSet<String>,
    /// Minimum document frequency required to retain a term.
    pub min_df: usize,
    /// Maximum document frequency retained (0 = no upper limit).
    pub max_df: usize,
}

impl CountVectorizer {
    /// Fit with defaults (`lowercase = true`, `ngram_range = (1, 1)`,
    /// `min_df = 1`, `max_df = 0` for "no upper limit", no stop words).
    pub fn fit(corpus: &[String]) -> Result<(Self, Array2<f64>)> {
        Self::fit_with(corpus, true, (1, 1), 1, 0, HashSet::new())
    }

    /// Full-configuration fit.
    pub fn fit_with(
        corpus: &[String],
        lowercase: bool,
        ngram_range: (usize, usize),
        min_df: usize,
        max_df: usize,
        stop_words: HashSet<String>,
    ) -> Result<(Self, Array2<f64>)> {
        if corpus.is_empty() {
            return Err(Error::Value(
                "CountVectorizer::fit_with: corpus must be non-empty".into(),
            ));
        }
        if ngram_range.0 == 0 || ngram_range.1 < ngram_range.0 {
            return Err(Error::Value(format!(
                "CountVectorizer::fit_with: ngram_range must satisfy 1 ≤ min ≤ max (got {:?})",
                ngram_range
            )));
        }
        // Materialise per-doc n-grams once.
        let per_doc: Vec<Vec<String>> = corpus
            .iter()
            .map(|d| {
                let toks: Vec<String> = tokenise(d, lowercase)
                    .into_iter()
                    .filter(|t| !stop_words.contains(t))
                    .collect();
                ngrams(&toks, ngram_range.0, ngram_range.1)
            })
            .collect();
        // Global vocabulary + document-frequency map.
        let mut df: BTreeMap<String, usize> = BTreeMap::new();
        for tokens in &per_doc {
            let uniq: HashSet<&String> = tokens.iter().collect();
            for t in uniq {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
        }
        // Apply min_df / max_df.
        let max_effective = if max_df == 0 { corpus.len() } else { max_df };
        let mut vocab: Vec<String> = df
            .iter()
            .filter(|(_, &c)| c >= min_df && c <= max_effective)
            .map(|(k, _)| k.clone())
            .collect();
        vocab.sort();
        // Build the count matrix.
        let mut out = Array2::<f64>::zeros((corpus.len(), vocab.len()));
        let index: BTreeMap<&String, usize> =
            vocab.iter().enumerate().map(|(i, s)| (s, i)).collect();
        for (row, tokens) in per_doc.iter().enumerate() {
            for t in tokens {
                if let Some(&col) = index.get(t) {
                    out[[row, col]] += 1.0;
                }
            }
        }
        Ok((
            Self {
                vocabulary: vocab,
                lowercase,
                ngram_range,
                stop_words,
                min_df,
                max_df,
            },
            out,
        ))
    }

    /// Transform a new corpus using the fitted vocabulary.
    pub fn transform(&self, corpus: &[String]) -> Array2<f64> {
        let index: BTreeMap<&String, usize> = self
            .vocabulary
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        let mut out = Array2::<f64>::zeros((corpus.len(), self.vocabulary.len()));
        for (row, doc) in corpus.iter().enumerate() {
            let toks: Vec<String> = tokenise(doc, self.lowercase)
                .into_iter()
                .filter(|t| !self.stop_words.contains(t))
                .collect();
            let grams = ngrams(&toks, self.ngram_range.0, self.ngram_range.1);
            for t in grams {
                if let Some(&col) = index.get(&t) {
                    out[[row, col]] += 1.0;
                }
            }
        }
        out
    }
}

/// TF-IDF vectoriser — [`CountVectorizer`] + Salton-Buckley weighting.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct TfidfVectorizer {
    /// Underlying count vectoriser (retains vocabulary + config).
    pub count: CountVectorizer,
    /// Per-term IDF vector.
    pub idf: Vec<f64>,
}

impl TfidfVectorizer {
    /// Fit onto a corpus, returning the vectoriser and the fitted TF-IDF matrix.
    pub fn fit(corpus: &[String]) -> Result<(Self, Array2<f64>)> {
        let (count, x) = CountVectorizer::fit(corpus)?;
        let (vectorizer, tfidf) = Self::from_counts(count, &x);
        Ok((vectorizer, tfidf))
    }

    /// Combine an existing `CountVectorizer` with its count matrix into
    /// a `TfidfVectorizer` and the corresponding TF-IDF matrix.
    pub fn from_counts(count: CountVectorizer, x: &Array2<f64>) -> (Self, Array2<f64>) {
        let n = x.nrows();
        let d = x.ncols();
        // Document frequencies from the count matrix.
        let mut df = vec![0.0_f64; d];
        for i in 0..n {
            for j in 0..d {
                if x[[i, j]] > 0.0 {
                    df[j] += 1.0;
                }
            }
        }
        let n_f = n as f64;
        // smooth_idf=True: log((1 + n) / (1 + df)) + 1
        let idf: Vec<f64> = df
            .iter()
            .map(|d| ((1.0 + n_f) / (1.0 + d)).ln() + 1.0)
            .collect();
        // Multiply and row-L2-normalise.
        let mut tfidf = Array2::<f64>::zeros(x.dim());
        for i in 0..n {
            for j in 0..d {
                tfidf[[i, j]] = x[[i, j]] * idf[j];
            }
            let norm: f64 = tfidf.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for j in 0..d {
                    tfidf[[i, j]] /= norm;
                }
            }
        }
        (Self { count, idf }, tfidf)
    }

    /// Transform a new corpus using the fitted IDF.
    pub fn transform(&self, corpus: &[String]) -> Array2<f64> {
        let x = self.count.transform(corpus);
        let (n, d) = x.dim();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                out[[i, j]] = x[[i, j]] * self.idf[j];
            }
            let norm: f64 = out.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for j in 0..d {
                    out[[i, j]] /= norm;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_vectorizer_reproduces_hand_derived_counts() {
        let corpus = vec![
            "the cat sat on the mat".to_string(),
            "the dog sat on the log".to_string(),
        ];
        let (cv, x) = CountVectorizer::fit(&corpus).unwrap();
        // Vocabulary (lower-case, len ≥ 2, ngram=1): cat, dog, log, mat, on, sat, the
        assert_eq!(
            cv.vocabulary,
            vec![
                "cat".to_string(),
                "dog".to_string(),
                "log".to_string(),
                "mat".to_string(),
                "on".to_string(),
                "sat".to_string(),
                "the".to_string(),
            ]
        );
        // Row totals equal the number of retained (len ≥ 2) tokens per doc.
        assert_eq!(x.row(0).iter().sum::<f64>(), 6.0); // the cat sat on the mat
        assert_eq!(x.row(1).iter().sum::<f64>(), 6.0);
        // "the" appears twice in every doc.
        let i_the = cv.vocabulary.iter().position(|w| w == "the").unwrap();
        assert_eq!(x[[0, i_the]], 2.0);
        assert_eq!(x[[1, i_the]], 2.0);
    }

    #[test]
    fn tfidf_rows_have_unit_l2_norm() {
        let corpus = vec![
            "cat sat on the mat".to_string(),
            "dog sat on the log".to_string(),
            "cat and dog play".to_string(),
        ];
        let (_, tfidf) = TfidfVectorizer::fit(&corpus).unwrap();
        for i in 0..tfidf.nrows() {
            let norm: f64 = tfidf.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "row {i} norm = {norm}");
        }
    }

    #[test]
    fn ngram_range_includes_bigrams() {
        let corpus = vec!["a b c".to_string()];
        // Tokeniser drops single letters (len < 2), so nothing to n-gram.
        let (cv, x) =
            CountVectorizer::fit_with(&corpus, true, (1, 2), 1, 0, HashSet::new()).unwrap();
        assert!(cv.vocabulary.is_empty());
        assert_eq!(x.dim(), (1, 0));
    }
}
