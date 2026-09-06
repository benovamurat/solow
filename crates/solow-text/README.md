# solow-text

**Text feature extraction for Rust NLP.** Count vectorizer, TF-IDF vectorizer, hashing vectorizer, feature hasher, and dict vectorizer with configurable tokenizer, n-grams, min and max document frequency, and stop-word support.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-text = "0.7"
```

## Quick example

```rust
use solow_text::TfidfVectorizer;

let mut vec = TfidfVectorizer::new()
    .lowercase(true)
    .ngram_range((1, 2))
    .min_df(2)
    .max_df(0.95);

let x = vec.fit_transform(&documents)?;
```

## What is inside

| Vectorizer | What it does |
|---|---|
| `CountVectorizer` | Bag-of-words with configurable `lowercase`, word `ngram_range = (n_min, n_max)`, `min_df` / `max_df` filters, optional stop-word set. Default token pattern `r"(?u)\b\w\w+\b"`. |
| `TfidfVectorizer` | TF-IDF weighting `idf(t) = log((1 + n) / (1 + df(t))) + 1` layered on top of CountVectorizer, with L2 row normalisation. |
| `HashingVectorizer` | Streaming-friendly hashing to a fixed feature space, no vocabulary. |
| `FeatureHasher` | Explicit feature hasher for dict-of-features input. |
| `DictVectorizer` | Vectorize a sequence of `{feature: value}` dicts into a numeric matrix with categorical one-hot expansion. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Deterministic hashing for reproducible feature indices.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
