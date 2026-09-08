# Text feature extraction

`solow-text` provides count vectorizer, TF-IDF vectorizer, hashing vectorizer, feature hasher, and dict vectorizer with configurable tokenizer, n-grams, min and max document frequency, and stop-word support.

## Vectorizers

| Vectorizer | What it does |
|---|---|
| `CountVectorizer` | Bag-of-words with configurable `lowercase`, word `ngram_range = (n_min, n_max)`, `min_df` / `max_df` filters, optional stop-word set. Default token pattern `r"(?u)\b\w\w+\b"`. |
| `TfidfVectorizer` | TF-IDF weighting `idf(t) = log((1 + n) / (1 + df(t))) + 1` layered on top of CountVectorizer, with L2 row normalisation. |
| `HashingVectorizer` | Streaming-friendly hashing to a fixed feature space, no vocabulary. |
| `FeatureHasher` | Explicit feature hasher for dict-of-features input. |
| `DictVectorizer` | Vectorize a sequence of `{feature: value}` dicts into a numeric matrix with categorical one-hot expansion. |

## Example

```rust
use solow_text::TfidfVectorizer;

let mut vec = TfidfVectorizer::new()
    .lowercase(true)
    .ngram_range((1, 2))
    .min_df(2)
    .max_df(0.95);

let x = vec.fit_transform(&documents)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management 24(5), 513-523.
