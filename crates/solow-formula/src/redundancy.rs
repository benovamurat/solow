//! Structural avoidance of redundancy in categorical coding.
//!
//! A direct port of patsy's `redundancy.py`. Given a term's categorical factors
//! (in term order) and the set of subterms already emitted by earlier terms in
//! the same numeric bucket, this decides how to code each categorical factor:
//! *full* rank (`includes_intercept = true`, all `k` levels) or *contrast*
//! (`false`, `k - 1` treatment dummies).
//!
//! The lattice trick: a term `a:b` conceptually expands to `1 + a- + b- + a-:b-`
//! (where `-` denotes contrast coding). Subterms already covered by previous
//! terms are dropped, and the survivors are then *simplified* by absorbing a
//! contrast factor into full coding wherever a strictly-smaller subterm lets us
//! (`a-:b-` absorbs `a-` to become `a+:b-`), which is what turns treatment
//! coding into full coding at exactly the right places.

use std::collections::HashMap;
use std::collections::HashSet;

/// An expanded factor: a categorical factor key plus whether it is full-rank.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExpandedFactor {
    includes_intercept: bool,
    factor: String,
}

/// A subterm: an (unordered) set of expanded factors.
#[derive(Debug, Clone)]
struct Subterm {
    efactors: Vec<ExpandedFactor>, // kept canonical (sorted) for set semantics
}

impl Subterm {
    fn new(mut efactors: Vec<ExpandedFactor>) -> Self {
        efactors.sort_by(|a, b| {
            a.factor
                .cmp(&b.factor)
                .then(a.includes_intercept.cmp(&b.includes_intercept))
        });
        efactors.dedup();
        Subterm { efactors }
    }

    fn key(&self) -> Vec<(bool, String)> {
        self.efactors
            .iter()
            .map(|e| (e.includes_intercept, e.factor.clone()))
            .collect()
    }

    fn factor_set(&self) -> HashSet<(bool, String)> {
        self.efactors
            .iter()
            .map(|e| (e.includes_intercept, e.factor.clone()))
            .collect()
    }

    /// True if `self` is like `a-:b-` and `other` is like `a-` (i.e. self has
    /// exactly one more factor than other and is a superset).
    fn can_absorb(&self, other: &Subterm) -> bool {
        self.efactors.len() as isize - other.efactors.len() as isize == 1
            && other.factor_set().is_subset(&self.factor_set())
    }

    /// Absorb `other`: the single extra (contrast) factor becomes full-rank.
    fn absorb(&self, other: &Subterm) -> Subterm {
        let self_set = self.factor_set();
        let other_set = other.factor_set();
        let diff: Vec<(bool, String)> = self_set.difference(&other_set).cloned().collect();
        debug_assert_eq!(diff.len(), 1);
        let (includes_intercept, factor) = diff[0].clone();
        debug_assert!(!includes_intercept);
        let mut new_factors: Vec<ExpandedFactor> = other.efactors.clone();
        new_factors.push(ExpandedFactor {
            includes_intercept: true,
            factor,
        });
        Subterm::new(new_factors)
    }
}

impl PartialEq for Subterm {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}
impl Eq for Subterm {}

/// The set of subterms already produced within the current numeric bucket.
#[derive(Default)]
pub(crate) struct UsedSubterms {
    set: HashSet<Vec<(bool, String)>>,
}

impl UsedSubterms {
    fn contains(&self, s: &Subterm) -> bool {
        self.set.contains(&s.key())
    }
    fn insert(&mut self, s: &Subterm) {
        self.set.insert(s.key());
    }
}

/// All subsets of `factors`, ordered shortest-first and (within a length) by
/// the original index order — matching patsy's `_subsets_sorted`.
fn subsets_sorted(factors: &[String]) -> Vec<Vec<usize>> {
    // Generate all subsets as index lists.
    let n = factors.len();
    let mut subsets: Vec<Vec<usize>> = Vec::new();
    for mask in 0u32..(1u32 << n) {
        let mut s = Vec::new();
        for (i, _) in factors.iter().enumerate() {
            if mask & (1 << i) != 0 {
                s.push(i);
            }
        }
        subsets.push(s);
    }
    // Sort by the index tuple first (stable natural order), then by length.
    subsets.sort();
    subsets.sort_by_key(|s| s.len());
    subsets
}

/// Greedily simplify subterms by absorption, left to right (patsy
/// `_simplify_subterms` / `_simplify_one_subterm`). Each successful pass
/// absorbs one shorter subterm into a longer one and removes the shorter.
fn simplify_subterms(subterms: &mut Vec<Subterm>) {
    'outer: loop {
        for short_i in 0..subterms.len() {
            for long_i in (short_i + 1)..subterms.len() {
                if subterms[long_i].can_absorb(&subterms[short_i]) {
                    subterms[long_i] = subterms[long_i].absorb(&subterms[short_i]);
                    subterms.remove(short_i);
                    continue 'outer;
                }
            }
        }
        break;
    }
}

/// Decide categorical codings for a term.
///
/// `cat_factors` are the term's categorical factor keys in term order.
/// Returns one coding map per emitted subterm; each map sends a factor key to
/// `true` (full coding) or `false` (contrast/treatment coding).
pub(crate) fn pick_contrasts_for_term(
    cat_factors: &[String],
    used: &mut UsedSubterms,
) -> Vec<HashMap<String, bool>> {
    let mut subterms: Vec<Subterm> = Vec::new();
    for subset in subsets_sorted(cat_factors) {
        let efactors: Vec<ExpandedFactor> = subset
            .iter()
            .map(|&i| ExpandedFactor {
                includes_intercept: false,
                factor: cat_factors[i].clone(),
            })
            .collect();
        let subterm = Subterm::new(efactors);
        if !used.contains(&subterm) {
            subterms.push(subterm);
        }
    }
    for s in &subterms {
        used.insert(s);
    }

    simplify_subterms(&mut subterms);

    subterms
        .iter()
        .map(|st| {
            let mut coding = HashMap::new();
            for e in &st.efactors {
                coding.insert(e.factor.clone(), e.includes_intercept);
            }
            coding
        })
        .collect()
}
