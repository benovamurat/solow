# Verification & methodology

Solow's central claim is not "these look like reasonable numbers" but "these are
the **same** numbers an authoritative reference produces." This chapter explains
how that claim is enforced, the naming discipline behind it, and how to add a new
model with its own verification fixture.

## The golden-fixture harness

Verification is built on **golden fixtures**: frozen numerical outputs captured
once from an authoritative Python reference and committed to the repository as
JSON. The Rust test suites read those committed files and assert that Solow
reproduces every value to a tight tolerance. Crucially, **the tests need no
Python at test time** — `cargo test` runs against the frozen numbers.

The flow has three pieces:

1. **Generators** — `tools/reference/gen_*.py` scripts call the reference (and
   `numpy`/`scipy` for the linear-algebra and distribution fixtures) and dump
   results to `tests/fixtures/*.json`. There is roughly one generator per area
   (`gen_models.py`, `gen_glm.py`, `gen_discrete.py`, `gen_duration.py`,
   `gen_multivariate.py`, …), often with `_ext` companions for extended
   coverage.

2. **Fixtures** — `tests/fixtures/*.json` hold the committed golden values:
   inputs, the reference's parameter estimates, standard errors, and the other
   reported statistics.

3. **Rust tests** — each crate has a `tests/reference*.rs` that loads its
   fixture and compares Solow's output element-by-element.

```text
tools/reference/gen_models.py  ──►  tests/fixtures/models.json  ──►  crates/solow-regression/tests/reference.rs
        (run once, by a human)            (committed JSON)                 (runs on every `cargo test`)
```

## Tolerances

Agreement is asserted to a tolerance appropriate to each computation:

| Area | Tolerance |
| --- | --- |
| Most quantities (regression, distributions, tests) | `1e-8` |
| Maximum-likelihood / variational-Bayes estimates | `1e-6` |
| The formula engine (design matrices) | `1e-12` |
| The Python bindings vs. the Rust results | ~`1e-12` |

The MLE tolerance is looser only because two independent optimizers converging
to the same optimum agree to optimizer precision, not to the last bit; the
underlying closed-form quantities still match to `1e-8` or better.

## Regenerating fixtures

The `numpy`/`scipy`-based fixtures regenerate directly:

```sh
python3 tools/reference/gen_linalg.py
python3 tools/reference/gen_distributions.py
```

The modeling fixtures load the reference statistics package **indirectly**, by a
module name supplied through an environment variable, so the repository never
hard-codes that package's name:

```sh
SOLOW_REFERENCE=<reference-import-name> python3 tools/reference/gen_models.py
```

## The naming discipline

This indirection is deliberate and is enforced as a project rule: **the name of
the source statistics package never appears in the Solow source tree.** The
reference is loaded through `SOLOW_REFERENCE` at fixture-generation time and is
referred to in prose only as "the reference" or "an authoritative reference."
Comparison against the broader scientific-Python ecosystem (`numpy`, `scipy`,
`matplotlib`, `patsy`) is named freely; those are general tools, not the
specific package Solow re-implements.

The reason is that Solow is an independent, from-scratch re-implementation, not a
wrapper or a fork. Keeping the source name out of the tree keeps that boundary
clean while still letting the test suite prove numerical equivalence.

## Adding a model and its fixture

To add a new estimator with verification, follow the same pattern every existing
crate uses:

1. **Implement the model** in its crate (e.g. `crates/solow-yourarea/src/`),
   exposing a results struct with the standard fields (`params`, `bse`,
   `tvalues`, `pvalues`, information criteria, …).

2. **Write a generator** `tools/reference/gen_yourarea.py` that fits the same
   model in the reference on a fixed input and writes the inputs plus the
   reference outputs to `tests/fixtures/yourarea.json`. Load the reference via
   the `SOLOW_REFERENCE` environment variable — never import it by name.

3. **Generate the fixture once** and commit the JSON:

   ```sh
   SOLOW_REFERENCE=<reference-import-name> python3 tools/reference/gen_yourarea.py
   ```

4. **Add a Rust test** `crates/solow-yourarea/tests/reference.rs` that reads the
   fixture, fits the model with Solow on the same input, and asserts each value
   agrees within the appropriate tolerance (use the `approx` crate's
   `assert_abs_diff_eq!`).

5. **Run it offline**: `cargo test -p solow-yourarea` now verifies against the
   frozen numbers with no Python in the loop.

Because the fixtures are committed, contributors and CI verify correctness
without any reference installation; only *regenerating* a fixture (when the
input or the reference version changes) needs Python.

## Verifying the snippets in this book

Every Rust snippet printed in this guide is itself compile-checked against the
real crate APIs. The examples are pasted into a scratch crate that depends on the
Solow crates by path and built with `cargo build`; a signature mismatch fails the
build. This keeps the documentation honest as the API evolves.
