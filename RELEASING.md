# Releasing RedHop

RedHop ships to three ecosystems from this one repo. They're independent — release
them one at a time. **Recommended order: PyPI first** (simplest, flagship), then npm,
then crates.io once there's demand.

All three are pinned to the same version (`0.1.0`). Bump in lockstep:
`python/pyproject.toml`, `nodejs/package.json`, and `[workspace.package] version` in
`Cargo.toml` (which the crate-to-crate deps in `[workspace.dependencies]` track —
update those `version = "…"` too).

## One-time setup

| Ecosystem | What to configure |
| --- | --- |
| **PyPI** | Trusted Publishing (OIDC) for this repo + a `pypi` environment. No token secret. |
| **npm** | An automation token as the `NPM_TOKEN` repo secret. |
| **crates.io** | `cargo login` locally (first publish is manual — see below). |

## PyPI (Python wheels)

Tag-triggered via `.github/workflows/release-python.yml`:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

Builds the **one self-contained wheel** (semantic engine + file parsers compiled in;
no Python deps) for Linux x86_64/aarch64, macOS x86_64/aarch64, and Windows x64, plus
an sdist, then publishes. `pip install redhop` then gives the user everything.

## npm (Node binding)

Tag-triggered via `.github/workflows/release-node.yml`. Builds a `.node` per platform
in `napi.triples` (package.json) and publishes the main package + per-platform optional
packages.

**Before publishing, regenerate the committed typings** if the API changed:

```bash
cd nodejs && npm install && npx napi build --platform   # updates index.js + index.d.ts
git add nodejs/index.js nodejs/index.d.ts && git commit -m "node: regenerate bindings"
```

The matrix in the workflow and `napi.triples` must stay in sync. Adding arm-linux /
musl / windows-arm64 needs a cross toolchain and an ORT build for that target — tune
with a real run before relying on them.

## crates.io (Rust crates)

Lowest priority; do the first publish **manually and deliberately** (not on a tag).
Only the crates in `redhop`'s dependency tree are strictly required, but publishing the
whole library set is simplest. Use `cargo-workspaces`, which computes the topological
order and waits for the index between dependent crates (the binding crates `redhop-py`
/ `redhop-node` are separate workspaces and are *not* published here):

```bash
cargo install cargo-workspaces
cargo workspaces publish --from-git --no-git-commit   # publishes in dependency order
```

Caveat: the dense/rerank tiers depend on `ort = "2.0.0-rc.10"` (a release candidate).
Publishing `redhop` to crates.io means users inherit that pre-release; revisit once a
stable `ort 2.0` ships. (The PyPI/npm artifacts bundle ORT, so they're unaffected.)

## Notes for users (worth saying in release notes)

- **First-run model download.** `retrieval="lexical"` (the default) needs no model. The
  `hybrid`/`semantic`/`rerank` tiers download a small ONNX model from Hugging Face on
  first use (cached afterward). The model revisions are pinned to commit SHAs for
  reproducibility. For air-gapped/CI use, pre-warm the cache or set `HF_HUB_OFFLINE=1`.
- The artifacts are self-contained: the ONNX runtime is statically linked, so there are
  no extra system or language dependencies to install.
