# Releasing RedHop

RedHop ships to three ecosystems (PyPI, crates.io, npm) from this one repo.
All three publish in **parallel from a single `v*` tag push** — no per-
ecosystem Run-workflow click. A fourth workflow creates the GitHub Release
with notes from `CHANGELOG.md`, which is how external watchers learn there's
a new version.

All three are pinned to the same version (`0.3.0` as of this writing). Bump in lockstep:
`python/pyproject.toml`, `nodejs/package.json` (+
`nodejs/.npm-overrides/win32-x64-msvc/package.json`), and
`[workspace.package] version` in `Cargo.toml`.

## The release flow

```text
   1. bump 6 version pins to X.Y.Z (see "Files to bump" below)
   2. write a "## [X.Y.Z] - YYYY-MM-DD" entry in CHANGELOG.md
   3. commit + push
   4. git tag vX.Y.Z && git push --tags        ← the release trigger
        │
        ├─► release-crates       (publishes redhop@X.Y.Z to crates.io)
        ├─► release-python       (builds 5 wheels + sdist, publishes to PyPI)
        ├─► release-node         (builds 5 platform packages, publishes to npm)
        └─► create-release       (extracts the CHANGELOG section, creates GH Release)
```

All four fire in parallel on the tag push. Each publish workflow runs a
fast `verify-version` step first that fails the build if the tag's version
doesn't match the source files — so a stale tag never publishes the wrong
content. `create-release` fails if there's no `## [X.Y.Z]` entry in
`CHANGELOG.md`, forcing the changelog to stay in lockstep with tags.

The three publish workflows still support `workflow_dispatch` as an escape
hatch — useful for re-running a single ecosystem if (say) one npm platform
hit a transient build error. Tag-pushed runs are the normal path.

## Files to bump for a release

| File | What |
|---|---|
| `Cargo.toml` | `[workspace.package] version = "X.Y.Z"` |
| `nodejs/Cargo.toml` | `redhop-node` cdylib version |
| `nodejs/package.json` | npm meta version |
| `nodejs/.npm-overrides/win32-x64-msvc/package.json` | Windows platform package version |
| `python/Cargo.toml` | `redhop-py` cdylib version |
| `python/pyproject.toml` | PyPI version |

## One-time setup

| Ecosystem | What to configure |
| --- | --- |
| **PyPI** | Trusted Publishing (OIDC) for this repo + a `pypi` environment. No token secret. |
| **npm** | An automation token as the `NPM_TOKEN` repo secret. |
| **crates.io** | `CARGO_REGISTRY_TOKEN` repo secret. |
| **GitHub Releases** | Nothing — uses the default `GITHUB_TOKEN`. |

## PyPI (Python wheels)

Actions tab → **release-python** → **Run workflow**. Builds the **one self-contained
wheel** (semantic engine + file parsers compiled in; no Python deps) for Linux
x86_64 + aarch64 (manylinux_2_28), macOS x86_64/aarch64, and Windows x64, plus an
sdist, then publishes via PyPI Trusted Publishing. `pip install redhop` then gives
the user everything.

> **Note on runners:** aarch64-Linux builds on a *native* `ubuntu-24.04-arm` runner
> (QEMU cross-compilation fails `ring`'s ARM assembler). Both macOS wheels build on
> `macos-14` (arm64 native, x86_64 cross) to avoid the scarce/queued macos-13 runners.

## npm (Node binding)

Actions tab → **release-node** → **Run workflow**. Builds a `.node` per platform in
`napi.triples` (package.json) and publishes the main package + per-platform optional
packages (auth via the `NPM_TOKEN` secret).

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
