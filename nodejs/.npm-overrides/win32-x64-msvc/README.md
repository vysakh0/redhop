# redhop-win-x64

This is the **Windows x64** native binary for [**redhop**](https://www.npmjs.com/package/redhop) — a reasoning-preserving context runtime for RAG (chunk, retrieve, and allocate the document context an LLM should see, with citations and a Decision Report).

## You probably don't want to install this directly

The main `redhop` package depends on this package as an
[optionalDependency](https://docs.npmjs.com/cli/v10/configuring-npm/package-json#optionaldependencies).
npm will install the correct platform binary for your machine automatically:

```sh
npm install redhop
```

If you're on a Windows x64 machine, this package will be installed alongside `redhop` so the native binding can load.

## What's inside

A single file: `redhop.win32-x64-msvc.node` — the [N-API](https://nodejs.org/api/n-api.html)
addon built from the Rust workspace at
[github.com/vysakh0/redhop](https://github.com/vysakh0/redhop), compiled for
`x86_64-pc-windows-msvc` against the stable Rust toolchain via
[`@napi-rs/cli`](https://github.com/napi-rs/napi-rs).

No installer, no postinstall scripts, no network access at install time.

## Links

- **Main package**: <https://www.npmjs.com/package/redhop>
- **Source**: <https://github.com/vysakh0/redhop>
- **Issues**: <https://github.com/vysakh0/redhop/issues>
- **Changelog**: <https://github.com/vysakh0/redhop/blob/main/CHANGELOG.md>

## License

Apache-2.0 © Vysakh Sreenivasan. See the
[LICENSE](https://github.com/vysakh0/redhop/blob/main/LICENSE) and
[NOTICE](https://github.com/vysakh0/redhop/blob/main/NOTICE) files in the
source repository.
