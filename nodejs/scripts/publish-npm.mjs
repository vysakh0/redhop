#!/usr/bin/env node
// Publishes per-platform npm packages and the meta package. Replaces
// `napi prepublish` so we can override the napi-default name for the
// Windows platform package — npm's spam detector blocks the conventional
// `redhop-win32-x64-msvc` (the win32+msvc token pair); the same tarball
// published cleanly as `redhop-win-x64`.
//
// Run after `napi create-npm-dir -t .` and `napi artifacts --dir artifacts`
// have populated `npm/<platform>/` with package.json + .node files.

import { readFileSync, writeFileSync, existsSync, cpSync, readdirSync, statSync } from 'node:fs'
import { execSync } from 'node:child_process'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '..')
const meta = JSON.parse(readFileSync(resolve(repoRoot, 'package.json'), 'utf8'))
const VERSION = meta.version

const NAME_OVERRIDES = {
  // npm spam filter blocks the default `redhop-win32-x64-msvc` name.
  // Verified by publishing an identical tarball under `redhop-win-x64`.
  'redhop-win32-x64-msvc': 'redhop-win-x64',
}

const npmDir = resolve(repoRoot, 'npm')
if (!existsSync(npmDir)) {
  console.error('[publish] npm/ missing — run `napi create-npm-dir -t .` first.')
  process.exit(1)
}

const optionalDeps = {}

for (const dirName of readdirSync(npmDir).sort()) {
  const platDir = join(npmDir, dirName)
  if (!statSync(platDir).isDirectory()) continue

  const defaultName = `${meta.napi.name}-${dirName}`
  const finalName = NAME_OVERRIDES[defaultName] ?? defaultName

  const overrideDir = resolve(repoRoot, '.npm-overrides', dirName)
  for (const f of ['package.json', 'README.md']) {
    const src = join(overrideDir, f)
    if (existsSync(src)) cpSync(src, join(platDir, f))
  }

  const platPkgPath = join(platDir, 'package.json')
  const platPkg = JSON.parse(readFileSync(platPkgPath, 'utf8'))
  platPkg.name = finalName
  platPkg.version = VERSION
  writeFileSync(platPkgPath, JSON.stringify(platPkg, null, 2) + '\n')

  console.log(`\n[publish] ${finalName}@${VERSION}`)
  execSync('npm publish --access public', { cwd: platDir, stdio: 'inherit' })
  optionalDeps[finalName] = VERSION
}

meta.optionalDependencies = optionalDeps
writeFileSync(resolve(repoRoot, 'package.json'), JSON.stringify(meta, null, 2) + '\n')

const indexPath = resolve(repoRoot, 'index.js')
let indexSrc = readFileSync(indexPath, 'utf8')
for (const [defaultName, finalName] of Object.entries(NAME_OVERRIDES)) {
  indexSrc = indexSrc.split(`require('${defaultName}')`).join(`require('${finalName}')`)
}
writeFileSync(indexPath, indexSrc)

console.log(`\n[publish] redhop@${VERSION} (meta)`)
execSync('npm publish --access public', { cwd: repoRoot, stdio: 'inherit' })
console.log('\n[publish] done.')
