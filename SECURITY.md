# Security Policy

## Supported versions

RedHop is pre-1.0 (`0.x`). Security fixes land on the latest `0.x` release; there
is no long-term support for older `0.x` lines. Pin a version and upgrade forward.

| Version | Supported |
| ------- | --------- |
| latest `0.x` | ✅ |
| older `0.x` | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security reports.**

Use GitHub's private vulnerability reporting:
**Security → Report a vulnerability** on the repository
(<https://github.com/redhop/redhop/security/advisories/new>). This opens a private
advisory visible only to you and the maintainers.

Please include:

- the affected version / commit,
- a description of the issue and its impact,
- and a minimal reproduction if you have one.

We aim to acknowledge a report within a few days, agree on a disclosure timeline,
and credit reporters who wish to be named once a fix is released.

## Scope

RedHop is a local library — it has no server, no network listener, and no
authentication surface. Realistic concerns are therefore things like: parsing
untrusted documents (`from_file` / `from_folder`), denial-of-service via
pathological inputs, or unsafe handling of a user-supplied model file. Issues in
**your** embedding model, ONNX Runtime, or other third-party dependencies should be
reported upstream, though we're happy to help coordinate.
