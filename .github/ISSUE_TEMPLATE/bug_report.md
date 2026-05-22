---
name: Bug report
about: Something doesn't work as documented
labels: bug
---

**What happened**
A clear description of the bug.

**Reproduce**
Minimal steps / code. If it's a context-assembly issue, the smallest
`build_context(...)` call (query + a few chunks) that shows it.

```python
# or rust / CLI
```

**Expected vs actual**
What you expected, and what you got. If relevant, paste the `ContextReport`
(`print(ctx.report)` or `redhop analyze-context`).

**Environment**
- RedHop version (`redhop.__version__` / crate version):
- Python / Rust version:
- OS:

**Notes**
Anything else (config, strategy, thresholds).
