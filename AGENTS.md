# Test procedure

Use the project `venv/` and the offline launcher, which isolates home/state and
blocks network access before collection. For a small change, pass related
test files to it. For the default local sweep:

```sh
venv/bin/python scripts/validate_offline.py -- tests -q -m 'not slow'
```

This omits real UI sessions. A UI change also needs its affected `slow` tests;
do not treat the fast result as UI coverage. Before landing:

```sh
venv/bin/python scripts/validate_offline.py -- tests -q
```

The complete offline suite is mandatory in `.cc-land.json`. Keep it unfiltered;
the other pytest gate can be narrowed by the shared landing tool. Reuse an app
session for related assertions where state permits; keep independent state and
fixed-bug coverage. See `docs/TEST_DIET.md` for retained behaviors and timings.
