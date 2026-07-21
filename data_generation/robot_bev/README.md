# Canonical Robot BEV data toolkit

This package generates and validates framework-independent Robot BEV datasets.
The first source adapter renders original Replica v1 PTex scenes with
Habitat-Sim 0.2.2. The canonical writer, validator, and geometry checks do not
depend on Habitat-Sim or BEVFusion.

The required workflow is:

```text
source adapter -> canonical writer -> strict validator -> geometry review
               -> BEVFusion converter -> training gates
```

Do not train directly from a generator's private state. The canonical root and
its root-relative `robot_infos_<split>.pkl` files are the source of truth.

## Package map

| Path | Responsibility |
| --- | --- |
| `schema.py` | Schema-v4 constants, path rules, and supervision-mask rules |
| `writer.py` | Atomic frame, manifest, index, metadata, and summary output |
| `validator.py` | Dependency-light structural, numeric, and quality checks |
| `geometry_checks.py` | Camera projection, BEV orientation, and sweep diagnostics |
| `sources/habitat_common.py` | Shared Habitat sensor, pose, navigation, and depth helpers |
| `sources/replica.py` | Replica asset preflight, semantic mapping, rendering, and orchestration |
| `cli/generate_replica.py` | Replica generation entry point |
| `cli/validate_dataset.py` | JSON validation and diagnostic entry point |
| `configs/` | Tracked 18-scene list and 14/2/2 split example |

## Start here

- [Usage guide for rendering and conversion](docs/usage_zh.md)
- [Schema and coordinates](docs/schema_v4.md)
- [Replica generation runbook](docs/habitat_replica.md)
- [Adding a source adapter](docs/add_new_source.md)
- [Validation and geometry gates](docs/quality_checks.md)

Generation success alone is not a release gate. Validate the complete root and
every split, inspect representative geometry bundles, run the downstream
single-sample and one-batch gates, and obtain human visual approval before
starting a full training run.
