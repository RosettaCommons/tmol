# Developing TMol

The maintained [development guide](docs/user_guide/development.md) covers source
builds, native extensions, tests, CI, and releases. See the
[contributor guide](docs/contributor_guide.md) for documentation and tutorial
changes, and the [development skill](skills/tmol-develop/SKILL.md) for a focused
implementation and validation workflow.

- Install from source with the `dev` extra; it includes optional ligand tools.
- Run `pre-commit run --all-files` and the tests affected by your changes.
- Pull requests run CPU tests on GitHub-hosted Linux and macOS runners.
- Maintainers can dispatch CUDA validation with
  `gh workflow run ci.yml --ref <branch> -f gpu=true`.
- Release publication requires the complete wheel matrix, source-distribution
  checks, and CUDA wheel scoring. See [releasing](docs/user_guide/development.md#releasing).
