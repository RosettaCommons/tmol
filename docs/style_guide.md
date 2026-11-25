# Style guide

Version: 2

These rules apply to code under `tmol/`. The naming and review rules are enforced on the
[public API](#what-counts-as-public) only, but should serve as guidelines for the entire codebase. The formatting rules (`black`, etc) cover all code. Rules marked **Checked** are enforced by
pre-commit, so a violation fails the commit; rules marked **Review** are
checked by reviewers; [provisional rules](#provisional-rules) are proposals
that are reported only on request.

Run the checks before committing:

```bash
pre-commit run --all-files
python scripts/check_api_style.py   # the API naming rules alone, in under a second
python scripts/check_api_style.py --extended   # also report provisional rules
```

## What counts as public

A name is public when it is listed in an `__all__` of a `tmol` package, along
with the public methods and fields of public classes. Implementation modules
are underscore-prefixed (`tmol/pose/_pose_stack.py`) and their objects are
re-exported through a package `__init__.py`. Helpers that are not exported
start with `_`.

The naming rules below apply only to public names. Private helpers may use
local abbreviations, but prefer the vocabulary there too.

## Formatting

| ID | Rule | Enforcement | Why |
|---|---|---|---|
| FMT1 | Format Python with Black. | Checked (`black`) | [why](#why-fmt1) |
| FMT2 | Pass Flake8 with the repository's `.flake8` configuration. | Checked (`flake8`) | [why](#why-fmt2) |
| FMT3 | Format C++ and CUDA with clang-format. | Checked (`clang-format`) | [why](#why-fmt3) |

## Naming

### API001: no one- or two-character public names

**Checked** ([why](#why-api001)). Function, method, parameter and field names in the public API
are longer than two characters.

```python
# Don't
def run_kin_min(pose_stack, sfxn, ff, mm): ...
def set_weight(self, st, weight): ...

# Do
def run_kin_min(pose_stack, sfxn, fold_forest, move_map): ...
def set_weight(self, score_type, weight): ...
```

Allowed exceptions, because each is the established term rather than an
abbreviation:

| Name | Why |
|---|---|
| `ph` | pH |
| `to` | `.to(device)`, matching `torch.Tensor.to` |

### API002: no banned abbreviations

**Checked** ([why](#why-api002)). These may not appear in a public name:

| Banned | Write instead | Applies to |
|---|---|---|
| `mc` | `mainchain` | any underscore-separated part of a name |
| `sc` | `sidechain` | any part |
| `mcsc` | `mainchain_or_sidechain` | any part |
| `mcs`, `scs` | `mainchain_torsions`, `sidechain_torsions` | any part |
| `torch_device` | `device` | whole name |
| `co` | `canonical_ordering` | whole name |
| `ff` | `fold_forest` | whole name |
| `mm` | `move_map` | whole name |
| `fn` | `constraint_function` | whole name |
| `st` | `score_type` | whole name |
| `rt` | `residue_type` | whole name |
| `bt` | `block_type` | whole name |

```python
# Don't
move_map.set_move_all_mc_tors_for_blocks(mask)
# Do
move_map.set_move_all_mainchain_torsions_for_blocks(mask)
```

### API003: `param_db` before `device`

**Checked** ([why](#why-api003)). A signature that takes both lists the parameter database first.

```python
# Don't
def make_thing(device, param_db=None): ...
# Do
def make_thing(param_db, device): ...
```

### API004: names follow types

**Checked** ([why](#why-api004)). A parameter annotated with one of these types uses the matching
name. A suffixed name is allowed where a signature needs to tell two objects
of the same type apart.

| Type | Name | Also allowed |
|---|---|---|
| `ParameterDatabase` | `param_db` | — |
| `torch.device` | `device` | `*_device`, such as `anneal_device` (never `torch_device`) |
| `PoseStack` | `pose_stack` | `*_pose_stack`, such as `orig_pose_stack` |

Container annotations such as `list[PoseStack]` are exempt; name those as
plurals (`pose_stacks`).

### Vocabulary

Use these names for these objects in public signatures and fields:

| Object | Name |
|---|---|
| `PoseStack` | `pose_stack` |
| `ParameterDatabase` | `param_db` |
| `torch.device` | `device` |
| `CanonicalOrdering` | `canonical_ordering` |
| `FoldForest` | `fold_forest` |
| `MoveMap` | `move_map` |
| `ScoreType` | `score_type` |
| Constraint scoring function | `constraint_function` |
| `RefinedResidueType` in sampler methods | `residue_type` |
| Main chain / side chain | `mainchain` / `sidechain` |
| Integer chain index tensor | `chain_id` (as on `PoseStack`; don't use it for string chain labels) |

## Provisional rules

Provisional rules are proposed but not yet enforced. Pre-commit ignores them;
`python scripts/check_api_style.py --extended` reports them alongside the
checked rules, without failing and without adding them to the baseline. Follow
them in new code where you can, so existing code is the only thing left to
change when a rule is adopted.

To adopt a provisional rule, move it into [Naming](#naming), fix or baseline
the existing violations, and bump the version.

### API005: boolean options are keyword-only (provisional)

**Provisional** ([why](#why-api005)). A boolean parameter with a default value in a public signature is
keyword-only: put `*` before the first one. Every parameter after the `*` is
then keyword-only as well.

```python
# Don't
def pack_rotamers(pose_stack, sfxn, task, verbose=False, seed=None): ...
pack_rotamers(pose_stack, sfxn, task, True)  # True what?

# Do
def pack_rotamers(pose_stack, sfxn, task, *, verbose=False, seed=None): ...
pack_rotamers(pose_stack, sfxn, task, verbose=True)
```

A parameter counts as boolean when it is annotated `bool`, `Optional[bool]` or
`bool | None`, or is unannotated with a `True` or `False` default. Exempt:

- a method whose only parameter is the boolean, such as
  `PackerTask.or_bump_check(setting)`, where the value is the subject of the
  call. Constructors are not exempt: `PackerPalette(True)` doesn't say what
  `True` turns on;
- parameters whose names start with `_`.

Not yet decided: whether unions such as `cuda_graph: bool | str` count.

### API006: no negative flags (provisional)

**Provisional** ([why](#why-api006)). A boolean parameter or field in the
public API names the behaviour it turns on, not the one it turns off. Its name
does not start with `no_`, `not_`, `disable_`, `dont_`, `without_` or `skip_`.
Flip the meaning and the default instead.

```python
# Don't
def build(structure, no_optH: bool = False): ...
build(structure, no_optH=False)  # does this optimize hydrogens? (yes)

# Do
def build(structure, optimize_hydrogens: bool = True): ...
build(structure, optimize_hydrogens=False)
```

A name counts as boolean on the same terms as API005, and fields count when
annotated `bool`. Words that merely begin with "non", such as `non_ideal` or
`noncanonical_design`, are not flagged: they name a property, not the absence
of a behaviour.

Not yet decided: the positive name for `no_optH`. The atom37 builder already
uses `opt_h`; `optimize_hydrogens` would also satisfy API001 and API002.

## Conventions checked in review

These conventions are checked by people in code review, not by
`scripts/check_api_style.py`. Each depends on intent that can't be read from
the code, such as whether a function is a converter, whether a docstring
explains anything, or whether an object is meant to be public, so an automated
check would either miss violations or flag correct code. Apply them to the
code you add or change.

| ID | Rule | Why |
|---|---|---|
| REV1 | Name converters `<output>_from_<input>`: `pose_stack_from_file`, `atom_array_from_pose_stack`, `file_from_pose_stack`. | [why](#why-rev1) |
| REV2 | Give public modules, classes and functions Google- or NumPy-style docstrings; they become the API reference (see the {doc}`contributor guide </contributor_guide>`). | [why](#why-rev2) |
| REV3 | When renaming a public name, update every in-repo caller, test, doc page, tutorial notebook and agent skill in the same change. | [why](#why-rev3) |
| REV4 | Export public objects through an `__all__`. Keep everything else underscore-prefixed. | [why](#why-rev4) |

## Reasoning

Why each rule exists. When a case seems to call for breaking a rule, check
it against the reasoning here; if the reasoning doesn't apply, propose
changing the rule rather than working around it.

### Why FMT1

**Black.** Formatting decided by a tool can't be argued about in review, and
diffs show only real changes rather than re-wrapped lines. It also means code
written by an agent and by a person looks the same.

### Why FMT2

**Flake8.** It catches real mistakes cheaply: undefined and unused names,
shadowed imports, unreachable code. The repository's configuration turns off
checks that conflict with Black, and E711, because `x != None` is a legitimate
element-wise comparison on tensors and arrays.

### Why FMT3

**clang-format.** TMol's C++ and CUDA kernels live next to their Python modules
and are reviewed together, so they get the same mechanical formatting for the
same reasons as FMT1.

### Why API001

**No one- or two-character names.** A name that short can't be understood
without opening the documentation: before this rule, `run_kin_min` took `ff`
and `mm`, `ScoreFunction.set_weight` took `st`, and
`atom_array_from_pose_stack` took `co`. TMol is used from notebooks by
scientists and by agents that read signatures, and both rely on names saying
what they are. The allowed exceptions are established terms (pH, PyTorch's
`.to`), not abbreviations.

### Why API002

**No banned abbreviations.** Each banned form had a spelled-out twin elsewhere
in the same API: `MoveMap` had `set_move_all_mc_tors_for_blocks` next to
`set_move_all_named_torsions_for_blocks`, and some functions took
`torch_device` while the rest took `device`. With one spelling per concept,
names can be guessed, and searching for `sidechain` finds every use. `mc` and
`sc` in particular are opaque outside Rosetta-style code.

### Why API003

**`param_db` before `device`.** 47 of the 53 signatures that take both already
put the database first. A consistent order lets callers rely on position; with
mixed orders, a device passed positionally to one function lands in another
function's `param_db` slot.

### Why API004

**Names follow types.** The same object should have the same name everywhere,
so code reads the same across modules and keyword arguments carry over from one
function to the next. Before this rule a `PoseStack` was called `pose_stack`,
`poses`, `systems` or `ps` depending on the file. Tying the name to the type
annotation also makes the rule checkable without guessing what a parameter
holds. The suffix form exists because some signatures take two objects of one
type (`anneal_device` alongside `device`).

### Why API005

**Keyword-only booleans (provisional).** A positional flag is unreadable at the
call site: in `pose_stack_from_atom37_and_topology(coords, topology, context,
True)`, nothing says what `True` does. Adjacent flags such as `no_optH` and
`backprop_compatible_rebuild` can be swapped by mistake with no error.
Keyword-only flags also make later changes safe: renaming or reordering a flag
then fails loudly with a `TypeError` instead of silently changing what existing
calls do, which matters while negative flags like `no_optH` are under
discussion. A setter's single argument is exempt because there the value is the
subject of the call.

### Why API006

**No negative flags (provisional).** A negative flag makes callers and readers
undo a double negative: `no_optH=False` means "do optimize hydrogens", and code
that branches on it reads `if not no_optH`. Positive names make the default
behaviour explicit in the signature (`optimize_hydrogens=True`) and keep the
option's name the same as the feature's name in docs and logs. TMol already
spells the same option both ways, `no_optH=False` in the atom37 functions and
`opt_h=True` on the atom37 builder, which shows how easily the two drift
apart; one rule picks a direction.

### Why REV1

**`<output>_from_<input>`.** Most converters already follow it, so a user who
knows the input and output types can guess the function name, and the reverse
conversion is easy to find (`atom_array_from_pose_stack`,
`pose_stack_from_atom_array`).

### Why REV2

**Docstrings.** The API reference pages are generated from docstrings, so a
missing or unstructured docstring is a missing or broken reference page. A
consistent style lets Sphinx render parameters and return values as structured
fields.

### Why REV3

**Complete renames.** A partial rename leaves something broken that no unit
test catches: tutorial notebooks only run in their own CI job, and agent skills
are instructions that silently go out of date. Doing the whole rename in one
change also keeps the old name from lingering as a second spelling, which
API002 and API004 exist to prevent.

### Why REV4

**`__all__` and underscores.** They tell users what is supported and what may
change without notice. The style checker also uses `__all__` to find the public
API, so an object missing from `__all__` escapes the naming rules.

## Versioning

The version at the top of this page and `STYLE_GUIDE_VERSION` in
`scripts/check_api_style.py` must match, and the checker also stores a
fingerprint of the [Naming](#naming) and [Provisional rules](#provisional-rules)
sections. If the two disagree, the checker
stops with an error instead of checking against rules it doesn't implement.

To change a checked rule:

1. Edit the Naming or Provisional rules section and increase the version
   number above by one.
2. Update the checker to implement the change.
3. Set `STYLE_GUIDE_VERSION` to the new number and `STYLE_GUIDE_FINGERPRINT`
   to the output of `python scripts/check_api_style.py --guide-fingerprint`.

Edits outside those two sections don't change the fingerprint and need no
version bump.

## The baseline

`scripts/api_style_baseline.txt` lists violations that existed before the
checked rules did. The check fails only on violations not in that file, so old
code doesn't block commits while new code has to follow the rules.

- To fix a baselined violation, fix the code, then run
  `python scripts/check_api_style.py --update-baseline`. The check reminds you
  when the file has entries that are already fixed.
- Never add an entry by hand to get a new violation past the check. If a rule
  is wrong for a case, change the rule (and this page) instead.
- `python scripts/check_api_style.py --all` shows every violation, including
  baselined ones.

## Open decisions

These are inconsistent in the current API but have no agreed convention yet.
Until one is agreed, match the code around your change and don't add a third
variant.

- `sfxn` vs `score_function` for `ScoreFunction` parameters.
- `pose_ind` vs `pose_index` (and `block_ind` vs `block`) for indices.
- Names for PDB-space labels: `chain` vs `chain_label` for chain letters,
  `residue_id` for residue labels with insertion codes, `atom_name` for atom
  names.
- `res_*` vs `residue_*`, and `tor` vs `torsion`.
- Whether non-boolean options after the main arguments are keyword-only
  (boolean options are covered by provisional rule API005).
- The positive replacement name for `no_optH` (negative flags themselves are
  covered by provisional rule API006).
- Whether `get_` prefixes accessors (`get_sequence` vs `residue_label`).
- Whether renamed public parameters keep a deprecated alias for a release.
- Whether the sampler and energy-term extension classes are public API; their
  methods use internal abbreviations (`gbt`, `kto`, `bt_inds`).
