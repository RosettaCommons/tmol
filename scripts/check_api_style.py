#!/usr/bin/env python3
"""Check TMol's public API against the naming rules in docs/style_guide.md.

The public API is every name listed in an ``__all__`` of a ``tmol`` package,
followed to the module that defines it. For each public function, class,
public method and public class field, the checker applies the rules below.
It reads source with ``ast`` only, so it needs neither a build nor an import
of TMol, and runs in well under a second.

Rules (IDs match docs/style_guide.md):

  API001  Public names (functions, methods, parameters, fields) are longer
          than two characters, apart from the allowlist in SHORT_NAME_ALLOWED.
  API002  Public names do not use the banned abbreviations in BANNED_TOKENS
          or BANNED_NAMES.
  API003  A signature taking both ``param_db`` and ``device`` lists
          ``param_db`` first.
  API004  A parameter's name matches its annotated type: ``ParameterDatabase``
          is ``param_db``; ``torch.device`` is ``device`` or ends in
          ``_device``; ``PoseStack`` is ``pose_stack`` or ends in
          ``_pose_stack``.

Provisional rules (reported only with --extended; never fail, never baselined):

  API005  Boolean parameters with a default are keyword-only, except a
          method's sole parameter and ``_``-prefixed parameters.
  API006  Boolean parameters and fields have positive names: none starts
          with a prefix in NEGATIVE_PREFIXES.

Violations that predate a rule live in scripts/api_style_baseline.txt so the
check fails only on new ones. Fix a baselined violation, then run with
``--update-baseline`` to drop it from the file.

Usage:
  python scripts/check_api_style.py                    # pre-commit mode
  python scripts/check_api_style.py --all              # ignore the baseline
  python scripts/check_api_style.py --extended         # also report provisional rules
  python scripts/check_api_style.py --update-baseline  # rewrite the baseline
  python scripts/check_api_style.py --guide-fingerprint  # values for a new guide version

Before checking anything, the script confirms it implements the current
docs/style_guide.md: the guide's "Version: N" must equal STYLE_GUIDE_VERSION,
and its Naming and Provisional rules sections must hash to
STYLE_GUIDE_FINGERPRINT. Otherwise it exits
with status 2 and asks for the script to be updated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "scripts" / "api_style_baseline.txt"
EXCLUDED_DIRS = {"tests", "extern"}

# The docs/style_guide.md version this checker implements, and the fingerprint
# of that guide's Naming and Provisional rules sections (see "Versioning" in
# the guide). After
# changing the rules, update both; --guide-fingerprint prints the new values.
STYLE_GUIDE = Path("docs") / "style_guide.md"
STYLE_GUIDE_VERSION = 2
STYLE_GUIDE_FINGERPRINT = (
    "f5e294fc769e80e890f4ce75e64b2d4418e60e46409662063c6bc46f9ebf2600"
)

# Sections of the guide whose text the fingerprint covers.
GUIDE_RULE_SECTIONS = ("Naming", "Provisional rules")

# Rules from the guide's Provisional rules section: reported only with --extended.
PROVISIONAL_RULES = {"API005", "API006"}

# API006: leading name parts that make a flag negative. "non" is deliberately
# absent: non_ideal and noncanonical name properties, not disabled behaviour.
NEGATIVE_PREFIXES = {"no", "not", "disable", "dont", "without", "skip"}

# API001: two-character names that are the established term, not an abbreviation.
SHORT_NAME_ALLOWED = {
    "ph",  # pH, the standard chemistry term
    "to",  # .to(device), matching torch.Tensor.to
}

# API002: banned underscore-separated name parts, mapped to the replacement.
BANNED_TOKENS = {
    "mc": "mainchain",
    "sc": "sidechain",
    "mcsc": "mainchain_or_sidechain",
    "mcs": "mainchain_torsions",
    "scs": "sidechain_torsions",
}

# API002: banned whole names, mapped to the replacement.
BANNED_NAMES = {
    "torch_device": "device",
    "co": "canonical_ordering",
    "ff": "fold_forest",
    "mm": "move_map",
    "fn": "constraint_function",
    "st": "score_type",
    "rt": "residue_type",
    "bt": "block_type",
}

# API004: annotated type -> (exact name, allowed suffix or None, banned names).
TYPED_NAMES = {
    "ParameterDatabase": ("param_db", None, set()),
    "device": ("device", "_device", {"torch_device"}),
    "PoseStack": ("pose_stack", "_pose_stack", set()),
}


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str  # repo-relative source file
    line: int
    owner: str  # "func", "Class.method" or "Class"
    detail: str  # what is wrong, e.g. "parameter 'rt'"
    hint: str

    @property
    def key(self) -> str:
        """Line-number-free identity used by the baseline file."""
        return f"{self.rule} {self.path}::{self.owner} {self.detail}"

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.owner}: {self.detail}; {self.hint}"


# --------------------------------------------------------------------------
# Locating the public API
# --------------------------------------------------------------------------


def _module_file(root: Path, module: str) -> Path | None:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _resolve_relative(package: str, level: int, module: str | None) -> str:
    parts = package.split(".")
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    return ".".join(parts + ([module] if module else []))


def _string_list(node: ast.AST) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts
    ):
        return [e.value for e in node.elts]
    return None


def _package_exports(init_file: Path, package: str) -> dict[str, str]:
    """Map each name in the package's ``__all__`` to the module that provides it."""
    tree = ast.parse(init_file.read_text())
    exported: list[str] = []
    source: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exported = _string_list(node.value) or []
        elif isinstance(node, ast.ImportFrom):
            module = (
                _resolve_relative(package, node.level, node.module)
                if node.level
                else node.module
            )
            for alias in node.names:
                source[alias.asname or alias.name] = module
        elif isinstance(node, ast.Dict):
            # lazy maps: {"Name": "tmol.module"}
            for k, v in zip(node.keys, node.values):
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(v, ast.Constant)
                    and isinstance(k.value, str)
                    and isinstance(v.value, str)
                    and v.value.startswith("tmol")
                ):
                    source[k.value] = v.value
        elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
            # lazy maps: ("tmol.module", ("Name", ...))
            first, names = node.elts
            listed = _string_list(names)
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.startswith("tmol")
                and listed
            ):
                for name in listed:
                    source[name] = first.value
    return {name: source.get(name, package) for name in exported}


def _find_definition(root: Path, module: str, name: str, depth: int = 0):
    """Return (file, node) for the def/class ``name`` provided by ``module``."""
    path = _module_file(root, module)
    if path is None or depth > 5:
        return None
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            return path, node
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    target = (
                        _resolve_relative(package, node.level, node.module)
                        if node.level
                        else node.module
                    )
                    return _find_definition(root, target, alias.name, depth + 1)
    return None


def public_definitions(root: Path):
    """Yield (file, node) for every public def/class, each source object once."""
    seen = set()
    for init_file in sorted(root.glob("tmol/**/__init__.py")):
        rel = init_file.relative_to(root)
        if EXCLUDED_DIRS & set(rel.parts):
            continue
        package = ".".join(rel.parts[:-1])
        for name, module in _package_exports(init_file, package).items():
            found = _find_definition(root, module, name)
            if found is None:
                continue
            path, node = found
            key = (path, node.lineno)
            if key not in seen:
                seen.add(key)
                yield path, node


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


def _parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    a = func.args
    params = a.posonlyargs + a.args + a.kwonlyargs
    return [p for p in params if p.arg not in ("self", "cls")]


def _annotation_type(annotation: ast.AST | None) -> str | None:
    """The TYPED_NAMES key an annotation refers to, if any."""
    if annotation is None:
        return None
    text = ast.unparse(annotation)
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        text = annotation.value
    # Only plain or Optional annotations, never containers such as list[PoseStack].
    if re.search(r"\b(list|List|Sequence|tuple|Tuple|dict|Dict|Mapping)\b", text):
        return None
    for type_name in TYPED_NAMES:
        if re.search(rf"(^|[.\s|\[]){type_name}\b", text):
            return type_name
    return None


def _name_rules(name: str, what: str):
    """API001/API002 for one name. Yields (rule, detail, hint)."""
    if name.startswith("_"):
        return
    if len(name) <= 2 and name not in SHORT_NAME_ALLOWED:
        hint = BANNED_NAMES.get(name, "spell the name out")
        yield "API001", f"{what} {name!r}", (
            f"use {hint!r}" if name in BANNED_NAMES else hint
        )
        return
    if name in BANNED_NAMES:
        yield "API002", f"{what} {name!r}", f"use {BANNED_NAMES[name]!r}"
        return
    parts = name.split("_")
    bad = [p for p in parts if p in BANNED_TOKENS]
    if bad:
        fixed = "_".join(BANNED_TOKENS.get(p, p) for p in parts)
        yield "API002", f"{what} {name!r}", f"spell out {', '.join(bad)} (e.g. {fixed!r})"


def _check_function(func, owner: str):
    for rule, detail, hint in _name_rules(func.name, "name"):
        yield func.lineno, owner, rule, detail, hint
    params = _parameters(func)
    names = [p.arg for p in params]
    for p in params:
        for rule, detail, hint in _name_rules(p.arg, "parameter"):
            yield p.lineno, owner, rule, detail, hint
        type_name = _annotation_type(p.annotation)
        if type_name:
            exact, suffix, banned = TYPED_NAMES[type_name]
            ok = p.arg == exact or (
                suffix and p.arg.endswith(suffix) and p.arg not in banned
            )
            if not ok and p.arg not in BANNED_NAMES:  # API002 already reports those
                want = repr(exact) + (f" or '*{suffix}'" if suffix else "")
                yield p.lineno, owner, "API004", f"parameter {p.arg!r} ({type_name})", f"name it {want}"
    db = next((i for i, n in enumerate(names) if n == "param_db"), None)
    dev = next(
        (i for i, n in enumerate(names) if n in ("device", "torch_device")), None
    )
    if db is not None and dev is not None and dev < db:
        yield func.lineno, owner, "API003", "device before param_db", "put param_db first"
    yield from _check_keyword_only_flags(func, owner)
    yield from _check_negative_flags(func, owner)


def _is_bool_annotation(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    text = ast.unparse(annotation)
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        text = annotation.value
    return bool(
        re.fullmatch(
            r"\s*(bool|Optional\[bool\]|bool\s*\|\s*None|None\s*\|\s*bool)\s*", text
        )
    )


def _is_bool_parameter(arg: ast.arg, default: ast.AST | None) -> bool:
    if arg.annotation is None:
        return isinstance(default, ast.Constant) and isinstance(default.value, bool)
    return _is_bool_annotation(arg.annotation)


def _is_negative_name(name: str) -> bool:
    return not name.startswith("_") and name.split("_")[0].lower() in NEGATIVE_PREFIXES


_NEGATIVE_HINT = "name the behaviour it turns on and flip the default"


def _check_negative_flags(func, owner: str):
    """API006 (provisional): boolean parameters have positive names."""
    a = func.args
    positional = a.posonlyargs + a.args
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    pairs = list(zip(positional, defaults)) + list(zip(a.kwonlyargs, a.kw_defaults))
    for p, default in pairs:
        if _is_negative_name(p.arg) and _is_bool_parameter(p, default):
            yield p.lineno, owner, "API006", f"parameter {p.arg!r}", _NEGATIVE_HINT


def _check_keyword_only_flags(func, owner: str):
    """API005 (provisional): defaulted boolean parameters are keyword-only."""
    a = func.args
    positional = a.posonlyargs + a.args
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    pairs = [
        (p, d) for p, d in zip(positional, defaults) if p.arg not in ("self", "cls")
    ]
    is_method = bool(a.posonlyargs + a.args) and (a.posonlyargs + a.args)[0].arg in (
        "self",
        "cls",
    )
    if is_method and func.name != "__init__" and len(pairs) == 1 and not a.kwonlyargs:
        return  # the boolean is the method's only parameter, e.g. or_bump_check(setting)
    for p, default in pairs:
        if default is None or p.arg.startswith("_"):
            continue
        if _is_bool_parameter(p, default):
            yield p.lineno, owner, "API005", f"parameter {p.arg!r} is positional", (
                f"put '*' before {p.arg!r} so it must be passed by keyword"
            )


def check_definition(path: Path, node, root: Path) -> list[Violation]:
    rel = path.relative_to(root).as_posix()
    found = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        found += list(_check_function(node, node.name))
    else:
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "__init__":
                    owner = node.name
                    for line, o, rule, detail, hint in _check_function(item, owner):
                        if detail != "name '__init__'":
                            found.append((line, o, rule, detail, hint))
                elif not item.name.startswith("_"):
                    found += list(_check_function(item, f"{node.name}.{item.name}"))
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                for rule, detail, hint in _name_rules(item.target.id, "field"):
                    found.append((item.lineno, node.name, rule, detail, hint))
                if _is_negative_name(item.target.id) and _is_bool_annotation(
                    item.annotation
                ):
                    found.append(
                        (
                            item.lineno,
                            node.name,
                            "API006",
                            f"field {item.target.id!r}",
                            _NEGATIVE_HINT,
                        )
                    )
    return [
        Violation(rule, rel, line, owner, detail, hint)
        for line, owner, rule, detail, hint in found
    ]


def collect_violations(root: Path, extended: bool = False) -> list[Violation]:
    """Violations of the checked rules, plus provisional ones if ``extended``."""
    violations = []
    for path, node in public_definitions(root):
        violations += check_definition(path, node, root)
    if not extended:
        violations = [v for v in violations if v.rule not in PROVISIONAL_RULES]
    unique = {v.key: v for v in violations}
    return sorted(unique.values(), key=lambda v: (v.path, v.line, v.key))


# --------------------------------------------------------------------------
# Style guide version
# --------------------------------------------------------------------------


class GuideMismatch(Exception):
    """The checker does not implement the rules in the current style guide."""


def guide_version(text: str) -> int:
    match = re.search(r"^Version:\s*(\d+)\s*$", text, re.MULTILINE)
    if match is None:
        raise GuideMismatch(f"{STYLE_GUIDE} has no 'Version: N' line")
    return int(match.group(1))


def guide_fingerprint(text: str) -> str:
    """SHA-256 of the guide's rule sections (GUIDE_RULE_SECTIONS), in order."""
    lines = []
    for title in GUIDE_RULE_SECTIONS:
        match = re.search(
            rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise GuideMismatch(f"{STYLE_GUIDE} has no '## {title}' section")
        lines.append(f"## {title}")
        lines += [line.rstrip() for line in match.group(1).strip().splitlines()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def check_guide(root: Path) -> None:
    """Raise GuideMismatch unless this script matches the guide under ``root``."""
    path = root / STYLE_GUIDE
    if not path.is_file():
        raise GuideMismatch(f"{STYLE_GUIDE} not found under {root}")
    text = path.read_text()
    version = guide_version(text)
    fingerprint = guide_fingerprint(text)
    update = (
        "Update scripts/check_api_style.py to implement it, then set "
        "STYLE_GUIDE_VERSION and STYLE_GUIDE_FINGERPRINT to the values printed "
        "by: python scripts/check_api_style.py --guide-fingerprint"
    )
    if version != STYLE_GUIDE_VERSION:
        raise GuideMismatch(
            f"{STYLE_GUIDE} is version {version}, but scripts/check_api_style.py "
            f"implements version {STYLE_GUIDE_VERSION}. {update}"
        )
    if fingerprint != STYLE_GUIDE_FINGERPRINT:
        raise GuideMismatch(
            f"The Naming or Provisional rules section of {STYLE_GUIDE} changed, "
            f"but its version is still {version}. Increase the version in "
            f"{STYLE_GUIDE} (if you already did, only STYLE_GUIDE_FINGERPRINT is "
            f"stale). {update}"
        )


# --------------------------------------------------------------------------
# Baseline and CLI
# --------------------------------------------------------------------------

BASELINE_HEADER = """\
# Known violations of scripts/check_api_style.py that predate the rules.
# The pre-commit check fails only on violations NOT listed here.
# Fix an entry, then run: python scripts/check_api_style.py --update-baseline
# Never add entries by hand to get a new violation past the check.
"""


def read_baseline(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def write_baseline(path: Path, violations: list[Violation]) -> None:
    body = "\n".join(sorted(v.key for v in violations))
    path.write_text(BASELINE_HEADER + body + ("\n" if body else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="ignore the baseline")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the baseline to the current violations",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="also report provisional rules (informational; never fails)",
    )
    parser.add_argument(
        "--guide-fingerprint",
        action="store_true",
        help="print the style guide's version and Naming-section fingerprint",
    )
    args, _ = parser.parse_known_args(argv)  # pre-commit may pass filenames
    root = args.root.resolve()
    baseline_path = args.baseline or root / "scripts" / "api_style_baseline.txt"

    if args.guide_fingerprint:
        text = (root / STYLE_GUIDE).read_text()
        print(f"STYLE_GUIDE_VERSION = {guide_version(text)}")
        print(f'STYLE_GUIDE_FINGERPRINT = "{guide_fingerprint(text)}"')
        return 0
    try:
        check_guide(root)
    except GuideMismatch as error:
        print(f"error: checker out of date with style guide: {error}", file=sys.stderr)
        return 2

    found = collect_violations(root, extended=args.extended)
    violations = [v for v in found if v.rule not in PROVISIONAL_RULES]
    provisional = [v for v in found if v.rule in PROVISIONAL_RULES]
    if args.update_baseline:
        # the baseline only ever holds checked rules
        write_baseline(baseline_path, violations)
        print(f"wrote {len(violations)} entries to {baseline_path}")
        return 0

    baseline = set() if args.all else read_baseline(baseline_path)
    new = [v for v in violations if v.key not in baseline]
    stale = baseline - {v.key for v in violations}

    for v in new:
        print(v.format())
    if provisional:
        print(f"\nProvisional rules ({len(provisional)}; informational, not enforced):")
        for v in provisional:
            print(v.format())
    if stale:
        print(
            f"note: {len(stale)} baseline entr{'y is' if len(stale) == 1 else 'ies are'} "
            "fixed; run with --update-baseline to remove them",
            file=sys.stderr,
        )
    if new:
        print(
            f"\n{len(new)} API style violation(s); see docs/style_guide.md",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
