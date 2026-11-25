"""Tests for scripts/check_api_style.py, run against a synthetic package."""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_api_style  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


@pytest.fixture
def repo(tmp_path):
    """A fake tmol tree exporting one function, a lazily mapped class, and a private helper."""
    guide = check_api_style.REPO_ROOT / check_api_style.STYLE_GUIDE
    _write(tmp_path, check_api_style.STYLE_GUIDE.as_posix(), guide.read_text())
    _write(
        tmp_path,
        "tmol/__init__.py",
        """
        __all__ = ["good", "Thing"]
        from tmol._impl import good
        _LAZY = {"Thing": "tmol.sub._thing"}
        """,
    )
    _write(
        tmp_path,
        "tmol/_impl.py",
        """
        def good(pose_stack: "PoseStack", param_db: "ParameterDatabase", device):
            pass

        def _private(st, mc):  # private: never checked
            pass
        """,
    )
    _write(tmp_path, "tmol/sub/__init__.py", "")
    _write(
        tmp_path,
        "tmol/sub/_thing.py",
        """
        class Thing:
            ph: float
            def __init__(self, device, param_db):
                pass
            def set_move_mc_tor(self, rt, poses: "PoseStack | None"):
                pass
            def to(self, device):
                pass
        """,
    )
    _write(tmp_path, "tmol/tests/__init__.py", '__all__ = ["bad"]\ndef bad(x): pass\n')
    return tmp_path


def _rules(violations):
    return sorted((v.rule, v.owner, v.detail) for v in violations)


def test_reports_each_rule_on_public_names_only(repo):
    found = _rules(check_api_style.collect_violations(repo))
    assert found == [
        ("API001", "Thing.set_move_mc_tor", "parameter 'rt'"),
        ("API002", "Thing.set_move_mc_tor", "name 'set_move_mc_tor'"),
        ("API003", "Thing", "device before param_db"),
        ("API004", "Thing.set_move_mc_tor", "parameter 'poses' (PoseStack)"),
    ]


def test_allowed_short_names_and_suffixes(repo):
    _write(
        repo,
        "tmol/_impl.py",
        """
        def good(orig_pose_stack: "PoseStack", anneal_device: "torch.device", ph=7.4):
            pass
        """,
    )
    found = _rules(check_api_style.collect_violations(repo))
    assert not [f for f in found if f[1] == "good"]


def test_torch_device_reported_once_as_banned_name(repo):
    _write(
        repo,
        "tmol/_impl.py",
        "def good(param_db, torch_device: 'torch.device'):\n    pass\n",
    )
    found = [
        f for f in _rules(check_api_style.collect_violations(repo)) if f[1] == "good"
    ]
    assert found == [("API002", "good", "parameter 'torch_device'")]


def test_container_annotations_are_not_typed_names(repo):
    _write(repo, "tmol/_impl.py", "def good(poses: 'list[PoseStack]'):\n    pass\n")
    found = [
        f for f in _rules(check_api_style.collect_violations(repo)) if f[1] == "good"
    ]
    assert found == []


def test_baseline_suppresses_known_and_fails_on_new(repo, capsys):
    baseline = repo / "baseline.txt"
    args = ["--root", str(repo), "--baseline", str(baseline)]
    assert check_api_style.main(args) == 1
    assert check_api_style.main(args + ["--update-baseline"]) == 0
    assert check_api_style.main(args) == 0

    _write(repo, "tmol/_impl.py", "def good(fn):\n    pass\n")
    capsys.readouterr()
    assert check_api_style.main(args) == 1
    out = capsys.readouterr().out
    assert (
        "tmol/_impl.py:1: API001 good: parameter 'fn'; use 'constraint_function'" in out
    )


def test_fixed_baseline_entries_are_reported_but_pass(repo, capsys):
    baseline = repo / "baseline.txt"
    args = ["--root", str(repo), "--baseline", str(baseline)]
    check_api_style.main(args + ["--update-baseline"])
    _write(
        repo,
        "tmol/sub/_thing.py",
        "class Thing:\n    def __init__(self, param_db, device):\n        pass\n",
    )
    assert check_api_style.main(args) == 0
    assert "fixed; run with --update-baseline" in capsys.readouterr().err


def _edit_guide(repo, old, new):
    path = repo / check_api_style.STYLE_GUIDE
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def test_guide_version_bump_requires_checker_update(repo, capsys):
    _edit_guide(
        repo,
        f"Version: {check_api_style.STYLE_GUIDE_VERSION}",
        f"Version: {check_api_style.STYLE_GUIDE_VERSION + 1}",
    )
    assert check_api_style.main(["--root", str(repo)]) == 2
    err = capsys.readouterr().err
    assert "out of date with style guide" in err
    assert "Update scripts/check_api_style.py" in err


def test_rule_change_without_version_bump_is_an_error(repo, capsys):
    _edit_guide(repo, "| `bt` | `block_type` | whole name |\n", "")
    assert check_api_style.main(["--root", str(repo)]) == 2
    assert "Increase the version" in capsys.readouterr().err


def test_edits_outside_naming_section_need_no_bump(repo):
    _edit_guide(repo, "## Open decisions\n", "## Open decisions\n\nA new note.\n")
    _write(repo, "tmol/_impl.py", "def good(param_db, device):\n    pass\n")
    _write(repo, "tmol/sub/_thing.py", "class Thing:\n    pass\n")
    assert check_api_style.main(["--root", str(repo)]) == 0


def test_missing_guide_is_an_error(repo, capsys):
    (repo / check_api_style.STYLE_GUIDE).unlink()
    assert check_api_style.main(["--root", str(repo)]) == 2
    assert "not found" in capsys.readouterr().err


def test_guide_fingerprint_matches_checker(capsys):
    assert check_api_style.main(["--guide-fingerprint"]) == 0
    out = capsys.readouterr().out
    assert f"STYLE_GUIDE_VERSION = {check_api_style.STYLE_GUIDE_VERSION}" in out
    assert check_api_style.STYLE_GUIDE_FINGERPRINT in out


FLAGS = """
class Task:
    def __init__(self, design=False):
        pass
    def or_bump_check(self, setting: bool = True):
        pass
    def build(self, pose_stack, verbose: bool = False, *, seed=None):
        pass
    def run(self, pose_stack, *, verbose: bool = False):
        pass
    def ramp(self, pose_stack, ramp: "Optional[bool]" = None, _debug=False):
        pass
"""


def _flag_repo(repo):
    _write(repo, "tmol/_impl.py", "def good(param_db, device):\n    pass\n")
    _write(repo, "tmol/sub/_thing.py", FLAGS.replace("class Task", "class Thing"))
    return repo


def test_provisional_rules_hidden_by_default(repo):
    found = check_api_style.collect_violations(_flag_repo(repo))
    assert not [v for v in found if v.rule in check_api_style.PROVISIONAL_RULES]


def test_extended_reports_keyword_only_flags(repo):
    found = check_api_style.collect_violations(_flag_repo(repo), extended=True)
    assert _rules(v for v in found if v.rule == "API005") == [
        ("API005", "Thing", "parameter 'design' is positional"),
        ("API005", "Thing.build", "parameter 'verbose' is positional"),
        ("API005", "Thing.ramp", "parameter 'ramp' is positional"),
    ]


def test_extended_never_fails_or_extends_baseline(repo, capsys):
    _flag_repo(repo)
    baseline = repo / "baseline.txt"
    args = ["--root", str(repo), "--baseline", str(baseline)]
    assert check_api_style.main(args + ["--extended"]) == 0
    assert "Provisional rules (3; informational" in capsys.readouterr().out
    check_api_style.main(args + ["--extended", "--update-baseline"])
    assert "API005" not in baseline.read_text()


NEGATIVE = """
class Thing:
    skip_protonation: bool
    non_ideal: bool
    no_count: int
    def build(self, pose_stack, *, no_optH: bool = False, optimize: bool = True):
        pass
    def load(self, disable_cache=False, non_polar=False, dont_check: "Optional[bool]" = None):
        pass
    def size(self, no_atoms: int = 0):
        pass
"""


def test_extended_reports_negative_flags(repo):
    _write(repo, "tmol/_impl.py", "def good(param_db, device):\n    pass\n")
    _write(repo, "tmol/sub/_thing.py", NEGATIVE)
    found = check_api_style.collect_violations(repo, extended=True)
    assert _rules(v for v in found if v.rule == "API006") == [
        ("API006", "Thing", "field 'skip_protonation'"),
        ("API006", "Thing.build", "parameter 'no_optH'"),
        ("API006", "Thing.load", "parameter 'disable_cache'"),
        ("API006", "Thing.load", "parameter 'dont_check'"),
    ]


def test_provisional_section_change_needs_version_bump(repo, capsys):
    _edit_guide(repo, "Not yet decided:", "Still not decided:")
    assert check_api_style.main(["--root", str(repo)]) == 2
    assert "Increase the version" in capsys.readouterr().err


def test_repository_has_no_new_violations():
    assert check_api_style.main([]) == 0
