"""Shared versioned score-function weights and options file contract."""

import logging
from tmol_core._yaml import _safe_load

SFXN_FORMAT_VERSION = "1.0"


def load_score_function_record(path, *, logger=None):
    """Read weights/options, preserving the mature format-version contract."""
    if logger is None:
        logger = logging.getLogger(__name__)
    with open(path) as f:
        data = _safe_load(f)

    # --- .sfxn format version check ---
    file_version = data.get("version")
    if file_version is None:
        raise ValueError(
            f"{path}: no 'version' field found in .sfxn file. "
            f"Current format version is {SFXN_FORMAT_VERSION}. "
            f"Regenerate the file with the current version."
        )
    else:
        file_version = str(file_version)
        file_major = file_version.split(".")[0]
        current_major = SFXN_FORMAT_VERSION.split(".")[0]
        if file_major != current_major:
            raise ValueError(
                f"{path}: .sfxn format version {file_version} is incompatible "
                f"with the current format version {SFXN_FORMAT_VERSION}. "
                f"Regenerate the file with the current writer."
            )
        if file_version != SFXN_FORMAT_VERSION:
            logger.info(
                "%s: .sfxn format version %s differs from current %s "
                "(backward-compatible minor version change)",
                path,
                file_version,
                SFXN_FORMAT_VERSION,
            )

    return data
