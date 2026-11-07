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


def sorted_score_terms(term_list, n_score_types, score_types):
    """Order terms by their earliest global channel, retaining native lane order."""
    result, channels = [], []
    covered = [False] * n_score_types
    by_channel = [None] * n_score_types
    for term in term_list:
        for channel in score_types(term):
            by_channel[channel.value] = term
    for term in by_channel:
        if term is not None and not any(covered[st.value] for st in score_types(term)):
            result.append(term)
            for st in score_types(term):
                covered[st.value] = True
                channels.append(st)
    return result, channels


def _term_score_types(term):
    return term.score_types()


def _term_n_bodies(term):
    return term.n_bodies()


class ScoreTermLifecycle:
    """Shared lazy term ownership and channel ordering, without weight storage.

    Adapters supply term construction, scalar weight lookup and cache invalidation.
    Removing a term releases this registry's reference; existing rendered scorers
    keep their term objects and immutable weight snapshots.
    """

    def __init__(
        self, n_score_types, *, score_types=_term_score_types, n_bodies=_term_n_bodies
    ):
        self._n_score_types = n_score_types
        self._term_score_types = score_types
        self._term_n_bodies = n_bodies
        self._term_for_st = [None] * n_score_types
        self._terms_version = 0
        self._all_score_types = []
        for group in ("all", "one_body", "two_body", "multi_body"):
            setattr(self, f"_{group}_terms", [])
            setattr(self, f"_{group}_terms_unordered", [])
            setattr(self, f"_{group}_terms_out_of_date", False)

    def _create_term_for_score_type(self, st):
        raise NotImplementedError

    def _term_membership_changed(self):
        """Frontend hook for invalidating device weight layouts or executors."""

    def get_weight(self, st):
        raise NotImplementedError

    def update_term_membership(self, st, weight):
        """Apply the mature lazy-create/final-zero-remove policy before storing weight."""
        present = self.score_type_covered_by_contained_term(st)
        if weight == 0 and not present:
            return
        if not present:
            self.retrieve_term_for_score_type(st)
        if weight == 0 and self.term_for_st_has_no_other_non_zero_weights(st):
            self.remove_term_for_score_type(st)

    def score_type_covered_by_contained_term(self, st):
        return self._term_for_st[st.value] is not None

    def _body_group(self, term):
        n = self._term_n_bodies(term)
        return "one_body" if n == 1 else "two_body" if n == 2 else "multi_body"

    def _change_membership(self, term, operation):
        for group in ("all", self._body_group(term)):
            getattr(getattr(self, f"_{group}_terms_unordered"), operation)(term)
            setattr(self, f"_{group}_terms_out_of_date", True)
        self._terms_version += 1
        self._term_membership_changed()

    def retrieve_term_for_score_type(self, st):
        term = self._create_term_for_score_type(st)
        assert st in self._term_score_types(term)
        for provided in self._term_score_types(term):
            self._term_for_st[provided.value] = term
        self._change_membership(term, "append")

    def term_for_st_has_no_other_non_zero_weights(self, st):
        term = self._term_for_st[st.value]
        return not any(
            self.get_weight(other) != 0
            for other in self._term_score_types(term)
            if other != st
        )

    def remove_term_for_score_type(self, st):
        """Release the term covering st; shared channels lose coverage together."""
        term = self._term_for_st[st.value]
        if term is None:
            return
        for covered in self._term_score_types(term):
            self._term_for_st[covered.value] = None
        self._change_membership(term, "remove")

    def _ordered_terms(self, group):
        if getattr(self, f"_{group}_terms_out_of_date"):
            terms, channels = sorted_score_terms(
                getattr(self, f"_{group}_terms_unordered"),
                self._n_score_types,
                self._term_score_types,
            )
            setattr(self, f"_{group}_terms", terms)
            setattr(self, f"_{group}_terms_out_of_date", False)
            if group == "all":
                self._all_score_types = channels
        return getattr(self, f"_{group}_terms")

    def all_terms(self):
        """Return the active terms in score-type order; do not mutate the list."""
        return self._ordered_terms("all")

    def all_score_types(self):
        """Channels in term order, preserving each term's native subterm order."""
        self.all_terms()
        return self._all_score_types

    def one_body_terms(self):
        return self._ordered_terms("one_body")

    def two_body_terms(self):
        return self._ordered_terms("two_body")

    def multi_body_terms(self):
        return self._ordered_terms("multi_body")
