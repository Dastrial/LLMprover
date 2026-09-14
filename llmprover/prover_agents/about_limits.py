"""Numeric limits shared by About lookup execution and prompt protocols."""

from __future__ import annotations

MAX_SEARCH_COMMANDS = 3
# Hard cap when collecting Search hits for the selection prompt.
MAX_SEARCH_HITS_COLLECT = 1000
# If total unique hits >= this, ask the model to pick at most MAX_SELECTED_SEARCH_HITS.
SEARCH_SELECTION_THRESHOLD = 10
MAX_SELECTED_SEARCH_HITS = 10
