"""Path relocation must keep historical input identities unambiguous."""

import json
from pathlib import Path

import pytest

from notesep.training_paths import configure, relocated


def test_longest_whole_prefix_wins(tmp_path):
    data = tmp_path / "data"
    recorded = tmp_path / "recorded"
    mapping = tmp_path / "paths.json"
    mapping.write_text(json.dumps({"/original": str(data),
                                   "/original/recorded": str(recorded)}))
    try:
        configure(mapping)
        assert relocated("/original/index.sqlite") == data / "index.sqlite"
        assert relocated("/original/recorded/shard-00/FROZEN.json") == recorded / "shard-00/FROZEN.json"
        assert relocated("/originally/index.sqlite") == Path("/originally/index.sqlite")
    finally:
        configure()


def test_mapping_requires_absolute_original_prefix(tmp_path):
    mapping = tmp_path / "paths.json"
    mapping.write_text(json.dumps({"original": "data"}))
    with pytest.raises(ValueError, match="absolute"):
        configure(mapping)
    configure()
