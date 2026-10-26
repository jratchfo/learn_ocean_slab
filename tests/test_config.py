from pathlib import Path

import pytest

from learn_ocean_slab import load_config


def test_load_config_reads_mapping(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("dataset:\n  name: sample\n", encoding="utf-8")

    config = load_config(path)

    assert config["dataset"]["name"] == "sample"


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        load_config(path)
