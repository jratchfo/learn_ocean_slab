from pathlib import Path

import pytest

from learn_ocean_slab import load_config


def test_load_config_reads_values(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        "model: pollard_millard\nlat: -30.0\nH: 80.0\nrho: 1025.0\ntau_x: 0.1\ntau_y: 0.0\n"
        "damping_days: 5.0\nduration_days: 2.0\ndt: 600.0\nu_initial: 0.1\nv_initial: 0.1\n"
        "wind_type: steady\nwind_period_hours: .inf\nwind_duration_hours: .inf\n"
        "gust_std: 0.0\ngust_hours: 1.0\nseed: 0\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.lat == -30.0
    assert config.H == 80.0


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        load_config(path)
