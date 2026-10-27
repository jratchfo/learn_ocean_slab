# learn_ocean_slab

Learning Oceanography with a minimal Python project layout.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/), for `.mp4` movies (`brew install ffmpeg`). Not a Python package, so uv
  can't install it. Without it, `slab_model_movie` can still write a `.gif` with `-o run.gif`.

## Setup (uv)

```bash
uv sync --extra dev
```

## Run tests

```bash
uv run pytest
```

## Example config

Project configuration lives in YAML files, for example `config/midlat_45N.yaml`.
The `model:` field says which slab model the file is for.

## Make a movie

```bash
uv run slab_model_movie config/midlat_45N.yaml
uv run slab_model_movie config/midlat_45N.yaml u_initial=0.5 lat=30 -o fast_start.mp4
```

Any `name=value` after the config file changes that value from the YAML for this run only.
The movie is saved in `movies/` unless `-o` gives another path; `--frame_hours` sets model hours per frame (default 1).
