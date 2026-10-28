# learn_ocean_slab

Learning Oceanography with a minimal Python project layout.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jratchfo/learn_ocean_slab/blob/main/notebooks/pollard_millard_inertial_oscillations.ipynb)
Runs in Google Colab with no setup: the notebook's first cell fetches the code.

## Requirements

- Python 3.12+ (developed on 3.14; 3.12 is what Colab runs)
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/), for `.mp4` movies (`brew install ffmpeg`). Not a Python package, so uv
  can't install it. Without it, `slab_model_movie` can still write a `.gif` with `-o run.gif`.

## First-time setup

```bash
git clone https://github.com/jratchfo/learn_ocean_slab.git
cd learn_ocean_slab
uv sync --extra dev
```

If you will commit changes to the notebooks, also run these once in your clone. They make git
remove notebook outputs (plots, movies) on `git add`, so commits contain only code and text:

```bash
git config filter.nbstripout.clean "uv run --quiet nbstripout --keep-id"
git config filter.nbstripout.smudge cat
git config diff.ipynb.textconv "uv run --quiet nbstripout --keep-id -t"
```

These settings live in your clone's `.git/config`, which git never copies between clones, so
each clone needs them once. Skipping them breaks nothing; your notebook commits just include outputs.

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
