"""Command line entry point: run a slab model from a YAML file and save a movie of the run.

    slab_model_movie config/midlat_45N.yaml u_initial=0.5 lat=30 -o fast_start.mp4

The YAML's `model:` field picks the slab model from MODELS.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from . import pm_slab
from .config import MOVIE_DIR, RunConfig, load_config
from .movie import make_movie

# model name in the YAML -> (its config dataclass, its run function returning an xarray Dataset)
MODELS = {
    "pollard_millard": (RunConfig, pm_slab.run),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="slab_model_movie",
        description="Run a slab model and save a movie of the run.",
        epilog="Example: slab_model_movie config/midlat_45N.yaml u_initial=0.5 lat=30 -o fast_start.mp4",
    )
    parser.add_argument("config", type=Path, help="YAML file with a model: field and every field that model needs")
    parser.add_argument("overrides", nargs="*", help="changes to the YAML values, written name=value")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="movie file, .mp4 or .gif (default: config name plus overrides, .mp4, in the project's movies folder)",
    )
    parser.add_argument("--frame_hours", type=float, default=1.0, help="model hours per movie frame (default 1)")
    parser.add_argument("--fps", type=float, default=15, help="movie frames per second (default 15)")
    args = parser.parse_intermixed_args(argv)  # lets overrides come before or after -o

    try:
        model = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)).get("model")
        if model not in MODELS:
            parser.error(f"model: {model!r} is not one of {', '.join(MODELS)}")
        config_class, run = MODELS[model]
        config = load_config(args.config, args.overrides, config_class)
    except OmegaConfBaseException as error:
        parser.error(str(error))
    print(config)

    # e.g. midlat_45N_u_initial0.5_lat30.mp4, so runs with different overrides don't overwrite each other
    name = "_".join([args.config.stem, *(o.replace("=", "") for o in args.overrides)]) + ".mp4"
    output = args.output or MOVIE_DIR / name
    path = make_movie(run(config), output, frame_hours=args.frame_hours, fps=args.fps)
    print(f"saved {path}")
