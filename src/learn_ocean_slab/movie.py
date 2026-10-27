"""Animation of a slab model run: hodograph, parcel path, current speed and wind over time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

from .config import SECONDS_PER_DAY


def _square_limits(ax, z):
    """Equal-aspect axis limits that hold every point of the complex array z, with a margin."""
    middle = (z.real.max() + z.real.min()) / 2 + 1j * (z.imag.max() + z.imag.min()) / 2
    half = 0.55 * max(np.ptp(z.real), np.ptp(z.imag), 1e-9)
    ax.set_xlim(middle.real - half, middle.real + half)
    ax.set_ylim(middle.imag - half, middle.imag + half)
    ax.set_aspect("equal")
    return half


def make_movie(results, path, frame_hours=1.0, fps=30, dpi=100):
    """Save an animation of a run from pm_slab.run to path.

    The file extension picks the format: .mp4 (made with ffmpeg) or .gif.
    One frame every frame_hours of model time. The wind arrow on the hodograph points along
    the wind stress and is colored by the wind work tau . u: red when the wind puts energy
    into the current, blue when it takes energy out.
    """
    days = results.time.values / SECONDS_PER_DAY
    s = (results.u.values + 1j * results.v.values) * 100  # current [cm/s]
    s_steady = (results.u_steady.values + 1j * results.v_steady.values) * 100  # [cm/s]
    tau = results.tau_x.values + 1j * results.tau_y.values  # [N/m^2]
    position = (results.x.values + 1j * results.y.values) / 1000  # [km]
    wind_work = results.wind_work.values  # [W/m^2]
    speed = np.abs(s)

    dt = results.time.values[1] - results.time.values[0]
    step = max(1, round(frame_hours * 3600 / dt))
    frames = list(range(0, len(days), step))

    fig = plt.figure(figsize=(12, 9), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1])
    ax_hodo = fig.add_subplot(grid[0, 0])
    ax_path = fig.add_subplot(grid[0, 1])
    ax_speed = fig.add_subplot(grid[1, :])
    ax_wind = fig.add_subplot(grid[2, :], sharex=ax_speed)
    a = results.attrs
    fig.suptitle(f"lat {a['lat']}°, H = {a['H']} m, {a['wind_type']} wind")

    # Hodograph: full run faint, the part so far bright, and the wind arrow from zero velocity
    half = _square_limits(ax_hodo, np.concatenate([s, s_steady, [0]]))
    ax_hodo.plot(s.real, s.imag, color="0.85", lw=1)
    ax_hodo.plot(0, 0, "+", color="0.5", ms=10)
    hodo_trail, = ax_hodo.plot([], [], color="C0", lw=1.5)
    hodo_now, = ax_hodo.plot([], [], "o", color="C0", ms=8, label="current")
    steady_now, = ax_hodo.plot([], [], "kx", ms=9, label="steady state for this wind")
    tau_max = max(np.abs(tau).max(), 1e-12)
    arrow_scale = 0.6 * half / tau_max  # longest wind arrow is 60% of the axis half-width
    work_max = max(np.abs(wind_work).max(), 1e-12)
    arrow = ax_hodo.quiver(
        [0], [0], [0], [0], [0], cmap="RdBu_r", norm=Normalize(-work_max, work_max),
        angles="xy", scale_units="xy", scale=1, width=0.012, zorder=3, edgecolor="k", linewidth=0.8,
    )
    fig.colorbar(arrow, ax=ax_hodo, shrink=0.7, label="wind work τ·u [W/m²]")
    ax_hodo.set(xlabel="u [cm/s]", ylabel="v [cm/s]", title="Hodograph (arrow: wind direction)")
    ax_hodo.legend(loc="upper right", fontsize=9)

    # Parcel path
    _square_limits(ax_path, np.concatenate([position, [0]]))
    ax_path.plot(position.real, position.imag, color="0.85", lw=1)
    path_trail, = ax_path.plot([], [], color="C1", lw=1.5)
    path_now, = ax_path.plot([], [], "o", color="C1", ms=8)
    ax_path.set(xlabel="x [km]", ylabel="y [km]", title="Parcel path")

    # Time series
    ax_speed.plot(days, speed, color="0.85", lw=1)
    speed_trail, = ax_speed.plot([], [], color="C0", lw=1.5)
    speed_now, = ax_speed.plot([], [], "o", color="C0", ms=6)
    ax_speed.set(yscale="log", ylabel="current speed [cm/s]")
    ax_speed.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax_speed.yaxis.set_minor_formatter(FormatStrFormatter("%g"))
    ax_speed.tick_params(labelbottom=False)

    ax_wind.plot(days, np.abs(tau), color="0.85", lw=1)
    wind_trail, = ax_wind.plot([], [], color="C2", lw=1.5)
    wind_now, = ax_wind.plot([], [], "o", color="C2", ms=6)
    ax_wind.set(xlabel="time [days]", ylabel="|τ| [N/m²]", xlim=(days[0], days[-1]), ylim=(0, 1.1 * tau_max))
    clock = ax_speed.text(0.99, 0.92, "", transform=ax_speed.transAxes, ha="right", va="top")

    def draw(n):
        hodo_trail.set_data(s.real[: n + 1], s.imag[: n + 1])
        hodo_now.set_data([s.real[n]], [s.imag[n]])
        steady_now.set_data([s_steady.real[n]], [s_steady.imag[n]])
        arrow.set_UVC([tau.real[n] * arrow_scale], [tau.imag[n] * arrow_scale], [wind_work[n]])
        path_trail.set_data(position.real[: n + 1], position.imag[: n + 1])
        path_now.set_data([position.real[n]], [position.imag[n]])
        speed_trail.set_data(days[: n + 1], speed[: n + 1])
        speed_now.set_data([days[n]], [speed[n]])
        wind_trail.set_data(days[: n + 1], np.abs(tau[: n + 1]))
        wind_now.set_data([days[n]], [np.abs(tau[n])])
        clock.set_text(f"day {days[n]:.2f}")

    path = Path(path)
    writer = "pillow" if path.suffix.lower() == ".gif" else "ffmpeg"
    if not writers.is_available(writer):
        plt.close(fig)
        raise RuntimeError(f"{writer} is not installed; install it (brew install ffmpeg) or save a .gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, draw, frames=frames)
    animation.save(path, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return path
