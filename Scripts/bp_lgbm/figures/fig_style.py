"""
Shared publication-style constants for all journal figures.

Usage (in any figures/ notebook):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(_find_bp_lgbm()))  # see robust_sys_path() below
    from fig_style import DPI, W_FULL, W_SINGLE, ASPECT, FONT_FAMILY, apply_base_style, save_fig

Colors are intentionally NOT defined here — each figure family owns its palette.
"""
from pathlib import Path
import matplotlib.pyplot as plt


# ── Journal size constants (combination bitmapped line/halftone, 500 dpi) ─────
DPI      = 500
W_FULL   = 7.48   # inches — full page width  (3740 px @ 500 dpi)
W_SINGLE = 3.54   # inches — single column    (1772 px @ 500 dpi)
ASPECT   = 0.65   # height/width ratio (adjust per figure if needed)

FONT_FAMILY = "Arial"

# Minimum pixel widths required by the journal
MIN_PX_FULL   = 3740
MIN_PX_SINGLE = 1772


def apply_base_style(ax, grid_axis="y"):
    """
    Apply the shared spine / grid style to an Axes.

    Parameters
    ----------
    ax         : matplotlib Axes
    grid_axis  : "x", "y", or "both"
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.tick_params(labelsize=7)
    ax.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        linewidth=0.5,
        alpha=0.7,
        zorder=0,
    )
    ax.set_axisbelow(True)


def save_fig(fig, stem: str, out_dir: Path, dpi: int = DPI):
    """
    Save *fig* as both PNG and PDF into *out_dir*.

    Parameters
    ----------
    fig     : matplotlib Figure
    stem    : filename without extension (e.g. "Fig1_R2_OriginalVs90_full")
    out_dir : directory Path (must exist or be created before calling)
    dpi     : raster DPI (default: module-level DPI = 500)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def robust_sys_path() -> str:
    """
    Return the path to the bp_lgbm package directory, working whether the
    Jupyter kernel CWD is figures/ or bp_lgbm/ (or anywhere in between).
    """
    p = Path.cwd()
    for _ in range(4):
        if (p / "local_paths.py").exists():
            return str(p)
        p = p.parent
    raise FileNotFoundError(
        "Cannot locate bp_lgbm/ (no local_paths.py found within 4 parent levels of CWD)."
    )
