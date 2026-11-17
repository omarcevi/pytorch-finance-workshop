"""Generates the pre-made workshop images: docs/hook_preview.png (Part 0 of the notebook) and guide/images/*.svg."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from workshop_utils import (
    catch_up_part_4,
    catch_up_part_5,
    plot_chronological_split,
    plot_direction_results,
    plot_ohlcv,
    plot_price_deception,
    plot_zoom_and_lag,
)


def _ensure_svg_dims(svg_path: Path, width: int = 960, height: int = 420) -> None:
    """Writes pixel width/height into the SVG header so the guide build script can read them."""
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r'<svg\b[^>]*>', lambda m: re.sub(r'width="[^"]+"', f'width="{width}"', re.sub(r'height="[^"]+"', f'height="{height}"', m.group(0))), text, count=1)
    svg_path.write_text(text, encoding="utf-8")


def _save_svg(fig: plt.Figure, path: Path, width: int, height: int) -> None:
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    _ensure_svg_dims(path, width, height)


def main() -> None:
    docs_dir = ROOT / "docs"
    img_dir = ROOT / "guide" / "images"
    docs_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    s4 = catch_up_part_4("THYAO.IS", window_size=30, epochs=15, seed=42)
    pr = s4["price_res"]

    # 1. Hook / Part 4 'perfect' prediction image (PNG for the notebook, SVG for the guide)
    fig_hook = plot_price_deception(s4["test_dates"], pr["actual_prices"], pr["predicted_prices"], "THYAO.IS")
    fig_hook.savefig(docs_dir / "hook_preview.png", dpi=150, bbox_inches="tight")
    _save_svg(fig_hook, img_dir / "hook_preview.svg", 960, 400)

    # 2. Part 1 OHLCV image
    _save_svg(plot_ohlcv(s4["raw_df"], "THYAO.IS"), img_dir / "ohlcv.svg", 960, 520)

    # 3. Part 2 chronological split image
    fig_split = plot_chronological_split(s4["train_df"], s4["val_df"], s4["test_df"], "THYAO.IS")
    _save_svg(fig_split, img_dir / "chronological_split.svg", 960, 400)

    # 4. Part 4 zoom and lag correlation image
    fig_zoom = plot_zoom_and_lag(
        s4["test_dates"],
        pr["actual_prices"],
        pr["predicted_prices"],
        pr["naive_prices"],
        "THYAO.IS",
        zoom_days=35,
    )
    _save_svg(fig_zoom, img_dir / "zoom_and_lag.svg", 1080, 410)

    # 5. Part 5 classification and walk-forward image
    s5 = catch_up_part_5("THYAO.IS", window_size=30, epochs=15, seed=42)
    fig_dir = plot_direction_results(s5["cls_res"]["confusion_matrix"], s5["wf_res"], "THYAO.IS")
    _save_svg(fig_dir, img_dir / "direction_results.svg", 1040, 400)

    print("Images generated:", docs_dir / "hook_preview.png", "and", img_dir)


if __name__ == "__main__":
    main()
