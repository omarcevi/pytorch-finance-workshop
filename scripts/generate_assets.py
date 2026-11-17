"""Atölye için ön-üretilmiş görselleri (docs/hook_preview.png ve guide/images/*.svg) üretir."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from workshop_utils import (
    plot_chronological_split,
    plot_direction_results,
    plot_ohlcv,
    plot_price_deception,
    plot_zoom_and_lag,
    yetistir_bolum_4,
    yetistir_bolum_5,
)


def _ensure_svg_dims(svg_path: Path, width: int = 960, height: int = 420) -> None:
    """build_match_day_guide.py benzeri betiklerin okuyabilmesi için SVG başlığına piksel width/height ekler."""
    text = svg_path.read_text(encoding="utf-8")
    import re

    text = re.sub(r'<svg\b[^>]*>', lambda m: re.sub(r'width="[^"]+"', f'width="{width}"', re.sub(r'height="[^"]+"', f'height="{height}"', m.group(0))), text, count=1)
    svg_path.write_text(text, encoding="utf-8")


def main() -> None:
    docs_dir = ROOT / "docs"
    img_dir = ROOT / "guide" / "images"
    docs_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    s4 = yetistir_bolum_4("THYAO.IS", window_size=30, epochs=15, seed=42)
    pr = s4["price_res"]

    # 1. Hook / Bölüm 4 'kusursuz' tahmin görseli
    fig_hook = plot_price_deception(s4["test_dates"], pr["actual_prices"], pr["predicted_prices"], "THYAO.IS")
    fig_hook.savefig(docs_dir / "hook_preview.png", dpi=150, bbox_inches="tight")
    fig_hook.savefig(img_dir / "hook_preview.svg", format="svg", bbox_inches="tight")
    plt.close(fig_hook)
    _ensure_svg_dims(img_dir / "hook_preview.svg", 960, 400)

    # 2. Bölüm 1 OHLCV görseli
    fig_ohlcv = plot_ohlcv(s4["raw_df"], "THYAO.IS")
    fig_ohlcv.savefig(img_dir / "ohlcv.svg", format="svg", bbox_inches="tight")
    plt.close(fig_ohlcv)
    _ensure_svg_dims(img_dir / "ohlcv.svg", 960, 520)

    # 3. Bölüm 2 kronolojik bölme görseli
    fig_split = plot_chronological_split(s4["train_df"], s4["val_df"], s4["test_df"], "THYAO.IS")
    fig_split.savefig(img_dir / "chronological_split.svg", format="svg", bbox_inches="tight")
    plt.close(fig_split)
    _ensure_svg_dims(img_dir / "chronological_split.svg", 960, 400)

    # 4. Bölüm 4 yakınlaştırma ve gecikme korelasyonu görseli
    fig_zoom = plot_zoom_and_lag(
        s4["test_dates"],
        pr["actual_prices"],
        pr["predicted_prices"],
        pr["naive_prices"],
        "THYAO.IS",
        zoom_days=35,
    )
    fig_zoom.savefig(img_dir / "zoom_and_lag.svg", format="svg", bbox_inches="tight")
    plt.close(fig_zoom)
    _ensure_svg_dims(img_dir / "zoom_and_lag.svg", 1080, 410)

    # 5. Bölüm 5 sınıflandırma ve walk-forward görseli
    s5 = yetistir_bolum_5("THYAO.IS", window_size=30, epochs=15, seed=42)
    fig_dir = plot_direction_results(s5["cls_res"]["confusion_matrix"], s5["wf_res"], "THYAO.IS")
    fig_dir.savefig(img_dir / "direction_results.svg", format="svg", bbox_inches="tight")
    plt.close(fig_dir)
    _ensure_svg_dims(img_dir / "direction_results.svg", 1040, 400)

    print("Görseller başarıyla üretildi:", docs_dir / "hook_preview.png", "ve", img_dir)


if __name__ == "__main__":
    main()
