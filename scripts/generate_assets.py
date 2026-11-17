"""Atölye için ön-üretilmiş görselleri üretir.

- docs/hook_preview.png: Türkçe defterdeki Bölüm 0 görseli (Türkçe etiketler).
- guide/images/*.svg: İngilizce rehber (omarcevi.dev) görselleri; aynı grafikler, etiketler İngilizceye çevrilir.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

from workshop_utils import (
    plot_chronological_split,
    plot_direction_results,
    plot_ohlcv,
    plot_price_deception,
    plot_zoom_and_lag,
    yetistir_bolum_4,
    yetistir_bolum_5,
)

# Etiketin tamamı eşleşirse kullanılır (kısa kelimeler başka etiketlerin içinde de geçtiği için).
EN_EXACT: dict[str, str] = {
    "Fiyat": "Price",
    "Tarih": "Date",
    "Hacim": "Volume",
    "Kapanış Fiyatı": "Close price",
    "LSTM Tahmini": "LSTM prediction",
    "Doğruluk (Accuracy)": "Accuracy",
}

# Etiketin içinde geçen parçalar (başlıklar sembol adı gibi değişken kısımlar içerir).
EN_PARTS: dict[str, str] = {
    "Günlük Kapanış Fiyatı ve İşlem Hacmi": "Daily Close Price and Volume",
    "Düzeltilmiş Kapanış (Close)": "Adjusted close",
    "Kronolojik Veri Bölme (Gelecekten Geçmişe Sızıntı Yok)": "Chronological Split (No Leakage from the Future)",
    "Eğitim (Train - %70)": "Train (70%)",
    "Doğrulama (Val - %15)": "Validation (15%)",
    "Test (%15)": "Test (15%)",
    "Test Seti — Gerçek Fiyat vs. LSTM Tahmini ('Kusursuz' Görünüm)": "Test Set — Actual Price vs. LSTM Prediction (the 'Perfect' Look)",
    "Gerçek Fiyat (Actual)": "Actual price",
    "LSTM Tahmini (Predicted)": "LSTM prediction",
    "Gerçek Fiyat (P_t)": "Actual price (P_t)",
    "Saf Referans (Yarın = Bugün)": "Naive baseline (tomorrow = today)",
    "Güne Yakınlaştırma (1 Günlük Gecikme!)": "Days (a One-Day Lag!)",
    "— Son ": "— Zoom on the Last ",
    "Tahmin ile Gerçek Seri Korelasyonu\n(Lag 1 = Dünün Fiyatıyla Korelasyon Zirvede!)": (
        "Correlation of Prediction and Actual Price\n(Peaks at Lag 1 = Yesterday's Price!)"
    ),
    "Pearson Korelasyonu (r)": "Pearson correlation (r)",
    "Tahmin: Düşüş (0)": "Predicted: down (0)",
    "Tahmin: Yükseliş (1)": "Predicted: up (1)",
    "Gerçek: Düşüş (0)": "Actual: down (0)",
    "Gerçek: Yükseliş (1)": "Actual: up (1)",
    "Karmaşıklık Matrisi (Test)": "Confusion Matrix (Test)",
    "LSTM Doğruluk (Accuracy)": "LSTM accuracy",
    "Çoğunluk Sınıfı Referansı": "Majority-class baseline",
    "%50 Rastgele Sınırı": "50% chance level",
    "Walk-Forward Doğrulama Sonuçları": "Walk-Forward Validation Results",
}


def _translate(text: str) -> str:
    if text in EN_EXACT:
        return EN_EXACT[text]
    for tr, en in sorted(EN_PARTS.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(tr, en)
    return text


def to_english(fig: plt.Figure) -> plt.Figure:
    """Şekildeki tüm başlık, eksen, lejant ve sabit eksen etiketlerini İngilizceye çevirir."""
    fig.canvas.draw()  # eksen etiketleri çizimde üretilir; önce çiz, sonra yalnızca değişenleri sabitle
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = [t.get_text() for t in axis.get_ticklabels()]
            translated = [_translate(s) for s in labels]
            if translated != labels:
                axis.set_ticks(axis.get_ticklocs(), labels=translated)
    for artist in fig.findobj(Text):
        artist.set_text(_translate(artist.get_text()))
    return fig


def _ensure_svg_dims(svg_path: Path, width: int = 960, height: int = 420) -> None:
    """Rehber derleme betiğinin okuyabilmesi için SVG başlığına piksel width/height ekler."""
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r'<svg\b[^>]*>', lambda m: re.sub(r'width="[^"]+"', f'width="{width}"', re.sub(r'height="[^"]+"', f'height="{height}"', m.group(0))), text, count=1)
    svg_path.write_text(text, encoding="utf-8")


def _save_guide_svg(fig: plt.Figure, path: Path, width: int, height: int) -> None:
    to_english(fig).savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    _ensure_svg_dims(path, width, height)


def main() -> None:
    docs_dir = ROOT / "docs"
    img_dir = ROOT / "guide" / "images"
    docs_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    s4 = yetistir_bolum_4("THYAO.IS", window_size=30, epochs=15, seed=42)
    pr = s4["price_res"]

    # 1. Hook / Bölüm 4 'kusursuz' tahmin görseli (defter için Türkçe PNG, rehber için İngilizce SVG)
    fig_hook = plot_price_deception(s4["test_dates"], pr["actual_prices"], pr["predicted_prices"], "THYAO.IS")
    fig_hook.savefig(docs_dir / "hook_preview.png", dpi=150, bbox_inches="tight")
    _save_guide_svg(fig_hook, img_dir / "hook_preview.svg", 960, 400)

    # 2. Bölüm 1 OHLCV görseli
    _save_guide_svg(plot_ohlcv(s4["raw_df"], "THYAO.IS"), img_dir / "ohlcv.svg", 960, 520)

    # 3. Bölüm 2 kronolojik bölme görseli
    fig_split = plot_chronological_split(s4["train_df"], s4["val_df"], s4["test_df"], "THYAO.IS")
    _save_guide_svg(fig_split, img_dir / "chronological_split.svg", 960, 400)

    # 4. Bölüm 4 yakınlaştırma ve gecikme korelasyonu görseli
    fig_zoom = plot_zoom_and_lag(
        s4["test_dates"],
        pr["actual_prices"],
        pr["predicted_prices"],
        pr["naive_prices"],
        "THYAO.IS",
        zoom_days=35,
    )
    _save_guide_svg(fig_zoom, img_dir / "zoom_and_lag.svg", 1080, 410)

    # 5. Bölüm 5 sınıflandırma ve walk-forward görseli
    s5 = yetistir_bolum_5("THYAO.IS", window_size=30, epochs=15, seed=42)
    fig_dir = plot_direction_results(s5["cls_res"]["confusion_matrix"], s5["wf_res"], "THYAO.IS")
    _save_guide_svg(fig_dir, img_dir / "direction_results.svg", 1040, 400)

    print("Görseller başarıyla üretildi:", docs_dir / "hook_preview.png", "ve", img_dir)


if __name__ == "__main__":
    main()
