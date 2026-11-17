"""Birim testleri (ağ bağlantısı gerektirmez):
- load_prices hata, boş tablo ve rate-limit (YFRateLimitError) yedekleme (fallback) davranışları
- SlidingWindowDataset ve PriceWindowDataset boyut ve hedef sızıntısı (leakage) kontrolü
- Kronolojik bölme ve StandardScaler'ın yalnızca train verisinde fit edilmesi
- make_starter.py çözüm bloğu temizleme kontrolü
- Her 'Yetiştir' (catch-up) hücresinin sıfır/temiz çekirdekte (fresh kernel) tek başına çalışması
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_starter import convert_notebook, strip_solution_blocks

from workshop_utils import (
    FEATURE_COLS,
    PriceWindowDataset,
    SlidingWindowDataset,
    build_features,
    chronological_split,
    fit_scaler_on_train,
    load_prices,
)

# --- 1. load_prices Fallback Testleri (Ağsız / Mock) -------------------------------------


def test_load_prices_fallback_on_exception(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """yfinance genel bir istisna fırlattığında yedek (snapshot) veriye düşmelidir."""
    monkeypatch.delenv("WORKSHOP_FORCE_FALLBACK", raising=False)
    with patch("workshop_utils.yf.download", side_effect=RuntimeError("Ağ bağlantı hatası")):
        df = load_prices("THYAO.IS", force_fallback=False)
    captured = capsys.readouterr().out
    assert "Yedek veri (snapshot) yükleniyor" in captured
    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_prices_fallback_on_empty_dataframe(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """yfinance boş DataFrame döndürdüğünde yedek veriye düşmelidir."""
    monkeypatch.delenv("WORKSHOP_FORCE_FALLBACK", raising=False)
    with patch("workshop_utils.yf.download", return_value=pd.DataFrame()):
        df = load_prices("ASELS.IS", force_fallback=False)
    captured = capsys.readouterr().out
    assert "Yedek veri (snapshot) yükleniyor" in captured
    assert len(df) > 1000


def test_load_prices_fallback_on_rate_limit_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """yfinance YFRateLimitError fırlattığında veya shared._ERRORS kaydı bıraktığında yedek veriye düşmelidir."""
    monkeypatch.delenv("WORKSHOP_FORCE_FALLBACK", raising=False)
    rate_exc = yf.exceptions.YFRateLimitError()
    with patch("workshop_utils.yf.download", side_effect=rate_exc):
        df = load_prices("XU100.IS", force_fallback=False)
    captured = capsys.readouterr().out
    assert "YFRateLimitError" in captured
    assert len(df) > 1000

    # Ayrıca yf.download'un kendi içinde hatayı yutup shared._ERRORS'a yazdığı gerçek davranışı da test et
    def _fake_swallowed_rate_limit(*args, **kwargs):
        yf.shared._ERRORS["AAPL"] = "YFRateLimitError('Too Many Requests.')"
        return pd.DataFrame()

    with patch("workshop_utils.yf.download", side_effect=_fake_swallowed_rate_limit):
        df_aapl = load_prices("AAPL", force_fallback=False)
    assert len(df_aapl) > 1000


def test_load_prices_invalid_ticker_and_short_history() -> None:
    """Geçersiz sembol ve çok kısa geçmiş için anlaşılır Türkçe ValueError fırlatılmalıdır."""
    with pytest.raises(ValueError, match="sembolü için"):
        load_prices("OLMAYAN_SEMBOL_999", force_fallback=True)

    with pytest.raises(ValueError, match="yeterli geçmiş veri yok"):
        load_prices("THYAO.IS", start="2026-10-01", end="2026-11-15", force_fallback=True, min_rows=120)


# --- 2. SlidingWindowDataset Boyut ve Hedef Sızıntısı Testleri ---------------------------


def test_sliding_window_shapes_and_no_target_leakage() -> None:
    """SlidingWindowDataset doğru boyutları üretmeli ve hedef gün giriş penceresine sızmamalıdır."""
    # Her günün değeri kendi zaman indeksi (0, 1, 2, ..., 99) olsun
    n, f, w = 100, 4, 30
    timeline = np.arange(n, dtype=np.float32)
    features = np.stack([timeline + k * 1000 for k in range(f)], axis=1)
    targets = timeline.copy()

    ds = SlidingWindowDataset(features, targets, window_size=w)
    assert len(ds) == n - w

    for idx in range(len(ds)):
        x_win, y_target = ds[idx]
        assert x_win.shape == (w, f)
        assert y_target.shape == ()
        # Giriş penceresindeki ilk özellik sütunu [idx, idx+1, ..., idx+w-1] günlerini içerir
        window_days = x_win[:, 0].numpy()
        target_day = float(y_target.item())
        assert np.allclose(window_days, np.arange(idx, idx + w, dtype=np.float32))
        assert target_day == float(idx + w)
        # Hedef gün kesinlikle penceredeki tüm günlerden büyük olmalıdır (gelecek sızıntısı yok)
        assert target_day > float(window_days.max())
        assert target_day not in set(window_days.tolist())


def test_price_window_dataset_no_leakage() -> None:
    """PriceWindowDataset hedef fiyatı pencere dışındaki (idx + W) günden almalıdır."""
    prices = np.arange(10.0, 110.0, dtype=np.float32)
    w = 30
    ds = PriceWindowDataset(prices, window_size=w)
    assert len(ds) == len(prices) - w

    x_rel, y_rel, base_p, last_p = ds[5]
    assert x_rel.shape == (w, 1)
    assert np.isclose(float(base_p), prices[5])
    assert np.isclose(float(last_p), prices[5 + w - 1])
    reconstructed_target = (float(y_rel) + 1.0) * float(base_p)
    assert np.isclose(reconstructed_target, prices[5 + w], atol=1e-4)


# --- 3. Kronolojik Bölme ve Scaler Sızıntısızlık Testleri -------------------------------


def test_chronological_split_and_train_only_scaler() -> None:
    """Veri bölme kesin kronolojik olmalı ve StandardScaler yalnızca train üzerinde fit edilmelidir."""
    df = load_prices("THYAO.IS", force_fallback=True)
    feat_df = build_features(df)
    train_df, val_df, test_df = chronological_split(feat_df, train_ratio=0.70, val_ratio=0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(feat_df)
    assert train_df.index.is_monotonic_increasing
    assert val_df.index.is_monotonic_increasing
    assert test_df.index.is_monotonic_increasing
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

    # Test setine kasıtlı olarak aşırı büyük bir değer ekleyelim; scaler yalnızca train'e fit edildiyse
    # scaler.mean_ ve scaler.scale_ bundan KESİNLİKLE etkilenmemelidir!
    test_tampered = test_df.copy()
    test_tampered[FEATURE_COLS] = test_tampered[FEATURE_COLS] + 99999.0

    scaler, x_train, x_val, _ = fit_scaler_on_train(train_df, val_df, test_tampered, FEATURE_COLS)
    expected_mean = train_df[FEATURE_COLS].values.astype(np.float32).mean(axis=0)
    expected_std = train_df[FEATURE_COLS].values.astype(np.float32).std(axis=0, ddof=0)

    assert np.allclose(scaler.mean_, expected_mean, atol=1e-5)
    assert np.allclose(scaler.scale_, expected_std, atol=1e-5)
    assert np.allclose(x_train.mean(axis=0), 0.0, atol=1e-5)
    assert not np.allclose(x_val.mean(axis=0), 0.0, atol=1e-3)


# --- 4. make_starter.py Çözüm Bloğu Temizleme Testleri ----------------------------------


def test_make_starter_strips_all_solution_blocks() -> None:
    """make_starter.py tüm # BEGIN SOLUTION bloklarını kaldırmalı ve çözüm kodu bırakmamalıdır."""
    sample_code = (
        "x = 1\n"
        "if x > 0:\n"
        "    # BEGIN SOLUTION: y değerini hesaplayın\n"
        "    secret_solution_var = x + 41\n"
        "    # END SOLUTION\n"
        "print(x)"
    )
    stripped, count = strip_solution_blocks(sample_code)
    assert count == 1
    assert "secret_solution_var" not in stripped
    assert "# BEGIN SOLUTION" not in stripped
    assert "# END SOLUTION" not in stripped
    assert "    # TODO: y değerini hesaplayın" in stripped
    assert '    raise NotImplementedError("TODO: y değerini hesaplayın")' in stripped

    sol_path = ROOT / "notebooks" / "workshop_solution.ipynb"
    sta_path = ROOT / "notebooks" / "workshop_starter.ipynb"
    sol_nb = json.loads(sol_path.read_text(encoding="utf-8"))
    sta_nb = json.loads(sta_path.read_text(encoding="utf-8"))

    converted_nb, total_blocks = convert_notebook(sol_nb)
    assert total_blocks >= 8
    assert converted_nb == sta_nb

    sta_raw = sta_path.read_text(encoding="utf-8")
    assert "# BEGIN SOLUTION" not in sta_raw
    assert "# END SOLUTION" not in sta_raw

    # Çözüm defterindeki her BEGIN SOLUTION bloğunun starter defterinde bir TODO hücresine karşılık geldiğini doğrula
    sol_todo_cells = [
        c["id"] for c in sol_nb["cells"] if c["cell_type"] == "code" and "# BEGIN SOLUTION" in "".join(c["source"])
    ]
    sta_todo_cells = [
        c["id"] for c in sta_nb["cells"] if c["cell_type"] == "code" and "NotImplementedError(\"TODO:" in "".join(c["source"])
    ]
    assert sol_todo_cells == sta_todo_cells


# --- 5. 'Yetiştir' Hücrelerinin Sıfır Çekirdekte (Fresh Kernel) Çalışma Testi ------------


def test_every_yetistir_cell_works_in_fresh_process() -> None:
    """Defterdeki her 'Yetiştir' (catch-up) hücresi sıfırdan başlayan temiz bir Python sürecinde hatasız çalışmalıdır."""
    sol_path = ROOT / "notebooks" / "workshop_solution.ipynb"
    sol_nb = json.loads(sol_path.read_text(encoding="utf-8"))
    catchup_cells = [
        c for c in sol_nb["cells"] if c["cell_type"] == "code" and "Yetiştir:" in "".join(c["source"])
    ]
    assert len(catchup_cells) == 6, f"6 Yetiştir hücresi beklendi, {len(catchup_cells)} bulundu."

    env = {**subprocess.os.environ, "WORKSHOP_FORCE_FALLBACK": "1"}
    for cell in catchup_cells:
        code = "".join(cell["source"])
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, f"{cell['id']} Yetiştir hücresi başarısız oldu:\n{proc.stderr}"
        assert "✅" in proc.stdout
