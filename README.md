# Finansal Verilerle Yapay Zekâyı Buluşturmak — PyTorch Atölyesi

[![Open Starter In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omarcevi/pytorch-finance-workshop/blob/main/notebooks/workshop_starter.ipynb)
[![Adım Adım Rehber](https://img.shields.io/badge/Rehber-omarcevi.dev-2563eb)](https://omarcevi.dev/workshops/pytorch-finance-workshop/)

**Işık IEEE Computer Society** için hazırlanan **2 saat 15 dakikalık** uygulamalı PyTorch atölyesi.
- 📍 **Yer:** Işık Üniversitesi Şile Kampüsü, DMF-214
- 🗓️ **Tarih ve Saat:** 18 Kasım 2026 Çarşamba, 16:30–18:45
- 📘 **Adım Adım Türkçe Rehber:** [omarcevi.dev/workshops/pytorch-finance-workshop](https://omarcevi.dev/workshops/pytorch-finance-workshop/)

> ⚠️ **Yasal Uyarı (Yatırım Tavsiyesi Değildir):** Bu depodaki tüm defterler, kodlar ve analizler yalnızca **makine öğrenmesi ve PyTorch eğitimi** amaçlıdır. Hiçbir içerik yatırım tavsiyesi niteliği taşımaz ve buradaki modeller gerçek alım-satım aracı olarak **kullanılmamalıdır**.

---

## Hızlı Başlangıç (Google Colab — Kurulum Gerekmez)

| Defter | Açıklama | Bağlantı |
|---|---|---|
| **`workshop_starter.ipynb`** | **Başlangıç Defteri (Katılımcılar için):** Atölye boyunca birlikte dolduracağımız `TODO` hücrelerini ve her bölüm başındaki **"🔄 Yetiştir"** kurtarma hücrelerini içerir. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omarcevi/pytorch-finance-workshop/blob/main/notebooks/workshop_starter.ipynb) |
| **`workshop_solution.ipynb`** | **Çözüm Defteri (Referans):** Tüm çözüm kodları ve ölçülmüş çıktılarıyla tam sürüm. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omarcevi/pytorch-finance-workshop/blob/main/notebooks/workshop_solution.ipynb) |

---

## Atölye Hikâyesi ve Zaman Çizelgesi

| # | Saat | Bölüm | Öğrenilenler ve Uygulama |
|---|---|---|---|
| **0** | 16:30–16:40 | **Giriş (Hook)** | Bölüm 4'teki "kusursuz" fiyat tahmini grafiğine ilk bakış: *"Bu grafikle para kazanabilir miyiz?"* |
| **1** | 16:40–16:55 | **Kurulum ve Veri** | `THYAO.IS` verisini `yfinance` (veya yedek snapshot) ile indirme, OHLCV ve düzeltilmiş fiyatların görselleştirilmesi. |
| **2** | 16:55–17:15 | **Temizlik ve Özellikler** | Eksik günler (`ffill`), logaritmik getiri, hareketli ortalama, volatilite, RSI, kronolojik bölme (%70/%15/%15) ve yalnızca eğitim setinde fit edilen `StandardScaler`. |
| **3** | 17:15–17:35 | **PyTorch Temelleri** | `Tensor`, 30 günlük `SlidingWindowDataset`, `DataLoader`, `nn.LSTM` mimarisi, her adımda tensör boyutları (`batch, seq_len, features`) ve 5 adımlı eğitim döngüsü. |
| **—** | 17:35–17:45 | **Ara (Mola)** | 10 dakika mola. |
| **4** | 17:45–18:05 | **Fiyat Tahmini ve Sürpriz** | Ertesi gün fiyat tahmini, *"Yarın = Bugün"* saf referans modeliyle (naive baseline) MAE/RMSE karşılaştırması ve yakınlaştırılmış grafikte 1 günlük gecikme (`Lag 1`) analizi. |
| **5** | 18:05–18:25 | **Doğru Kurulum** | Getiri tabanlı özelliklerle ertesi gün yön sınıflandırması (`BCEWithLogitsLoss`), çoğunluk sınıfı referansı, karmaşıklık matrisi ve 4 katlı Walk-Forward doğrulama. |
| **6** | 18:25–18:40 | **Meydan Okumalar** | Kendi sembolünü deneme, hacim özelliği ekleme, `GRU` ve `TinyTransformer` modelleri, işlem maliyetli (`15 bps`) backtest. |
| **7** | 18:40–18:45 | **Kapanış** | Eve götürülecek 3 temel ders, gerçek quant ekiplerinin farkı ve kaynaklar. |

---

## Depo Yapısı

```text
notebooks/
  workshop_solution.ipynb      # Tek doğruluk kaynağı (Single source of truth)
  workshop_starter.ipynb       # make_starter.py ile otomatik üretilir (elle düzenlenmez)
src/
  workshop_utils.py            # Yedekli veri yükleme, özellikler, modeller, grafikler ve Yetiştir fonksiyonları
data/
  README.md                    # Yedek veri (snapshot) açıklaması (ham CSV'ler git dışındadır)
scripts/
  make_starter.py              # Çözüm bloklarını (# BEGIN/END SOLUTION) TODO hücrelerine dönüştürür
  snapshot_data.py             # Varsayılan 5 sembol için yedek CSV dosyalarını yeniler
  build_notebooks.py           # Çözüm ve başlangıç defterlerini oluşturur
  generate_assets.py           # Ön-üretilmiş görselleri (hook_preview.png ve SVG diyagramları) üretir
guide/
  pytorch-finance-workshop.md  # omarcevi.dev/workshops/pytorch-finance-workshop rehber kaynağı
  images/                      # Rehber SVG diyagramları
docs/
  VERIFIED.md                  # Doğrulanmış yfinance ve Colab paket davranışları
  INSTRUCTOR.md                # Dakika dakika eğitmen akış çizelgesi, konuşma metinleri ve SSS
  DRY_RUN.md                   # Canlı Colab provası süreleri ve 5 sembol × 3 tohum ölçüm tablosu
  SLIDES_OUTLINE.md            # Açılış ve kapanış için 9 slaytlık sunum taslağı
tests/
  test_unit.py                 # Birim testleri (ağsız fallback, sızıntısızlık, Yetiştir hücreleri)
  test_notebooks.py            # Uçtan uca defter yürütme, süre (< 5 dk) ve metrik testleri
```

---

## Yerel Çalıştırma ve Testler

```bash
pip install -r requirements.txt
python3 scripts/snapshot_data.py              # Yedek verileri data/ klasörüne indirir
python3 scripts/make_starter.py               # Başlangıç defterini günceller
ruff check src/ scripts/ tests/               # Kod kalitesi kontrolü
WORKSHOP_FORCE_FALLBACK=1 pytest -v           # Tüm birim ve defter yürütme testlerini çalıştırır
```
