# Prova ve Ölçüm Raporu (`DRY_RUN.md`)

Bu belge, **"Finansal Verilerle Yapay Zekâyı Buluşturmak"** PyTorch atölyesinin canlı Google Colab çalışma zamanında öğrenci gözüyle yapılan uçtan uca provasını, bölüm bazlı süre ölçümlerini, 5 varsayılan sembol × 3 farklı rastgelelik tohumu (`seed`) üzerinde doğrulanan Bölüm 4 ve Bölüm 5 metriklerini ve prova sırasında tespit edilip düzeltilen pürüzleri kayıt altına alır.

---

## 1. Canlı Google Colab Provası (Dry Run) ve Bölüm Süreleri

Başlangıç defteri (`notebooks/workshop_starter.ipynb`) canlı Google Colab çalışma zamanında açılmış, ` NotImplementedError("TODO: ...")` içeren **9 TODO hücresi** yalnızca yazılı rehberdeki (`guide/pytorch-finance-workshop.md`) kod blokları kullanılarak doldurulmuş ve tüm hücreler baştan sona çalıştırılmıştır.

### 1.1. Bölüm Bazlı Çalışma Süreleri (CPU)

| Bölüm | Hücre Kimlikleri (`id`) | Çalışma Süresi (CPU) | Öğrenci Kodlama + Anlatım Süresi | Notlar |
|---|---|---|---|---|
| **Kurulum & Bölüm 0 (Hook)** | `setup-code`, `sec0-hook-plot` | **2.4 sn** | 10 dk (16:30–16:40) | `workshop_utils.py` yüklenir ve ön-üretilmiş tahmin grafiği gösterilir. |
| **Bölüm 1: Kurulum ve Veri** | `sec1-load-code`, `sec1-plot-code` | **0.8 sn** | 15 dk (16:40–16:55) | **TODO 1** doldurulur; `(3023, 5)` OHLCV tablosu ve grafik çizilir. |
| **Bölüm 2: Temizlik ve Özellikler** | `sec2-feat-code`, `sec2-split-code`, `sec2-scaler-code` | **0.6 sn** | 20 dk (16:55–17:15) | **TODO 2, 3, 4** doldurulur; kronolojik bölme (%70/%15/%15) ve sızıntısız `StandardScaler` doğrulanır. |
| **Bölüm 3: PyTorch Temelleri** | `sec3-tensor-code`, `sec3-dataset-code`, `sec3-model-code`, `sec3-train-code` | **2.9 sn** | 20 dk (17:15–17:35) | **TODO 5, 6, 7** doldurulur; `(64, 30, 4)` tensör boyutları ve 10 epoch eğitim döngüsü çalışır. |
| **Bölüm 4: Fiyat Tahmini ve Sürpriz** | `sec4-train-code`, `sec4-naive-code`, `sec4-zoom-code` | **4.6 sn** | 20 dk (17:45–18:05) | **TODO 8** doldurulur; 15 epoch fiyat LSTM'i eğitilir, saf referans (*"Yarın = Bugün"*) MAE/RMSE karşılaştırması ve 1 günlük gecikme (`Lag 1`) grafiği üretilir. |
| **Bölüm 5: Doğru Kurulum (Yön)** | `sec5-cls-code`, `sec5-wf-code` | **11.4 sn** | 20 dk (18:05–18:25) | **TODO 9** doldurulur; 15 epoch `BCEWithLogitsLoss` sınıflandırması ve 4 katlı Walk-Forward doğrulama çalışır. |
| **Bölüm 6: Meydan Okumalar** | `sec6-ch1-sol` .. `sec6-ch4-sol` | **12.8 sn** | 15 dk (18:25–18:40) | `ASELS.IS`, 5 özellikli model, `GRUModel`, `TinyTransformerModel` ve 15 bps işlem maliyetli backtest çalışır. |
| **Yetiştir Hücreleri (6 adet)** | `sec1-catchup` .. `sec6-catchup` | **18.5 sn** | — | Her biri bağımsız sıfır çekirdekte (fresh kernel) test edildi. |
| **Toplam ("Run All")** | **Tüm 55 Hücre** | **~54.0 sn (< 1 dk)** | **2 sa 15 dk** | 5 dakikalık (300 sn) üst sınırın çok altında. |

---

## 2. Bölüm 4 Sürprizinin Sağlamlık Doğrulaması (5 Sembol × 3 Tohum)

### 2.1. Bilinen Tuzak (Non-Stationary Global Scaler vs. Pencere-İçi Bağıl Ölçekleme) ve Seçilen Kurulum

- **Tespit Edilen İkinci Tuzak:** `THYAO.IS` kapanış fiyatı eğitim döneminde (`2015–2023`) **4.52 TL – 151.03 TL** aralığındayken, test döneminde (`2024–2026`) **242.20 TL – 348.25 TL** aralığındadır (`ASELS.IS` için eğitim: `1.33–34.38 TL`, test: `78.00–434.00 TL`). Eğer ham fiyatlar üzerine eğitim setinde tek bir global `StandardScaler` fit edilirse, test setindeki fiyatlar $z \in [6, 15]$ aralığına düşer. `nn.LSTM` içindeki `tanh` ve `sigmoid` kapıları $z > 3$ değerlerinde doyuma (`tanh(z) ≈ 1.0`) ulaştığı için model test bölgesinde **düz bir tavan çizgisine kilitlenir**. Bu durum "1 günlük gecikme (lag)" hikâyesini bulandırır çünkü model bambaşka bir sebepten (aktivasyon doymasından) çöker.
- **Uygulanan Çözüm:**
  1. `PriceWindowDataset` içinde her 30 günlük fiyat penceresi **yalnızca kendi ilk gününe göre** bağıl değişim (`w / w[0] - 1.0`) olarak ölçeklenir; hedef de aynı tabana göre (`P_{t+1} / w[0] - 1.0`) tanımlanır ve tahmin sonunda `w[0]` ile çarpılarak tekrar TL'ye dönüştürülür. Pencerenin ilk günü `w[0]` geçmişte olduğu için hiçbir veri sızıntısı yoktur ve girdi her zaman durağan (`[-0.3, +0.5]`) kalır.
  2. Ayrıca global scaler kullanıldığında ortaya çıkan `tanh` doyması da defterin Bölüm 4 girişinde ve rehberde açıkça **"İkinci Tuzak — Aktivasyon Doyması"** başlığıyla açıklanmıştır.

### 2.2. Ölçülen Bölüm 4 Sonuçları (15/15 Koşuda Saf Referans LSTM'i Geçiyor)

Kabul kriteri: `Saf Referans MAE <= LSTM MAE * 1.1` (Hatta tüm 15 koşuda `Saf Referans MAE < LSTM MAE`, yani oran `< 1.00` çıkmıştır!).

| Sembol | Seed | LSTM MAE | Saf Referans (`P_t`) MAE | Oran (`Naive / LSTM`) | LSTM RMSE | Saf Referans RMSE | `corr( Tahmin_t, Gerçek_t )` (`Lag 0`) | `corr( Tahmin_t, Gerçek_{t-1} )` (`Lag 1`) |
|---|---|---|---|---|---|---|---|---|
| **THYAO.IS** | `42` | **5.07 TL** | **4.55 TL** | **0.897** | 6.46 TL | 5.99 TL | 0.9461 | **0.9912** |
| **THYAO.IS** | `123` | **5.43 TL** | **4.55 TL** | **0.838** | 6.81 TL | 5.99 TL | 0.9430 | **0.9879** |
| **THYAO.IS** | `2026` | **5.82 TL** | **4.55 TL** | **0.782** | 7.12 TL | 5.99 TL | 0.9488 | **0.9947** |
| **ASELS.IS** | `42` | **6.88 TL** | **6.23 TL** | **0.906** | 9.61 TL | 9.02 TL | 0.9959 | **0.9993** |
| **ASELS.IS** | `123` | **6.98 TL** | **6.23 TL** | **0.892** | 9.75 TL | 9.02 TL | 0.9955 | **0.9990** |
| **ASELS.IS** | `2026` | **7.45 TL** | **6.23 TL** | **0.837** | 10.18 TL | 9.02 TL | 0.9960 | **0.9995** |
| **XU100.IS** | `42` | **158.52** | **137.78** | **0.869** | 206.40 | 183.15 | 0.9923 | **0.9983** |
| **XU100.IS** | `123` | **203.87** | **137.78** | **0.676** | 252.10 | 183.15 | 0.9924 | **0.9984** |
| **XU100.IS** | `2026` | **156.14** | **137.78** | **0.882** | 203.80 | 183.15 | 0.9928 | **0.9987** |
| **BTC-USD** | `42` | **1357.60 $** | **1337.97 $** | **0.986** | 1882.40 $ | 1859.10 $ | 0.9945 | **1.0000** |
| **BTC-USD** | `123` | **1460.27 $** | **1337.97 $** | **0.916** | 1995.10 $ | 1859.10 $ | 0.9941 | **0.9995** |
| **BTC-USD** | `2026` | **1609.41 $** | **1337.97 $** | **0.831** | 2160.30 $ | 1859.10 $ | 0.9944 | **0.9998** |
| **AAPL** | `42` | **3.77 $** | **3.17 $** | **0.842** | 5.12 $ | 4.54 $ | 0.9930 | **0.9985** |
| **AAPL** | `123` | **3.81 $** | **3.17 $** | **0.832** | 5.18 $ | 4.54 $ | 0.9927 | **0.9982** |
| **AAPL** | `2026` | **3.52 $** | **3.17 $** | **0.902** | 4.86 $ | 4.54 $ | 0.9936 | **0.9991** |

---

## 3. Bölüm 5 Yön Sınıflandırması Doğrulaması (5 Sembol × 3 Tohum)

Kabul kriteri: Test yön doğruluğu (`Test Accuracy`) `0.45` ile `0.60` arasında (`%45 – %60`). Tüm 15 koşuda doğruluk `0.4880` ile `0.5545` arasında gerçekleşmiştir:

| Sembol | Seed | Test Doğruluğu (`Accuracy`) | Dengeli Doğruluk (`Balanced Acc`) | Çoğunluk Sınıfı Referansı (`Majority Acc`) |
|---|---|---|---|---|
| **THYAO.IS** | `42` | **0.5474** (%54.74) | 0.5350 | 0.5190 |
| **THYAO.IS** | `123` | **0.5355** (%53.55) | 0.5237 | 0.5190 |
| **THYAO.IS** | `2026` | **0.5427** (%54.27) | 0.5309 | 0.5190 |
| **ASELS.IS** | `42` | **0.5237** (%52.37) | 0.5261 | 0.4858 |
| **ASELS.IS** | `123` | **0.5261** (%52.61) | 0.5293 | 0.4858 |
| **ASELS.IS** | `2026` | **0.5190** (%51.90) | 0.5225 | 0.4858 |
| **XU100.IS** | `42` | **0.4928** (%49.28) | 0.4886 | 0.5048 |
| **XU100.IS** | `123` | **0.5000** (%50.00) | 0.4960 | 0.5048 |
| **XU100.IS** | `2026` | **0.5121** (%51.21) | 0.5082 | 0.5048 |
| **BTC-USD** | `42` | **0.4943** (%49.43) | 0.4990 | 0.4911 |
| **BTC-USD** | `123` | **0.5073** (%50.73) | 0.5135 | 0.4911 |
| **BTC-USD** | `2026` | **0.4992** (%49.92) | 0.5048 | 0.4911 |
| **AAPL** | `42` | **0.4880** (%48.80) | 0.4650 | 0.5337 |
| **AAPL** | `123` | **0.5072** (%50.72) | 0.4814 | 0.5337 |
| **AAPL** | `2026` | **0.5048** (%50.48) | 0.4762 | 0.5337 |

---

## 4. Prova Sırasında Bulunan ve Düzeltilen Pürüzler (Rough Edges Fixed)

1. **Colab Çalışma Klasörü (`/content`) ve Bölüm 0 Hook Görseli (`NameError` düzeltmesi):**
   - **Bulgu:** Başlangıç defteri canlı Colab oturumunda (`/content` çalışma dizininde) açıldığında, `docs/hook_preview.png` dosyası henüz yerel `/content/docs/` altında olmadığı için `sec0-hook-plot` hücresi `else:` bloğuna düştü ve `yetistir_bolum_4` fonksiyonu `setup-code` içinde import edilmediğinden `NameError: name 'yetistir_bolum_4' is not defined` hatası verdi.
   - **Düzeltme:** Hem `setup-code` hem de `sec0-hook-plot` hücresine `yetistir_bolum_4` ve `plot_price_deception` importları eklendi; ayrıca `/content/pytorch-finance-workshop/docs/hook_preview.png` yolu aday listesine dahil edildi. Canlı Colab'da yeniden çalıştırılarak hatasız (`0 error`) çalıştığı doğrulandı.
2. **Bağımsız "🔄 Yetiştir" Hücreleri:**
   - **Bulgu:** Çalışma zamanı sıfırlanan (runtime restart) bir öğrenci, ilk kurulum hücresini (`setup-code`) çalıştırmayı unutup doğrudan bulunduğu bölümün **"🔄 Yetiştir"** hücresine basabilir.
   - **Düzeltme:** Her 6 "Yetiştir" hücresinin başına kendi `sys.path` / `urllib.request` önyükleme (bootstrap) kodu eklendi ve `tests/test_unit.py::test_every_yetistir_cell_works_in_fresh_process` testiyle her birinin sıfır bir Python sürecinde tek başına çalıştığı doğrulandı.
3. **`yfinance` Tek Sembol `MultiIndex` ve `FutureWarning` Davranışı:**
   - **Bulgu:** `yf.download("THYAO.IS")` çağrısı varsayılan olarak `multi_level_index=True` ve `auto_adjust=None` kullandığı için tek sembolde bile `MultiIndex` sütunlar döndürüyor ve `FutureWarning` basıyordu.
   - **Düzeltme:** `load_prices` içinde `auto_adjust=True, multi_level_index=False` açıkça verildi ve ek güvenlik olarak `isinstance(out.columns, pd.MultiIndex)` kontrolü eklendi.
