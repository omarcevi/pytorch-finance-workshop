# Sunum Taslağı (`SLIDES_OUTLINE.md`)

**Etkinlik:** Finansal Verilerle Yapay Zekâyı Buluşturmak (PyTorch Atölyesi)  
**Topluluk:** Işık IEEE Computer Society · Işık Üniversitesi Şile Kampüsü (DMF-214)  
**Tarih / Saat:** 18 Kasım 2026 Çarşamba, 16:30–18:45  

Bu taslak, uygulamalı Colab çalışmasının başındaki 10 dakikalık açılış (Slayt 1–5) ve sonundaki 5 dakikalık kapanış (Slayt 6–9) için hazırlanmış **9 slaytlık** sunum planıdır.

---

## Slayt 1 · Kapak ve Hızlı Başlangıç (Açılış — 16:30)
- **Başlık:** Finansal Verilerle Yapay Zekâyı Buluşturmak: PyTorch ile Zaman Serileri
- **Alt Başlık:** Işık IEEE Computer Society Uygulamalı Atölyesi · Omar Elcircevi
- **Maddeler:**
  - 💻 **Başlangıç Defteri (Colab):** `https://colab.research.google.com/github/omarcevi/pytorch-finance-workshop/blob/main/notebooks/workshop_starter.ipynb` *(Slaytta büyük QR kod + kısa link: `omarcevi.dev/w/pytorch-finans`)*
  - 📘 **Adım Adım Türkçe Rehber:** `https://omarcevi.dev/workshops/pytorch-finance-workshop/`
  - ⚠️ **Not:** Eğitim amaçlıdır; yatırım tavsiyesi değildir.
- **Görsel:** Colab rozeti ve başlangıç defterine giden büyük QR kod.

---

## Slayt 2 · Kanca (Hook): "Bu Grafikle Para Kazanabilir miyiz?" (16:34)
- **Başlık:** 15 Satır PyTorch Koduyla "Kusursuz" Fiyat Tahmini mi?
- **Maddeler:**
  - Mavi çizgi: `THYAO.IS` gerçek test dönemi kapanış fiyatları; kırmızı çizgi: LSTM modelimizin tahmini.
  - İnternetteki yüzlerce blog yazısı ve videoda bu grafiği "yapay zekâ borsayı çözdü" başlığıyla görürsünüz.
  - **Soru:** Yarın sabah bu modeli canlıya bağlasak para kazanır mıyız? Cevabı saat 17:50'de birlikte ortaya çıkaracağız.
- **Kullanılacak Defter Grafiği:** `docs/hook_preview.png` (`Bölüm 0 / Bölüm 4 — plot_price_deception` çıktısı).

---

## Slayt 3 · Bugünün Yol Haritası (16:37)
- **Başlık:** 2 Saat 15 Dakikada Neler İnşa Edeceğiz?
- **Maddeler:**
  - **16:40 – 17:15 | Veri ve Özellik Mühendisliği:** Borsa İstanbul (`THYAO.IS`) OHLCV verisi, logaritmik getiri, volatilite, RSI ve veri sızıntısız (leak-free) kronolojik bölme.
  - **17:15 – 17:35 | Sıfırdan PyTorch:** `Tensor`, `SlidingWindowDataset` (30 gün), `nn.LSTM` mimarisi ve 5 adımlı eğitim döngüsü (`training loop`).
  - **17:45 – 18:25 | Büyük Sürpriz ve Doğru Kurulum:** Fiyat tahmininin arkasındaki 1 günlük gecikme yanılsaması, saf referans modeli (naive baseline) ve yön sınıflandırması + Walk-Forward doğrulama.
- **Görsel:** Zaman çizelgesi akış şeması (`Bölüm 1 -> Bölüm 6` ve her bölüm başındaki **🔄 Yetiştir** güvencesi).

---

## Slayt 4 · Altın Kural: Zaman Serilerinde Geleceği Sızdırmamak (16:55 Geçişi)
- **Başlık:** Veri Sızıntısı (Data Leakage) Nerede Saklanır?
- **Maddeler:**
  - **Kural 1:** Rastgele `shuffle=True` ile train/test bölünmez; geçmişte eğitilir, gelecekte test edilir (%70 Train / %15 Val / %15 Test).
  - **Kural 2:** `StandardScaler.fit()` yalnızca **Eğitim (Train)** bölgesinde çağrılır; tüm veriye `fit` etmek gelecekteki ortalama ve oynaklığı geçmişe sızdırır.
  - **Kural 3:** Eksik günler yalnızca geçmiş değerle (`ffill`) doldurulur; asla gelecekten geriye (`bfill`) doldurulmaz.
- **Kullanılacak Defter Grafiği:** `guide/images/chronological_split.svg` (`Bölüm 2 — plot_chronological_split` çıktısı).

---

## Slayt 5 · PyTorch LSTM Mimarisi ve Boyut Takibi (17:15 Geçişi)
- **Başlık:** 30 Günlük Pencereden Tek Tahmine: Tensör Boyutları
- **Maddeler:**
  - Girdi tensörü `bx`: **`(batch=64, seq_len=30, features=4)`** — 64 örnek, 30 iş günü, 4 özellik.
  - `nn.LSTM(4, 32, batch_first=True)` çıktısı: **`(64, 30, 32)`** — her gün için 32 boyutlu gizli durum.
  - Son günün özeti `out[:, -1, :]`: **`(64, 32)`** $\rightarrow$ `nn.Linear(32, 1)` $\rightarrow$ **`(64,)`** tahmin vektörü.
- **Kullanılacak Defter Grafiği:** `Bölüm 3 — plot_training_curves` (Eğitim ve doğrulama MSE kayıp eğrisi).

---

## Slayt 6 · Sürpriz Anı: Neden "Yarın = Bugün" Modeli LSTM'i Geçti? (17:55)
- **Başlık:** Göz Yanılsaması: Tahmin Değil, 1 Günlük Gecikme (Lag)!
- **Maddeler:**
  - `THYAO.IS` Test Seti: **LSTM MAE = 5.07 TL** iken, hiçbir şey öğrenmeyen **Saf Referans ($\hat{P}_{t+1} = P_t$) MAE = 4.55 TL**!
  - Son 35 güne yakınlaştığımızda kırmızı tahmin çizgisinin, mavi gerçek fiyat serisinin **1 gün sağa kaydırılmış kopyası** olduğu görülür (`Lag 1` korelasyonu = **0.991**).
  - Neden? Fiyatlar rastgele yürüyüşe ($P_{t+1} = P_t + \epsilon$) çok yakındır; MSE kaybını minimize etmenin en kolay yolu dünün fiyatını tekrar etmektir.
- **Kullanılacak Defter Grafiği:** `guide/images/zoom_and_lag.svg` (`Bölüm 4 — plot_zoom_and_lag` ikili grafiği).

---

## Slayt 7 · Doğru Kurulum: Yön Sınıflandırması ve %50 Gerçeği (18:15)
- **Başlık:** Fiyat Seviyesi Yerine Getiri Yönü (`BCEWithLogitsLoss`)
- **Maddeler:**
  - Durağan özellikler (`log_return`, hareketli ortalama, volatilite, RSI) ile yarının yönünü (`Yükseliş = 1 / Düşüş = 0`) sınıflandırıyoruz.
  - Tek test bölgesi doğruluğu **%54.74** (Çoğunluk sınıfı referansı: **%51.90**); 4 katlı **Walk-Forward** ortalaması **~%51.5**.
  - Dürüst bir kurulumda günlük yön tahmini neden %50–%55 bandındadır? **Etkin piyasalar**, **yüksek gürültü** ve **değişen piyasa rejimleri**.
- **Kullanılacak Defter Grafiği:** `guide/images/direction_results.svg` (`Bölüm 5 — plot_direction_results`: Karmaşıklık Matrisi ve Walk-Forward çubuk grafiği).

---

## Slayt 8 · Küçük Bir Üstünlük Neden Yetmez? İşlem Maliyetleri (18:35)
- **Başlık:** Komisyon ve Alım-Satım Makası (Slippage) Gerçeği
- **Maddeler:**
  - Model %54 doğrulukla yön bilse bile her pozisyon değişiminde ödenen **15 baz puan (%0.15)** işlem maliyeti brüt getiriyi hızla eritir.
  - Gerçek kantitatif (quant) ekipler tek bir hissede günlük fiyat tahmini yapmaz: yüzlerce varlığı birlikte sıralar, alternatif veri kullanır ve işlem maliyetini doğrudan optimizasyona katar.
- **Kullanılacak Defter Grafiği:** `Bölüm 6 Bonus 4 — İşlem Maliyetli Backtest` grafiği (Al-ve-Tut vs. Brüt Strateji vs. 15 bps Net Strateji).

---

## Slayt 9 · Eve Götürülecek 3 Ders ve Kapanış (18:40)
- **Başlık:** Özet: İyi Bir Veri Bilimci Olmanın 3 Kuralı
- **Maddeler:**
  1. **Her zaman saf bir referansla (baseline) yarışın:** "Yarın = Bugün" veya çoğunluk sınıfını geçemeyen model hiçbir şey öğrenmemiştir.
  2. **Zaman sırasına saygı duyun:** Kronolojik bölme ve Walk-Forward doğrulama olmadan elde edilen yüksek skorlar yanıltıcıdır.
  3. **Sızıntıyı (leakage) avlayın:** Gerçek olamayacak kadar iyi bir grafik (%90 doğruluk veya üst üste binen fiyat çizgisi) görürseniz, önce veri sızıntısını ve 1 günlük gecikmeyi kontrol edin.
  - 🔗 **Kodlar ve Rehber:** `github.com/omarcevi/pytorch-finance-workshop` · `omarcevi.dev/workshops/pytorch-finance-workshop`
  - ⚠️ *Bu atölyedeki hiçbir içerik yatırım tavsiyesi değildir.*
