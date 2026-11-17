# Eğitmen Kılavuzu (`INSTRUCTOR.md`)

**Etkinlik:** Finansal Verilerle Yapay Zekâyı Buluşturmak (PyTorch Atölyesi)  
**Topluluk:** Işık IEEE Computer Society  
**Yer ve Zaman:** Işık Üniversitesi Şile Kampüsü, DMF-214 · 18 Kasım 2026 Çarşamba, **16:30–18:45** (2 saat 15 dakika)  
**Hedef Kitle:** Temel Python ve `pandas` bilen, ancak daha önce sıfırdan PyTorch eğitim döngüsü (`training loop`) yazmamış başlangıç–orta seviye üniversite öğrencileri (~40 kişi, kendi dizüstü bilgisayarlarında Google Colab CPU).

> ⚠️ **Eğitmen Hatırlatması:** Açılışta, Bölüm 4 sürprizinde ve kapanışta bu çalışmanın **yatırım tavsiyesi olmadığını** ve buradaki modellerin gerçek alım-satım aracı olarak kullanılmaması gerektiğini açıkça vurgulayın.

---

## 1. Etkinlik Öncesi Kontrol Listesi (Pre-Event Checklist)

### Sabah (18 Kasım 2026, 09:00–12:00)
- [ ] **Colab üzerinden `yfinance` testi:** Temiz bir Google Colab CPU çalışma zamanı açıp `load_prices("THYAO.IS")` çağrısını hem canlı `yfinance` yoluyla hem de `force_fallback=True` ile test edin.
- [ ] **Yedek Veri (`SNAPSHOT_BASE_URL`) kontrolü:** `src/workshop_utils.py` içindeki `SNAPSHOT_BASE_URL` adresinin (`https://storage.googleapis.com/omarcevi-workshop-data/pytorch-finance-2026` veya katılımcılara özel paylaşılan bulut klasörünün) erişilebilir olduğunu doğrulayın; DMF-214'te 40 öğrenci aynı anda Yahoo Finance'e bağlandığında hız sınırına (`YFRateLimitError`) takılma olasılığı çok yüksektir.
- [ ] **Kısa bağlantı ve QR kod:** Başlangıç defteri (`workshop_starter.ipynb`) Colab bağlantısı ve rehber sayfası (`https://omarcevi.dev/workshops/pytorch-finance-workshop/`) için kısa bağlantı (örn. `omarcevi.dev/w/pytorch-finans` / `bit.ly/isik-pytorch-2026`) ve ilk slayta eklenecek büyük QR kodu doğrulayın.
- [ ] **Yedek sekme:** Tarayıcınızda ikinci bir sekmede **çalıştırılmış ve tüm çıktıları hazır** `workshop_solution.ipynb` defterini açık tutun.
- [ ] **Çevrimdışı yerel yedek (`requirements.txt`):** Google Colab veya kampüs internetinde genel bir kesinti olması ihtimaline karşı dizüstü bilgisayarınızda `pip install -r requirements.txt` ile kurulmuş yerel Jupyter ortamını ve `data/*.csv` dosyalarını hazır bulundurun.

### Salonda (16:00–16:25, DMF-214)
- [ ] Projeksiyon çözünürlüğünü kontrol edin; Colab yazı tipi boyutunu `%125–%135` yakınlaştırmaya alın ve açık temayı (Light Theme) seçin (projeksiyonda koyu tema okunaklılığını kaybeder).
- [ ] Tahtaya Wi-Fi bilgilerini, rehber adresini (`omarcevi.dev/workshops/pytorch-finance-workshop`) ve *"Geride kalırsanız bölüm başındaki 🔄 Yetiştir hücresine basın"* notunu yazın.

---

## 2. Dakika Dakika Akış Çizelgesi (16:30–18:45)

| Saat | Süre | Bölüm | Açık Olacak Hücre (`id`) | Eğitmen Ne Yapar? | Öğrenciler Ne Yapar? |
|---|---|---|---|---|---|
| **16:30–16:35** | 5 dk | **Açılış & Kurulum** | `intro-header`, `setup-code` | Kısa tanışma, yasal uyarı (yatırım tavsiyesi değildir), Colab bağlantısını açtırma ve ilk kurulum hücresini çalıştırma. | Starter defterini Colab'da açar, `setup-code` hücresini çalıştırır (`PyTorch 2.14.1 \| CPU`). |
| **16:35–16:40** | 5 dk | **Bölüm 0: Giriş (Hook)** | `sec0-md`, `sec0-hook-plot` | Ön-üretilmiş THYAO fiyat tahmin grafiğini ekrana yansıtır. *"Bu grafikle para kazanabilir miyiz?"* sorusunu sorar, el kaldırarak oylama yapar, cevabı Bölüm 4'e saklar. | Grafiği inceler ve tahmin yürütür. |
| **16:40–16:48** | 8 dk | **Bölüm 1: Veri İndirme** | `sec1-load-code` | `load_prices("THYAO.IS")` TODO'sunu canlı yazar. Yahoo hız sınırı çıkarsa yedek verinin nasıl devreye girdiğini gösterir. OHLCV ve temettü/bölünme düzeltmesini (`auto_adjust=True`) 3 cümlede anlatır. | **TODO 1:** `raw_df = load_prices(TICKER, ...)` satırını tamamlar ve son 5 satırı inceler. |
| **16:48–16:55** | 7 dk | **Bölüm 1: Görselleştirme** | `sec1-plot-code` | Kapanış fiyatı ve hacim grafiğini çizdirir; 2015'teki 5 TL ile 2026'daki 300+ TL arasındaki ölçek farkına dikkat çeker. | Grafiği çalıştırır ve inceler. |
| **16:55–17:03** | 8 dk | **Bölüm 2: Özellikler** | `sec2-feat-code` | Neden `bfill` yerine yalnızca `ffill` kullandığımızı ve neden ham fiyat yerine log getiri ($\ln(P_t/P_{t-1})$) hesapladığımızı açıklar. | **TODO 2:** `log_return`, `roll_mean_5` ve `roll_std_10` sütunlarını yazar. |
| **17:03–17:09** | 6 dk | **Bölüm 2: Kronolojik Bölme** | `sec2-split-code` | Zaman serilerinde rastgele karıştırmanın (`shuffle=True`) neden geleceği kopyalamak olduğunu anlatır; %70/%15/%15 grafiğini çizdirir. | **TODO 3:** `chronological_split` çağrısını tamamlar ve 3 bölgeli grafiği çizer. |
| **17:09–17:15** | 6 dk | **Bölüm 2: Scaler ve Sızıntı** | `sec2-scaler-code` | `StandardScaler.fit_transform` işleminin **yalnızca `train_df`** üzerinde yapılmasını gösterir; tüm veriye `fit` etmenin veri sızıntısı (data leakage) olduğunu vurgular. | **TODO 4:** `scaler.fit_transform(train)` ve `scaler.transform(val/test)` satırlarını yazar. |
| **17:15–17:22** | 7 dk | **Bölüm 3: Tensor & Dataset** | `sec3-tensor-code`, `sec3-dataset-code` | `SlidingWindowDataset` (30 gün) yapısını gösterir. Bölmenin kronolojik kaldığını ama `train_loader` içindeki pencereleri karıştırmanın (`shuffle=True`) neden güvenli olduğunu açıklar. | **TODO 5:** `SlidingWindowDataset` ve `DataLoader` nesnelerini oluşturur; `(64, 30, 4)` boyutunu ekranda görür. |
| **17:22–17:28** | 6 dk | **Bölüm 3: `nn.Module` & Boyutlar** | `sec3-model-code` | `SimpleLSTM` sınıfını yazar; `bx -> lstm_out -> last_step -> pred_out` boyut geçişlerini satır satır ekranda gösterir. | **TODO 6:** `forward` metodunu tamamlar ve 4 adımlı boyut çıktısını doğrular. |
| **17:28–17:35** | 7 dk | **Bölüm 3: Eğitim Döngüsü** | `sec3-train-code` | 5 adımlı PyTorch döngüsünü (`zero_grad`, `forward`, `loss`, `backward`, `step`) canlı yazar ve kayıp eğrisini çizdirir. | **TODO 7:** Eğitim döngüsünü tamamlar, 10 epoch çalıştırır (~3 sn). |
| **17:35–17:45** | 10 dk | **☕ Ara (Mola)** | `break-md` | Sorusu olan öğrencilerin yanına gider; geride kalanlara `sec4-catchup` (**🔄 Yetiştir: Bölüm 4**) hücresini gösterir. | Mola verir. |
| **17:45–17:52** | 7 dk | **Bölüm 4: Fiyat Tahmini** | `sec4-catchup`, `sec4-train-code` | Moladan dönüşte herkese `sec4-catchup` çalıştırtır. Fiyat tahmin modelini eğitir (`train_price_lstm`) ve uzaktan "kusursuz" görünen test grafiğini çizdirir. Ayrıca global scaler kullanılsaydı `tanh` doymasının (2. tuzak) ne yapacağını kısaca belirtir. | `sec4-train-code` hücresini çalıştırır, "kusursuz" grafiği kendi ekranında görür. |
| **17:52–17:58** | 6 dk | **Bölüm 4: Sürpriz 1 (Saf Referans)** | `sec4-naive-code` | **Kilit Öğretim Anı:** Saf referans modelini (*"Yarın = Bugün"*, $\hat{P}_{t+1}=P_t$) hesaplatır. Saf referansın MAE'si (**4.55 TL**) LSTM'in MAE'sini (**5.07 TL**) geçer! | **TODO 8:** `lstm_mae`, `naive_mae`, `lstm_rmse`, `naive_rmse` hesaplarını yazar ve sürprizi görür. |
| **17:58–18:05** | 7 dk | **Bölüm 4: Sürpriz 2 (1 Günlük Gecikme)** | `sec4-zoom-code` | Son 35 güne yakınlaştırır (`plot_zoom_and_lag`). Kırmızı tahmin çizgisinin mavi gerçek çizginin **1 gün sağa kaydırılmış kopyası** olduğunu ve `Lag 1` korelasyonunun **0.991** çıktığını gösterir. | Yakınlaştırılmış grafiği ve gecikme korelasyonu çubuklarını çalıştırır. |
| **18:05–18:15** | 10 dk | **Bölüm 5: Yön Sınıflandırması** | `sec5-catchup`, `sec5-cls-code` | Problemi getiri özellikleri üzerinde yarının yönü (`up_day`, `BCEWithLogitsLoss`) olarak yeniden kurar. Çoğunluk sınıfı referansıyla (`%51.90`) LSTM doğruluğunu (`%54.74`) ve dengeli doğruluğu (`%53.50`) karşılaştırır. | **TODO 9:** `train_direction_classifier` çağrısını tamamlar ve metrikleri yazdırır. |
| **18:15–18:25** | 10 dk | **Bölüm 5: Walk-Forward & %50 Gerçeği** | `sec5-wf-code` | 4 katlı Walk-Forward doğrulamayı çalıştırır (`ortalama ~%51.5`). Etkin piyasalar, gürültü ve rejim değişimleri nedeniyle dürüst sonucun neden %50'ye yakın çıktığını açıklar. | Walk-Forward tablosunu ve karmaşıklık matrisini üretir. |
| **18:25–18:40** | 15 dk | **Bölüm 6: Meydan Okumalar** | `sec6-catchup`, `sec6-ch1..4` | Salonda dolaşarak öğrencilere yardım eder; son 4 dakikada Bonus 4'teki işlem maliyetli backtest grafiğini (15 bps komisyonun getiriyi nasıl erittiğini) ekranda gösterir. | Kendi sembolünü, hacim özelliğini, `GRUModel` veya `TinyTransformerModel` meydan okumalarını dener. |
| **18:40–18:45** | 5 dk | **Bölüm 7: Kapanış & Sorular** | `sec7-md` | 3 temel dersi özetler, gerçek quant ekiplerinin neyi farklı yaptığını anlatır, yasal uyarıyı hatırlatır ve soruları alır. | Notları ve bağlantıları kaydeder. |

---

## 3. Geride Kalırsak Neyi Keseceğiz? (Acil Durum Planı)

- **10 Dakika Gerideysek:**
  - Bölüm 2'de `sec2-feat-code` ve `sec2-split-code` hücrelerini canlı yazmak yerine doğrudan `sec3-catchup` (**🔄 Yetiştir: Bölüm 3**) çalıştırtıp grafik üzerinden anlatın (+5 dk kazandırır).
  - Bölüm 6 (Meydan Okumalar) süresini 15 dakikadan 10 dakikaya indirin; öğrencilerin yalnızca Meydan Okuma 1 (kendi sembolü) veya Meydan Okuma 3 (`GRUModel`) seçmesini söyleyin (+5 dk kazandırır).
  - **Asla Kesilmeyecek Bölüm:** Bölüm 4 (Fiyat Tahmini ve Sürpriz) ile Bölüm 5 (Yön Sınıflandırması).
- **20 Dakika Gerideysek:**
  - Bölüm 3'te tensor/Dataset kurulumunu `sec3-catchup` ile geçip yalnızca `SimpleLSTM.forward` ve 5 satırlık eğitim döngüsünü (`sec3-train-code`) canlı yazdırın (+8 dk kazandırır).
  - Bölüm 5'te `sec5-wf-code` (Walk-Forward) hücresini hazır çalıştırıp sonucu grafik üzerinden 2 dakikada yorumlayın (+4 dk kazandırır).
  - Bölüm 6'da bireysel kodlama yerine **Bonus 4 (İşlem Maliyetli Backtest)** çözüm hücresini birlikte çalıştırıp doğrudan kapanışa geçin (+8 dk kazandırır).

---

## 4. Konuşma Metinleri (Script Lines)

### 4.1. Bölüm 0 — Giriş (Hook) Konuşma Metni (`16:35`)
> *"Arkadaşlar ekrandaki grafiğe dikkatle bakın. Mavi çizgi Türk Hava Yolları hissesinin test dönemindeki gerçek kapanış fiyatı; kırmızı kesikli çizgi ise birazdan 15 satır PyTorch koduyla eğiteceğimiz LSTM modelinin tahmini. Neredeyse milimetrik olarak üst üste binmiş durumda! Şimdi dürüstçe el kaldıralım: Yarın sabah bu modeli canlıya bağlayıp alım-satım yapsak para kazanır mıyız? 'Kazanırız' diyenler? 'Burada bir bit yeniği var' diyenler? Güzel — cevabı vermiyorum. Önce bu modeli sıfırdan birlikte kuracağız, sonra Bölüm 4'te bu grafiğe büyüteçle bakacağız."*

### 4.2. Bölüm 4 — Sürpriz Anı (Twist) Konuşma Metni (`17:52`)
> *"Modelimiz eğitildi ve Bölüm 0'da gördüğümüz o muhteşem grafiği kendi ekranınızda ürettiniz. Şimdi kendimize şu soruyu soralım: Hiç yapay zekâ, hiç PyTorch, hiç ekran kartı kullanmasaydık; dünyanın en tembel tahmin kuralını yazsaydık ne derdik? 'Yarının fiyatı, bugünün kapanış fiyatıyla aynı olsun ($\hat{P}_{t+1} = P_t$).' Şimdi `sec4-naive-code` hücresini çalıştırın ve rakamlara bakın: LSTM modelimizin ortalama hatası **5.07 TL**, hiçbir şey öğrenmeyen 'Yarın = Bugün' kuralının hatası ise **4.55 TL**! Peki nasıl olur da grafikte bu kadar kusursuz görünen bir model, 'yarın bugünün aynısıdır' diyen saf kuraldan bile kötü çıkar? Şimdi bir alt hücreyi (`sec4-zoom-code`) çalıştırıp son 35 güne yakınlaşalım... Bakın: Kırmızı çizgi, mavi çizginin tam **1 gün sağa kaydırılmış kopyası**! Modelimiz geleceği tahmin etmeyi öğrenmedi; karesel hatayı (MSE) küçültmenin en zahmetsiz yolunun dünün fiyatını aynen tekrar etmek olduğunu öğrendi. Uzaktan bakınca binlerce günü tek grafiğe sıkıştırdığımız için o 1 günlük yatay kayma gözle görünmüyor!"*

---

## 5. Öğrencilerin En Sık Soracağı 5 Soru ve Cevapları

1. **"Bu modeli gerçek borsada alım-satım (trading) yapmak için kullanabilir miyim?"**
   - **Cevap:** Hayır. Bu atölye tamamen eğitim amaçlıdır. Gördüğünüz gibi yalnızca günlük geçmiş fiyatlarla eğitilen bir model komisyon ve alım-satım makası (slippage) düşüldüğünde para kazandırmaz; üstelik tek bir hisseye dayalı korumasız stratejiler ciddi kayıplara yol açar.
2. **"Neden daha fazla LSTM katmanı (3–4 katman) veya daha fazla nöron eklemiyoruz?"**
   - **Cevap:** Finansal günlük veride örnek sayımız sınırlı (~2.000–3.000 gün) ve gürültü oranı çok yüksektir. Katman ve parametre sayısını artırmak sinyali artırmaz; modelin eğitim setindeki rastgele gürültüyü ezberlemesine (**aşırı öğrenme / overfitting**) ve test setinde daha da kötüleşmesine neden olur.
3. **"Haberleri, KAP bildirimlerini veya sosyal medya duygu analizini (sentiment) eklesek sonuç değişir mi?"**
   - **Cevap:** Evet, gerçek dünyada tek başına fiyat geçmişinden çok daha değerli olan şey **alternatif veridir** (KAP bildirimleri, bilanço kalemleri, faiz/kur beklentileri, sektörel haber akışı). Ancak orada da zaman damgası sızıntısına (haberin seans içinde mi yoksa seans kapandıktan sonra mı yayınlandığına) çok dikkat etmek gerekir.
4. **"Neden Transformer yerine LSTM kullandık?"**
   - **Cevap:** Kısa ve düşük boyutlu (30 adım × 4 özellik) zaman serilerinde küçük bir `nn.LSTM` veya `nn.GRU`, ardışık zaman önyargısı (inductive bias) sayesinde çok hızlı eğitilir ve az veride kararlıdır. Transformer mimarileri özellikle çok uzun bağlamlarda veya çok değişkenli büyük veri kümelerinde fark yaratır (Bölüm 6 Bonus'ta küçük bir Transformer Encoder'ı da bizzat deniyoruz!).
5. **"%52–%54 doğruluk (accuracy) gerçek hayatta işe yarar mı?"**
   - **Cevap:** Tek bir hissede yüksek işlem maliyetiyle hayır; komisyonlar o %2–%4'lük farkı eritir (Bölüm 6'daki backtest grafiğinde gördüğümüz gibi). Ancak düşük işlem maliyetli bir kurumsal altyapıda, yüzlerce bağımsız hissede ve sıkı risk yönetimiyle %52–%53'lük istikrarlı bir üstünlük (edge), Büyük Sayılar Yasası sayesinde anlamlı bir Sharpe oranına dönüşebilir.
