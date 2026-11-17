# Yedek Veri (Snapshot) Klasörü

Yahoo Finance Kullanım Koşulları (Terms of Service) uyarınca ham piyasa verileri (`*.csv`) herkese açık git deposuna dahil edilmemiştir (`.gitignore` içinde hariç tutulmuştur).

## Yedek Verilerin Oluşturulması ve Sunulması

1. **Atölye Öncesi (Eğitmen):**
   ```bash
   python3 scripts/snapshot_data.py
   ```
   Bu komut `THYAO.IS`, `ASELS.IS`, `XU100.IS`, `BTC-USD` ve `AAPL` sembollerinin `2015-01-01` – bugün arası günlük OHLCV verilerini `data/*.csv` olarak indirir. Eğitmen bu dosyaları yalnızca katılımcılarla paylaşılan bulut klasörüne (`src/workshop_utils.py` içindeki `SNAPSHOT_BASE_URL` sabiti) yükler.

2. **Çevrimdışı / CI Ortamı:**
   `load_prices` fonksiyonu `yfinance` erişimi başarısız olduğunda sırasıyla:
   - Yerel `data/{ticker}.csv` dosyasını,
   - `SNAPSHOT_BASE_URL` adresindeki CSV dosyasını,
   - Hiçbiri yoksa `workshop_utils.py` içindeki kalibre edilmiş deterministik çevrimdışı yedek üreticiyi kullanır.
   Böylece temiz ve tamamen çevrimdışı bir Colab çalışma zamanında veya CI testlerinde bile tüm hücreler sorunsuz çalışır.
