# Doğrulanmış Ortam ve API Davranışları (`VERIFIED.md`)

Bu belge, **"Finansal Verilerle Yapay Zekâyı Buluşturmak"** PyTorch atölyesi geliştirilirken çalışma ortamında bizzat test edilerek doğrulanmış paket sürümlerini, `yfinance` API davranışlarını ve veri yedekleme (fallback) kararlarını kayıt altına alır.

---

## 1. Google Colab Çalışma Ortamı Sürümleri (17 Kasım 2026)

| Paket | Doğrulanmış Sürüm | Notlar |
|---|---|---|
| `torch` | `2.14.1+cu130` | CPU üzerinde tüm eğitim hücreleri < 15 sn içinde tamamlanmaktadır. |
| `yfinance` | `0.2.66` | `curl_cffi` tabanlı `requests.Session(impersonate="chrome")` kullanır. |
| `pandas` | `2.2.3` | `DatetimeIndex` ve zaman serisi işlemleri doğrulandı. |
| `numpy` | `2.4.6` | Sabit tohumlama (`np.random.seed`) doğrulandı. |
| `scikit-learn` | `1.6.1` | `StandardScaler` ve sınıflandırma metrikleri için kullanıldı. |
| `matplotlib` | `3.10.0` | Tüm görselleştirmeler ek kütüphane olmadan üretildi. |

---

## 2. `yfinance` (`0.2.66`) API Doğrulaması

### 2.1. `yf.download` İmzası ve Varsayılan Argümanlar

```python
yf.download(
    tickers, start=None, end=None, actions=False, threads=True,
    ignore_tz=None, group_by='column', auto_adjust=None, back_adjust=False,
    repair=False, keepna=False, progress=True, period=None, interval='1d',
    prepost=False, proxy=_SENTINEL_, rounding=False, timeout=10,
    session=None, multi_level_index=True
)
```

Doğrulanan kritik davranışlar:

1. **`auto_adjust` varsayılanı (`auto_adjust=None` → `True`):**
   - Fonksiyon imzasında `auto_adjust=None` olarak tanımlıdır. Argüman verilmezse `FutureWarning: YF.download() has changed argument auto_adjust default to True` uyarısı üretir ve içeride `auto_adjust=True` değerini kullanır.
   - `auto_adjust=True` olduğunda dönen sütunlar `['Close', 'High', 'Low', 'Open', 'Volume']` şeklindedir; ayrıca bir `'Adj Close'` sütunu **dönmez** (kapanış ve OHLC fiyatları temettü ve bölünmelere göre otomatik düzeltilmiştir).
   - Uyarıyı bastırmak ve davranışı sabitlemek için `auto_adjust=True` açıkça geçirilir.
2. **Tek sembol (ticker) için bile `MultiIndex` sütunlar (`multi_level_index=True`):**
   - `yf.download("THYAO.IS", ...)` tek bir sembolle çağrıldığında bile varsayılan `multi_level_index=True` nedeniyle sütunlar `pd.MultiIndex` olarak döner: `[('Close', 'THYAO.IS'), ('High', 'THYAO.IS'), ...]`.
   - Tek seviyeli sütun elde etmek için `multi_level_index=False` verilmeli veya `isinstance(df.columns, pd.MultiIndex)` kontrolü ile seviye düşürülmelidir (`df.columns.get_level_values(0)`). `load_prices` her iki durumu da güvenli biçimde düzleştirir.
3. **Zaman dilimi (`ignore_tz`):**
   - Günlük (`interval="1d"`) veride `yf.download` varsayılan olarak `ignore_tz=True` kullanır ve `DatetimeIndex` zaman dilimini (`tz_localize(None)`) temizler.

### 2.2. Hata Sınıfları (`yfinance.exceptions`) ve Rate-Limit Davranışı

`yfinance.exceptions` modülünde yer alan sınıflar:
- `YFException` (temel sınıf)
- `YFRateLimitError` (`"Too Many Requests. Rate limited. Try after a while."`)
- `YFPricesMissingError`
- `YFTzMissingError`
- `YFTickerMissingError`
- `YFDataException`, `YFEarningsDateMissing`, `YFInvalidPeriodError`, `YFNotImplementedError`

### 2.3. `yf.download` ile `Ticker.history` Arasındaki Kritik Farklar

| Özellik | `yf.download(ticker, ...)` | `yf.Ticker(ticker).history(...)` |
|---|---|---|
| **Rate-limit (`YFRateLimitError`) davranışı** | `yfinance.multi._download_one` tüm `Exception` sınıflarını `try/except` ile **yakalar**, `yfinance.shared._ERRORS[ticker]` içine yazar ve **boş `DataFrame`** döndürür (istisna fırlatmaz!). | `PriceHistory.history` içinde `except YFRateLimitError: raise` bloğu vardır; `raise_errors=False` (varsayılan) olsa bile `YFRateLimitError` istisnasını **doğrudan fırlatır**. |
| **Diğer hatalar (geçersiz sembol, ağ hatası)** | Yakalar, `shared._ERRORS` içine kaydeder ve boş `DataFrame` döndürür. | `raise_errors=False` (varsayılan) ise boş `DataFrame` döndürür; `raise_errors=True` ise `YFTzMissingError` / `YFPricesMissingError` fırlatır. |
| **Sütun yapısı** | Varsayılan `MultiIndex` (`multi_level_index=True`), 5 sütun (`Open, High, Low, Close, Volume`). | Her zaman tek seviyeli `Index`, varsayılan `actions=True` nedeniyle `Dividends` ve `Stock Splits` sütunlarını da içerir. |
| **Zaman dilimi (`DatetimeIndex.tz`)** | Günlük veride `ignore_tz=True` → `tz=None`. | Borsa yerel saat dilimine (`Europe/Istanbul`, `America/New_York` vb.) sahip `DatetimeIndex` döndürür. |

**Sonuç:** `src/workshop_utils.py` içindeki `load_prices` fonksiyonu:
- Hem `YFRateLimitError` ve genel `Exception` sınıflarını yakalar,
- Hem `yfinance.shared._ERRORS` kaydını ve dönen `DataFrame`'in boş olup olmadığını kontrol eder,
- `MultiIndex` sütunları ve zaman dilimi bilgilerini standartlaştırarak her zaman tek seviyeli `['Open', 'High', 'Low', 'Close', 'Volume']` sütunlarına ve `tz=None` `DatetimeIndex`'e sahip temiz bir `DataFrame` döndürür.

---

## 3. Yahoo Finance Kullanım Koşulları ve Yedek Veri (Snapshot) Kararı

- **Koşullar:** Yahoo Kullanım Koşulları (Terms of Service), Yahoo Finance piyasa verilerinin herkese açık bir depoda (public GitHub repo) yeniden dağıtılmasına izin vermez.
- **Uygulanan Çözüm:**
  1. `data/*.csv` dosyaları `.gitignore` içine eklenerek git geçmişinin dışında tutulmuştur.
  2. `src/workshop_utils.py` içinde tek bir sabit (`SNAPSHOT_BASE_URL`) tanımlanmıştır; eğitmen katılımcılarla paylaşacağı özel bulut/bucket adresini bu sabit (veya `WORKSHOP_SNAPSHOT_URL` ortam değişkeni) üzerinden sağlar.
  3. `load_prices` sırasıyla:
     - Önce canlı `yfinance` indirmesini dener,
     - Hata, boş tablo veya rate-limit durumunda Türkçe bilgi mesajı yazdırarak önce yerel `data/{ticker}.csv` önbelleğine, ardından `SNAPSHOT_BASE_URL` adresine, en son olarak da tamamen çevrimdışı (offline) temiz bir Colab çalışma zamanında bile çalışmayı garanti eden kalibre edilmiş deterministik çevrimdışı yedek üreticiye başvurur.
