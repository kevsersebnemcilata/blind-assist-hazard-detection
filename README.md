# Gerçek Zamanlı Tehlike Tespit Sistemi

Görme engelli bireylere yönelik, tek kamera ile çalışan hibrit tehlike tespit sistemi. YOLOv8, ByteTrack, MiDaS ve Optik Akış bileşenlerini birleştirerek her nesne için risk skoru hesaplar; kritik durumlarda Türkçe sesli uyarı verir.

---

## Sistem Mimarisi

```
Kamera Görüntüsü
      │
      ├─► YOLOv8s (nesne tespiti)
      │        └─► ByteTrack + Re-ID (kararlı nesne takibi)
      │
      ├─► MiDaS_small (derinlik tahmini — async thread)
      │
      ├─► Farneback Optik Akış (hareket haritası)
      │
      └─► Risk Hesaplama
               ├─ Nesne başına: sınıf önceliği + derinlik + hareket + yaklaşma + bbox alanı
               └─ Sahne geneli: DangerAnalyzer (zamansal düzleştirme + trend)
                        └─► Sesli Uyarı (ID bazlı, her ID için bir kez)
```

### Risk Skoru Bileşenleri

| Bileşen | Ağırlık | Açıklama |
|---|---|---|
| Sınıf önceliği | 0.28 | Araç > insan > bisiklet > trafik öğesi |
| Derinlik | 0.22 | MiDaS normalize derinlik değeri |
| Hareket | 0.20 | Optik akış büyüklüğü (nesne bölgesi) |
| Yaklaşma | 0.18 | Zaman içindeki derinlik değişimi |
| BBox alanı | 0.07 | Frame içindeki göreli nesne büyüklüğü |
| Güven skoru | 0.05 | YOLO tespit güveni |

Risk skoru 0.65 eşiğini aşan her nesne için sesli uyarı tetiklenir.

---

## Özellikler

- **Hibrit tehlike skoru** — YOLO, MiDaS ve Optik Akış çıktıları tek bir skorda birleştirilir
- **Kararlı nesne takibi** — ByteTrack + özel Re-ID tamponu (kameradan çıkıp giren nesnelere aynı ID yeniden atanır)
- **Asenkron derinlik tahmini** — MiDaS ayrı thread'de çalışır, ana döngüyü bloke etmez
- **Sesli uyarı (Türkçe)** — Her track ID için yalnızca bir kez, yön bilgisiyle birlikte ("solda araç yaklaşıyor!")
- **Web arayüzü** — Tarayıcıdan MJPEG canlı görüntü, Web Speech API ile ses; aynı anda birden fazla cihaz desteklenir
- **CSV loglama** — Her frame için tehlike skoru, hareket, derinlik ve trend kaydedilir

---

## Gereksinimler

- Python 3.10+
- CUDA destekli GPU *(opsiyonel ama önerilir — CPU'da FPS düşer)*
- Webcam

### Bağımlılıklar

```
opencv-python >= 4.8.0
numpy >= 1.21.0
torch >= 2.0.0
torchvision >= 0.15.0
ultralytics >= 8.0.0
flask >= 3.0.0
pyttsx3 >= 2.90
matplotlib >= 3.5.0
```

---

## Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/neslihankaradenizz/Bitirme-Projesi.git
cd Bitirme-Projesi

# 2. Sanal ortam oluştur (önerilir)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

### Model dosyaları

İlk çalıştırmada MiDaS ağırlıkları otomatik indirilir ve `models/` klasörüne kaydedilir.  
YOLOv8s ağırlıklarını manuel olarak indirip `models/` klasörüne koy:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
mv yolov8s.pt models/
```

---

## Çalıştırma

### Terminal modu (cv2 penceresi)

```bash
python main.py
# Çıkmak için: q tuşu
```

### Web arayüzü

```bash
python web_app.py
# Tarayıcıda aç: http://localhost:5000
```

**Telefon / uzak cihazdan erişim:**  
Bilgisayarının yerel IP adresini öğren (`ipconfig` / `ip a`) ve telefon tarayıcısına yaz:

```
http://192.168.x.x:5000
```

Tüm cihazların aynı Wi-Fi ağında olması yeterli.

---

## Dizin Yapısı

```
Bitirme-Projesi/
├── main.py                  # Terminal modu (cv2.imshow)
├── web_app.py               # Flask web sunucusu
├── requirements.txt
│
├── src/
│   ├── core/
│   │   └── danger_analyzer.py       # Sahne geneli tehlike skoru + trend
│   ├── modules/
│   │   ├── depth_estimator.py       # MiDaS async wrapper
│   │   ├── optical_flow.py          # Farneback optik akış
│   │   └── object_tracker.py        # YOLOv8 + ByteTrack + Re-ID
│   └── utils/
│       ├── config.py                # Tüm parametreler
│       ├── audio_alert.py           # Türkçe sesli uyarı (espeak / pyttsx3)
│       ├── approach_tracker.py      # Nesne bazlı yaklaşma takibi
│       ├── risk.py                  # Risk skoru hesaplama
│       ├── overlay.py               # HUD çizimi
│       └── logger.py                # CSV loglama
│
├── templates/
│   └── index.html           # Web arayüzü
│
├── scripts/
│   ├── download_models.py
│   └── plot_logs.py         # Log verilerini görselleştirme
│
├── models/                  # Ağırlık dosyaları (.pt)
├── logs/                    # CSV çıktıları
└── outputs/                 # Kayıt videoları
```

---

## Yapılandırma

Tüm parametreler `src/utils/config.py` dosyasındadır:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam indeksi |
| `FRAME_SCALE` | `0.5` | İşleme çözünürlüğü (1.0 = tam) |
| `DEPTH_EVERY_N_FRAMES` | `5` | MiDaS kaç frame'de bir çalışır |
| `FLOW_EVERY_N_FRAMES` | `3` | Optik akış kaç frame'de bir çalışır |
| `YOLO_CONF_THRESHOLD` | `0.4` | YOLO minimum güven eşiği |
| `TEHLIKE_THRESHOLD` | `0.45` | Tehlike seviyesi eşiği |
| `DIKKAT_THRESHOLD` | `0.30` | Dikkat seviyesi eşiği |
| `ENABLE_BYTETRACK` | `True` | ByteTrack tracking aç/kapat |

---

## Log Görselleştirme

```bash
python scripts/plot_logs.py
```

`logs/` klasöründeki CSV dosyalarından tehlike skoru, hareket ve derinlik grafiklerini çizer.

---

## Sorun Giderme

**Kamera açılmıyor**  
`config.py` içinde `CAMERA_INDEX` değerini değiştir (0, 1, 2 dene).

**Düşük FPS**  
GPU kurulumunu kontrol et. CPU'da YOLOv8s + MiDaS kombinasyonu ağırdır; `YOLO_MODEL_PATH`'i `models/yolov8n.pt` olarak değiştirip daha hafif modele geçebilirsin.

**Sesli uyarı çalışmıyor (terminal modu)**  
`espeak-ng` yükle:
```bash
# Ubuntu / Debian
sudo apt install espeak-ng
# Windows
# https://github.com/espeak-ng/espeak-ng/releases adresinden yükleyici indir
```

**Web Speech API sesi çıkmıyor (tarayıcı)**  
Tarayıcılar ilk kullanıcı etkileşimi olmadan ses çalmaz. "Başlat" butonuna tıklamak bu kısıtlamayı kaldırır.
