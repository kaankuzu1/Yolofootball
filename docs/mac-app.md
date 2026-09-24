# 1v1 Analiz: Mac uygulaması

Kamerayı seç, 1v1'i kaydet, kayıt bitince gol, şut, pas, top kapma ve çalımların
zaman damgalı listesini al. Analiz, komut satırındaki `football-analyse` ile
aynıdır; uygulama sadece onun önüne bir pencere koyar.

## Kurulum (bir kez)

1. **Python 3.11 veya daha yenisi.** Terminal'de `python3 --version` yazın.
   Eskiyse [python.org](https://www.python.org/downloads/macos/) kurulum
   dosyasını ya da `brew install python@3.12` kullanın.
2. **Model dosyaları.** GitHub'a sığmadıkları için depoda yoklar. Şu üçünü
   `assets/models/` klasörüne koyun: `forzasys_soccer.pt`, `yolo11s-pose.pt`,
   `goal_yolo11n.pt` (proje klasöründeki teslim zip'inde var).
3. **Çalıştırın.** Finder'da `scripts/mac/1v1 Analiz.command` dosyasına çift
   tıklayın. İlk açılışta kendi Python ortamını (`.venv`) kurar; 1-2 GB indirir,
   birkaç dakika sürer. Sonraki açılışlar saniyeler sürer.
   (Terminal'den: `python -m football_analysis.app`)
4. **İsteğe bağlı: Uygulamalar'a ekleyin.** `bash scripts/mac/make_app.sh`
   çalıştırınca `~/Applications/1v1 Analiz.app` oluşur; Launchpad'den,
   Spotlight'tan ve Dock'tan açılır.

İlk açılışta macOS kamera izni sorar. `.command` dosyasıyla açtıysanız izin
Terminal'e verilir. Reddettiyseniz: Sistem Ayarları › Gizlilik ve Güvenlik ›
Kamera.

## Kullanım

1. **Kamera seçin.** Sol üstteki listede Mac'in ön kamerası (FaceTime HD)
   varsayılan olarak ilk sıradadır. USB kamera ya da iPhone (Süreklilik
   Kamerası) bağlayınca listeye kendiliğinden eklenir; eklenmezse ↻'ya basın.
   Seçim bir sonraki açılışta hatırlanır.
2. **Kamerayı sabitleyin.** Sistem kameranın oynamadığını varsayar (tripod ya
   da sabit bir yer). Kale tamamen görünsün.
3. **Kaleyi işaretleyin** (G). Kalenin dört köşesine sırayla tıklayın: sol
   direğin dibi, sağ direğin dibi, üst direğin sağ ucu, üst direğin sol ucu.
   Kale ölçüsünü listeden seçin ya da girin. Her kamera için ayrı saklanır; kamera
   yerinden oynamadıkça bir kez yeter. Kale olmadan da çalışır ama gol ve
   isabetli şut bulunamaz.
4. **Kaydedin** (R ya da boşluk). Canlı görüntüde oyuncular, top ve kale
   kutuyla gösterilir. Kaydı bitirince analiz kendiliğinden başlar.
5. **İnceleyin.** Sağdaki listede bir olaya tıklayınca video o anın 1,5 sn
   öncesinden oynar. Alttaki şeritte olaylar renkli çizgilerdir; tıklayınca
   oraya gider. Üstteki sayaçlara tıklayınca liste o türe süzülür. "kontrol et"
   etiketi, kuralın emin olamadığı olayları gösterir.
6. **Dışa aktarın.** CSV (Excel/Numbers), JSON (tam zaman çizelgesi) ya da
   "Klasörü aç".

Kısayollar: Boşluk oynat/durdur (kamerada: kayıt) · ←/→ kare kare · G kale · Esc iptal.

Daha önce çekilmiş bir videoyu "Video dosyası aç…" ile açıp aynı şekilde
analiz edebilirsiniz.

## Dosyalar nereye gider

Varsayılan: `~/Movies/Football Analysis/` (sol alttaki "Kayıt klasörü…" ile
değişir). Her kaydın sonuçları yanında, aynı adla durur:

| Dosya | İçerik |
| --- | --- |
| `2026-09-24 18.05.12.mp4` | kayıt |
| `….goal.json` | kalenin köşeleri ve ölçüsü |
| `….events.json` | olaylar (CLI ile aynı biçim) |
| `….annotated.mp4` | kutular ve olay yazıları çizilmiş kopya |
| `….states.jsonl` | kare kare kayıt; `football-analyse --replay` ile kurallar saniyeler içinde yeniden çalışır |

## Bilinmesi gerekenler

- **Olaylar kayıt bittikten sonra çıkar, canlı değil.** Olay kuralları kararı
  klibin tamamına bakarak verir (bir şutun gol olup olmadığı birkaç kare sonra
  belli olur), bu yüzden canlı görüntüde yalnızca kutular var.
- **Hız.** Mac'te Apple silikon GPU'su (MPS) otomatik kullanılır; bu makinede
  ölçülmedi. GPU'suz bir bilgisayarda analiz saniyede ~1,4 kare işler (3
  dakikalık klip ≈ 55 dk). "Hız" ayarındaki "2 karede bir" süreyi yarıya
  indirir, zaman hassasiyetini biraz düşürür.
- **Zaman damgaları.** Web kameraları loş ışıkta kare hızını düşürür. Kayıt,
  her kareyi geldiği ana göre yerleştirir, böylece videodaki 12,40 sn gerçekten
  kaydın 12,40. saniyesidir (±1 kare, 33 ms).
- **Canlı kutular** tespit süresi kadar geride kalır; GPU'da fark edilmez,
  CPU'da yarım saniyeye çıkar.
