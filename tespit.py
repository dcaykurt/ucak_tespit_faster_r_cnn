import os
import glob
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_YOLU = 'modeller/ucak_model/frozen_inference_graph.pb'
RESIM_KLASOR = 'veri/OID/test/Airplane'
ETIKET_CSV = 'veri/test_etiketler.csv'
CIKTI_KLASOR = 'sonuclar/tespit'
ESIK = 0.5
IOU_ESIK = 0.5


def iou_hesapla(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    kesisim = max(0, x2 - x1) * max(0, y2 - y1)
    alan_a = (a[2] - a[0]) * (a[3] - a[1])
    alan_b = (b[2] - b[0]) * (b[3] - b[1])
    birlesim = alan_a + alan_b - kesisim
    if birlesim <= 0:
        return 0.0
    return float(kesisim) / float(birlesim)


def kutu_ciz(resim, kutu, skor, renk, kalinlik):
    cv2.rectangle(resim, (kutu[0], kutu[1]), (kutu[2], kutu[3]), renk, kalinlik)
    yazi = 'ucak {}%'.format(int(skor * 100))
    olcek = kalinlik * 0.3
    (yw, yh), _ = cv2.getTextSize(yazi, cv2.FONT_HERSHEY_SIMPLEX, olcek, 1)
    y = kutu[1]
    if y - yh - 6 < 0:
        y = kutu[1] + yh + 6
    cv2.rectangle(resim, (kutu[0], y - yh - 6), (kutu[0] + yw + 4, y), renk, -1)
    cv2.putText(resim, yazi, (kutu[0] + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, olcek, (255, 255, 255), 1)


gercekler = {}
tablo = pd.read_csv(ETIKET_CSV)
for index, satir in tablo.iterrows():
    ad = satir['filename']
    if ad not in gercekler:
        gercekler[ad] = []
    gercekler[ad].append([int(satir['xmin']), int(satir['ymin']), int(satir['xmax']), int(satir['ymax'])])

graf = tf.Graph()
with graf.as_default():
    graf_tanim = tf.GraphDef()
    with tf.gfile.GFile(MODEL_YOLU, 'rb') as dosya:
        graf_tanim.ParseFromString(dosya.read())
    tf.import_graph_def(graf_tanim, name='')

oturum = tf.Session(graph=graf)
giris = graf.get_tensor_by_name('image_tensor:0')
kutular_t = graf.get_tensor_by_name('detection_boxes:0')
skorlar_t = graf.get_tensor_by_name('detection_scores:0')
sayi_t = graf.get_tensor_by_name('num_detections:0')

if not os.path.isdir(CIKTI_KLASOR):
    os.makedirs(CIKTI_KLASOR)

tespit_dosya = open('sonuclar/tespitler.csv', 'w')
tespit_dosya.write('resim;skor;xmin;ymin;xmax;ymax;iou;dogru\n')
ozet_dosya = open('sonuclar/resim_ozet.csv', 'w')
ozet_dosya.write('resim;gercek_kutu;tespit;dogru;yanlis;kacirilan\n')

toplam_gercek = 0
toplam_dogru = 0
toplam_yanlis = 0

resimler = sorted(glob.glob(os.path.join(RESIM_KLASOR, '*.jpg')))
for resim_yol in resimler:
    ad = os.path.basename(resim_yol)
    resim = cv2.imread(resim_yol)
    yukseklik = resim.shape[0]
    genislik = resim.shape[1]
    rgb = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
    girdi = np.expand_dims(rgb, axis=0)

    kutular, skorlar, sayi = oturum.run([kutular_t, skorlar_t, sayi_t], feed_dict={giris: girdi})

    gercek = []
    if ad in gercekler:
        gercek = gercekler[ad]
    eslesti = []
    for i in range(len(gercek)):
        eslesti.append(False)

    kalinlik = max(2, int(genislik / 300))
    dogru = 0
    yanlis = 0
    for i in range(int(sayi[0])):
        skor = float(skorlar[0][i])
        if skor < ESIK:
            continue
        k = kutular[0][i]
        kutu = [int(k[1] * genislik), int(k[0] * yukseklik), int(k[3] * genislik), int(k[2] * yukseklik)]

        en_iyi_iou = 0.0
        en_iyi_j = -1
        for j in range(len(gercek)):
            if eslesti[j]:
                continue
            iou = iou_hesapla(kutu, gercek[j])
            if iou > en_iyi_iou:
                en_iyi_iou = iou
                en_iyi_j = j

        if en_iyi_iou >= IOU_ESIK:
            eslesti[en_iyi_j] = True
            dogru = dogru + 1
            sonuc = 1
            kutu_ciz(resim, kutu, skor, (0, 200, 0), kalinlik)
        else:
            yanlis = yanlis + 1
            sonuc = 0
            kutu_ciz(resim, kutu, skor, (0, 0, 230), kalinlik)

        tespit_dosya.write('{};{:.3f};{};{};{};{};{:.3f};{}\n'.format(
            ad, skor, kutu[0], kutu[1], kutu[2], kutu[3], en_iyi_iou, sonuc))

    kacirilan = len(gercek) - dogru
    for j in range(len(gercek)):
        if not eslesti[j]:
            g = gercek[j]
            cv2.rectangle(resim, (g[0], g[1]), (g[2], g[3]), (0, 200, 255), kalinlik)

    ozet_dosya.write('{};{};{};{};{};{}\n'.format(ad, len(gercek), dogru + yanlis, dogru, yanlis, kacirilan))
    cv2.imwrite(os.path.join(CIKTI_KLASOR, ad), resim)

    toplam_gercek = toplam_gercek + len(gercek)
    toplam_dogru = toplam_dogru + dogru
    toplam_yanlis = toplam_yanlis + yanlis
    print('{}: {} gercek, {} dogru, {} yanlis'.format(ad, len(gercek), dogru, yanlis))

tespit_dosya.close()
ozet_dosya.close()

kesinlik = 0.0
if toplam_dogru + toplam_yanlis > 0:
    kesinlik = float(toplam_dogru) / float(toplam_dogru + toplam_yanlis)
duyarlilik = 0.0
if toplam_gercek > 0:
    duyarlilik = float(toplam_dogru) / float(toplam_gercek)
f1 = 0.0
if kesinlik + duyarlilik > 0:
    f1 = 2 * kesinlik * duyarlilik / (kesinlik + duyarlilik)

dosya = open('sonuclar/genel_sonuc.txt', 'w')
dosya.write('esik: {}\n'.format(ESIK))
dosya.write('resim sayisi: {}\n'.format(len(resimler)))
dosya.write('gercek kutu: {}\n'.format(toplam_gercek))
dosya.write('dogru tespit: {}\n'.format(toplam_dogru))
dosya.write('yanlis tespit: {}\n'.format(toplam_yanlis))
dosya.write('kacirilan: {}\n'.format(toplam_gercek - toplam_dogru))
dosya.write('kesinlik: {:.3f}\n'.format(kesinlik))
dosya.write('duyarlilik: {:.3f}\n'.format(duyarlilik))
dosya.write('f1: {:.3f}\n'.format(f1))
dosya.close()

print('')
print('kesinlik: {:.3f}  duyarlilik: {:.3f}  f1: {:.3f}'.format(kesinlik, duyarlilik, f1))
