import os
import glob
import cv2
import pandas as pd

def veri_donustur(bolum):
    resim_klasor = os.path.join('veri', 'OID', bolum, 'Airplane')
    etiket_klasor = os.path.join(resim_klasor, 'Label')
    cikti_csv = os.path.join('veri', bolum + '_etiketler.csv')

    satirlar = []
    atlanan_kutu = 0

    resim_yollari = glob.glob(os.path.join(resim_klasor, '*.jpg'))
    for resim_yol in resim_yollari:
        dosya_adi = os.path.basename(resim_yol)
        etiket_adi = os.path.splitext(dosya_adi)[0] + '.txt'
        etiket_yol = os.path.join(etiket_klasor, etiket_adi)

        if not os.path.exists(etiket_yol):
            continue

        resim = cv2.imread(resim_yol)
        yukseklik = resim.shape[0]
        genislik = resim.shape[1]

        dosya = open(etiket_yol, 'r')
        for satir in dosya:
            satir = satir.strip()
            if not satir:
                continue
            parcalar = satir.split(' ')
            if len(parcalar) < 5:
                continue
            x1 = int(round(float(parcalar[1])))
            y1 = int(round(float(parcalar[2])))
            x2 = int(round(float(parcalar[3])))
            y2 = int(round(float(parcalar[4])))

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > genislik:
                x2 = genislik
            if y2 > yukseklik:
                y2 = yukseklik

            kutu_genislik = x2 - x1
            kutu_yukseklik = y2 - y1

            if kutu_genislik < 2 or kutu_yukseklik < 2:
                atlanan_kutu = atlanan_kutu + 1
                continue

            satirlar.append([dosya_adi, genislik, yukseklik, 'ucak', x1, y1, x2, y2])
        dosya.close()

    sutunlar = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(satirlar, columns=sutunlar)
    df.to_csv(cikti_csv, index=False)
    print('{}: {} satir yazildi, {} kutu atlandi.'.format(bolum, len(satirlar), atlanan_kutu))

veri_donustur('train')
veri_donustur('test')
