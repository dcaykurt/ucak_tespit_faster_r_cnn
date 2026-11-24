import os
import sys
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from collections import namedtuple
from object_detection.utils import dataset_util

def sinif_adi_sayiya(sinif_adi):
    if sinif_adi == 'ucak':
        return 1
    return 0

def tablolari_grupla(tablo, sutun):
    veri_yapisi = namedtuple('veri_yapisi', ['filename', 'object'])
    gruplar = tablo.groupby(sutun)
    sonuc = []
    for isim, grup in gruplar:
        sonuc.append(veri_yapisi(isim, grup))
    return sonuc

def ornek_olustur(grup, resim_klasor):
    resim_yol = os.path.join(resim_klasor, grup.filename)
    with tf.gfile.GFile(resim_yol, 'rb') as dosya:
        kodlanmis_resim = dosya.read()
    resim_bayt = io.BytesIO(kodlanmis_resim)
    resim = Image.open(resim_bayt)
    genislik, yukseklik = resim.size

    dosya_adi = grup.filename.encode('utf8')
    resim_formati = b'jpeg'
    x_minler = []
    x_maxlar = []
    y_minler = []
    y_maxlar = []
    sinif_metinleri = []
    sinif_sayilari = []

    for index, satir in grup.object.iterrows():
        x_minler.append(float(satir['xmin']) / float(genislik))
        x_maxlar.append(float(satir['xmax']) / float(genislik))
        y_minler.append(float(satir['ymin']) / float(yukseklik))
        y_maxlar.append(float(satir['ymax']) / float(yukseklik))
        sinif_metinleri.append(satir['class'].encode('utf8'))
        sinif_sayilari.append(sinif_adi_sayiya(satir['class']))

    tf_ornek = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(yukseklik),
        'image/width': dataset_util.int64_feature(genislik),
        'image/filename': dataset_util.bytes_feature(dosya_adi),
        'image/source_id': dataset_util.bytes_feature(dosya_adi),
        'image/encoded': dataset_util.bytes_feature(kodlanmis_resim),
        'image/format': dataset_util.bytes_feature(resim_formati),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_minler),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_maxlar),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_minler),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_maxlar),
        'image/object/class/text': dataset_util.bytes_list_feature(sinif_metinleri),
        'image/object/class/label': dataset_util.int64_list_feature(sinif_sayilari),
    }))
    return tf_ornek

csv_yolu = sys.argv[1]
resim_klasoru = sys.argv[2]
cikti_yolu = sys.argv[3]

yazici = tf.python_io.TFRecordWriter(cikti_yolu)
tablo = pd.read_csv(csv_yolu)
gruplar = tablolari_grupla(tablo, 'filename')
ornek_sayisi = 0

for grup in gruplar:
    tf_ornek = ornek_olustur(grup, resim_klasoru)
    yazici.write(tf_ornek.SerializeToString())
    ornek_sayisi = ornek_sayisi + 1

yazici.close()
print('{}: {} ornek yazildi.'.format(cikti_yolu, ornek_sayisi))
