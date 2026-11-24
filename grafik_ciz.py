import os
import glob
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EGITIM_KLASOR = 'egitim/ucak'
DEGERLENDIRME_KLASOR = 'egitim/ucak_eval_tum'
CIKTI_KLASOR = 'sonuclar'
GRAFIK_KLASOR = 'sonuclar/grafikler'

TOPLAM = 'Losses/TotalLoss'
RPN_NESNE = 'Losses/Loss/RPNLoss/objectness_loss'
RPN_KONUM = 'Losses/Loss/RPNLoss/localization_loss'
SINIF = 'Losses/Loss/BoxClassifierLoss/classification_loss'
KUTU = 'Losses/Loss/BoxClassifierLoss/localization_loss'
MAP = 'PascalBoxes_Precision/mAP@0.5IOU'
PENCERE = 5


def olaylari_oku(klasor, etiketler):
    degerler = {}
    for etiket in etiketler:
        degerler[etiket] = {}
    dosyalar = glob.glob(os.path.join(klasor, 'events.out.tfevents.*'))
    for dosya in dosyalar:
        for olay in tf.train.summary_iterator(dosya):
            for deger in olay.summary.value:
                if deger.tag in degerler:
                    degerler[deger.tag][olay.step] = deger.simple_value
    return degerler


def yumusat(liste, pencere):
    sonuc = []
    for i in range(len(liste)):
        bas = max(0, i - pencere + 1)
        toplam = 0.0
        for j in range(bas, i + 1):
            toplam = toplam + liste[j]
        sonuc.append(toplam / (i - bas + 1))
    return sonuc


def sirali(sozluk):
    adimlar = sorted(sozluk.keys())
    sayilar = []
    for adim in adimlar:
        sayilar.append(sozluk[adim])
    return adimlar, sayilar


if not os.path.isdir(GRAFIK_KLASOR):
    os.makedirs(GRAFIK_KLASOR)

kayip_etiketleri = [TOPLAM, RPN_NESNE, RPN_KONUM, SINIF, KUTU]
egitim = olaylari_oku(EGITIM_KLASOR, kayip_etiketleri)
degerlendirme = olaylari_oku(DEGERLENDIRME_KLASOR, [MAP, RPN_NESNE, RPN_KONUM, SINIF, KUTU])

adimlar, toplam = sirali(egitim[TOPLAM])
dosya = open(os.path.join(CIKTI_KLASOR, 'egitim_kayiplari.csv'), 'w')
dosya.write('adim;toplam;rpn_nesnellik;rpn_konum;siniflandirma;kutu_konum\n')
for i in range(len(adimlar)):
    a = adimlar[i]
    dosya.write('{};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f}\n'.format(
        a, egitim[TOPLAM][a], egitim[RPN_NESNE][a], egitim[RPN_KONUM][a], egitim[SINIF][a], egitim[KUTU][a]))
dosya.close()

map_adimlar, map_degerler = sirali(degerlendirme[MAP])
test_toplam = []
dosya = open(os.path.join(CIKTI_KLASOR, 'degerlendirme.csv'), 'w')
dosya.write('adim;map_0_5;test_toplam_kayip;rpn_nesnellik;rpn_konum;siniflandirma;kutu_konum\n')
for a in map_adimlar:
    t = degerlendirme[RPN_NESNE][a] + degerlendirme[RPN_KONUM][a] + degerlendirme[SINIF][a] + degerlendirme[KUTU][a]
    test_toplam.append(t)
    dosya.write('{};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f}\n'.format(
        a, degerlendirme[MAP][a], t, degerlendirme[RPN_NESNE][a], degerlendirme[RPN_KONUM][a],
        degerlendirme[SINIF][a], degerlendirme[KUTU][a]))
dosya.close()

en_iyi = 0
for i in range(len(map_degerler)):
    if map_degerler[i] > map_degerler[en_iyi]:
        en_iyi = i

plt.figure(figsize=(8, 5))
plt.plot(adimlar, toplam, color='tab:blue', alpha=0.3, linewidth=1, label='ham')
plt.plot(adimlar, yumusat(toplam, PENCERE), color='tab:blue', linewidth=2, label='ortalama')
plt.legend()
plt.title('Eğitim Toplam Kaybı')
plt.xlabel('Adım')
plt.ylabel('Kayıp')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASOR, 'egitim_kaybi.png'), dpi=120)
plt.close()

isimler = ['RPN nesnellik', 'RPN konum', 'Sınıflandırma', 'Kutu konum']
etiketler = [RPN_NESNE, RPN_KONUM, SINIF, KUTU]
plt.figure(figsize=(10, 7))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    x, y = sirali(egitim[etiketler[i]])
    plt.plot(x, y, color='tab:blue', alpha=0.3, linewidth=1)
    plt.plot(x, yumusat(y, PENCERE), color='tab:blue', linewidth=2)
    plt.title(isimler[i])
    plt.xlabel('Adım')
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASOR, 'kayip_bilesenleri.png'), dpi=120)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(map_adimlar, map_degerler, marker='o', color='tab:green', linewidth=2)
plt.scatter([map_adimlar[en_iyi]], [map_degerler[en_iyi]], s=160, facecolors='none', edgecolors='tab:red', linewidths=2)
plt.annotate('en iyi: {} adım, {:.3f}'.format(map_adimlar[en_iyi], map_degerler[en_iyi]),
             (map_adimlar[en_iyi], map_degerler[en_iyi]), textcoords='offset points', xytext=(-60, -30))
plt.title('Test Seti mAP@0.5')
plt.xlabel('Adım')
plt.ylabel('mAP@0.5')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASOR, 'map.png'), dpi=120)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(adimlar, yumusat(toplam, PENCERE), color='tab:blue', linewidth=2, label='Eğitim')
plt.plot(map_adimlar, test_toplam, marker='o', color='tab:orange', linewidth=2, label='Test')
plt.title('Eğitim ve Test Kaybı')
plt.xlabel('Adım')
plt.ylabel('Toplam kayıp')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASOR, 'egitim_test_kaybi.png'), dpi=120)
plt.close()

print('egitim kaybi: {} nokta, ilk {:.3f}, son {:.3f}'.format(len(adimlar), toplam[0], toplam[-1]))
for i in range(len(map_adimlar)):
    print('adim {}: mAP {:.4f}, test kaybi {:.4f}'.format(map_adimlar[i], map_degerler[i], test_toplam[i]))
print('en iyi checkpoint: {} ({:.4f})'.format(map_adimlar[en_iyi], map_degerler[en_iyi]))
print('grafikler: ' + GRAFIK_KLASOR)
