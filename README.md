# DOLANDIRCILIK TESPİTİNDE ALARM VEREN SİSTEM

Bu proje, gerçek banka hesap hareketlerinden oluşturulan bir veri seti üzerinde dolandırıcılık Tespiti (Fraud Detection) yapmak amacıyla geliştirilmiş bir Pekiştirmeli Öğrenme (Reinforcement Learning) ve Denetimli Öğrenme (Supervised Learning) tabanlı makine öğrenimi ile yapılan bir sistemdir.

## Projenin Amacı
Gerçek hayattaki banka işlemlerine benzer bir akış üzerinde:
- Her işlemde alarm ver / verme kararı üretmek.
- Fraud tespit edilirse ödül kazanmak / Fraud kaçırıldığında büyük ceza almak.
- Yanlış alarmlarda ceza almak.
- Böylece fraud tespitini optimize eden bir politika (policy) üretmek.

##  Proje Mimarisi
Ana aşamalar:
1) Veri temizleme
2) Feature engineering
3) Reinforcement Learning ortamı oluşturma
4) Random policy testi
5) Logistic Regression model eğitimi
6) Supervised policy testi
7) Grafik ve Q-table üretimi

## 🔬 Veri Temizleme Kısmında Yapılan Değişiklikler
- UTF-8 encoding düzeltildi.
- Tarih → datetime olarak yapıldı.
- Tutar → float olarak çevirildi.
- Başlık ve bozuk satırlar temizlendi.

Çıktı: **hesap_hareketleri_clean.csv**

### Action:
- 0 → normal
- 1 → alarm

### Reward Fonksiyonu:
| Fraud | Alarm | Reward |
|-------|--------|--------|
| 1     | 1      | +20    |
| 1     | 0      | -30    |
| 0     | 1      | -10    |
| 0     | 0      | +1     |

## Random Policy Sonucu
Toplam reward ≈ -430

## Logistic Regression Policy
- `train_baseline.py` ile eğitildi.
- `eval_baseline_policy.py` ile RL ortamında test edildi.
- Toplam reward ≈ +1084

## Sonuç Grafik ve Tablo
- `OdulAdımsal.png`
- `OdulKumulatif.png`
- `rewardOrnek.png`
- `OdulTablo.csv`


