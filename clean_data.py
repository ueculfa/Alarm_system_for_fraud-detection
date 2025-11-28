import pandas as pd
import numpy as np

# ilk veri
file_path = "hesap_hareketleri_raw.csv"

df = pd.read_csv(
    file_path,
    sep=';',           
    encoding='latin1', 
    header=None        
)

print("İlk birkaç satır (ham):")
print(df.head())
print("\nToplam kolon sayısı:", len(df.columns))

df = df.iloc[:, 0:4]
df.columns = ['islem_tarihi', 'aciklama', 'islem_tutari', 'yeni_bakiye']

# Kullanılmayan satırları sil
df = df.dropna(how='all')

print("\nKolon isimleri düzeltildikten sonra (ham):")
print(df.head())

mask_tarih = df['islem_tarihi'].astype(str).str.contains(r'\d')
df = df[mask_tarih].reset_index(drop=True)

print("\nBaşlık benzeri satırlar atıldıktan sonra ilk 5 satır:")
print(df.head())

# işlem tarihini datetime a çevrildi
df['islem_tarihi'] = pd.to_datetime(
    df['islem_tarihi'],
    format='%d.%m.%Y',
    errors='coerce'
)

def turkish_to_float(x):
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    s = s.replace('.', '')            
    s = s.replace(',', '.')             
    s = s.replace('TL', '').replace('₺', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

df['islem_tutari'] = df['islem_tutari'].apply(turkish_to_float)
df['yeni_bakiye'] = df['yeni_bakiye'].apply(turkish_to_float)

# Tarihe göre sırala
df = df.sort_values('islem_tarihi').reset_index(drop=True)

print("\nTipler:")
print(df.dtypes)
print("\nBoş değer sayıları:")
print(df.isnull().sum())
print("\nSon haliyle ilk 10 satır:")
print(df.head(10))

# 9) clean data olarak kaydet
output_path = "hesap_hareketleri_clean.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Temiz veri '{output_path}' dosyasına kaydedildi.")
