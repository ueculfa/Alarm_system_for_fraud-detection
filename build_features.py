import pandas as pd
import numpy as np

# Verilerin temizlenmiş hali
df = pd.read_csv("hesap_hareketleri_clean.csv")

print("Temiz veri yüklendi. İlk satırlar:")
print(df.head())

#zaman
df['hour'] = pd.to_datetime(df['islem_tarihi']).dt.hour
df['day_of_week'] = pd.to_datetime(df['islem_tarihi']).dt.weekday
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# para işlemi
df['direction'] = df['islem_tutari'].apply(lambda x: 1 if x > 0 else -1)
df['amount_log'] = np.log1p(np.abs(df['islem_tutari']))

# 7 günlük hareket ortalama ve standart sapma
df['amount_mean_7d'] = df['islem_tutari'].rolling(20).mean()
df['amount_std_7d'] = df['islem_tutari'].rolling(20).std()
df['is_amount_outlier'] = (
    np.abs(df['islem_tutari']) > df['amount_mean_7d'] + 2 * df['amount_std_7d']
).astype(int)

# işlem türü
def get_txn_type(desc):
    d = str(desc).upper()
    if "ALIŞVER" in d or "POS" in d:
        return "POS"
    if "GELEN FAST" in d or "GELEN" in d:
        return "FAST_IN"
    if "GIDEN FAST" in d or "GÖNDER" in d:
        return "FAST_OUT"
    if "HAVALE" in d or "EFT" in d:
        return "EFT"
    if "KREDI" in d or "KART" in d:
        return "KK_ODEME"
    if "ATM" in d:
        return "ATM"
    return "OTHER"

df['txn_type'] = df['aciklama'].apply(get_txn_type)

# konum
def extract_location(desc):
    d = str(desc).upper()
    if "IST" in d:
        return "IST"
    if "BUR" in d:
        return "BUR"
    return "UNKNOWN"

df['location'] = df['aciklama'].apply(extract_location)
df['location_changed'] = (df['location'] != df['location'].shift(1)).astype(int)

# 10 işlem oltalama gibi 24 saatlik tutma
df['txn_count_24h'] = df['islem_tutari'].rolling(10).count()
df['sum_amount_24h'] = df['islem_tutari'].rolling(10).sum()
df['avg_amount_24h'] = df['islem_tutari'].rolling(10).mean()

# sentetik fraud skoru üretme
def fraud_score(row):
    score = 0
    if row['is_weekend'] == 1:
        score += 1
    if row['is_amount_outlier'] == 1:
        score += 2
    if row['txn_type'] in ["FAST_OUT", "ATM"]:
        score += 1
    if row['location_changed'] == 1:
        score += 2
    if row['txn_count_24h'] >= 5:
        score += 3
    return score

df['risk_score'] = df.apply(fraud_score, axis=1)
df['is_fraud'] = (df['risk_score'] >= 4).astype(int)

# yeni veri seti
output_path = "hesap_hareketleri_features.csv"
df.to_csv(output_path, index=False)

print("\n🎉 Feature'lı veri kaydedildi:", output_path)
print("İlk 10 satır:")
print(df.head(10))
