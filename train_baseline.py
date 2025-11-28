import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # modeli kaydetmek için

# 1) Featurelı veri
df = pd.read_csv("hesap_hareketleri_features.csv")

txn_dummies = pd.get_dummies(df['txn_type'], prefix='txn')
loc_dummies = pd.get_dummies(df['location'], prefix='loc')

df = pd.concat([df, txn_dummies, loc_dummies], axis=1)

required_txn_cols = ['txn_POS', 'txn_FAST_IN', 'txn_FAST_OUT',
                     'txn_EFT', 'txn_KK_ODEME', 'txn_ATM', 'txn_OTHER']
required_loc_cols = ['loc_IST', 'loc_BUR', 'loc_UNKNOWN']

for col in required_txn_cols + required_loc_cols:
    if col not in df.columns:
        df[col] = 0

# RL için kullanılacak başlıklar
state_cols = [
    'hour',
    'day_of_week',
    'is_weekend',
    'direction',
    'amount_log',
    'is_amount_outlier',
    'txn_count_24h',
    'sum_amount_24h',
    'avg_amount_24h',
    'location_changed',

    'txn_POS',
    'txn_FAST_IN',
    'txn_FAST_OUT',
    'txn_EFT',
    'txn_KK_ODEME',
    'txn_ATM',
    'txn_OTHER',

    'loc_IST',
    'loc_BUR',
    'loc_UNKNOWN'
]

X = df[state_cols].fillna(0).values
y = df['is_fraud'].values

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Pipeline oluştur
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# eğitim
clf.fit(X_train, y_train)

# test sonuç
y_pred = clf.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# kayıt
joblib.dump(clf, "baseline_fraud_model.joblib")
print("\n✅ Model 'baseline_fraud_model.joblib' olarak kaydedildi.")
