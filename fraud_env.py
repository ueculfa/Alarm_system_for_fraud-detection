import pandas as pd
import numpy as np

class FraudEnv:
    def __init__(self, csv_path="hesap_hareketleri_features.csv"):
        # Veriyi yükle
        self.df = pd.read_csv(csv_path)

        # Kategorik değişkenleri sayısala çevir
        self.df = self._encode_categories(self.df)

        # RL  kullanılacak başlıklar
        self.state_cols = [
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

        # fraud etiketleri
        self.labels = self.df['is_fraud'].values

        
        self.states = self.df[self.state_cols].fillna(0).values

        # Episode ayarları
        self.n_steps = len(self.df)
        self.current_step = 0


    def _encode_categories(self, df):
        """ txn_type ve location kolonlarını one-hot encode eder. Eksik olanları 0 ekler. """

        # One-hot encode işlemleri
        txn_dummies = pd.get_dummies(df['txn_type'], prefix='txn')
        loc_dummies = pd.get_dummies(df['location'], prefix='loc')

        df = pd.concat([df, txn_dummies, loc_dummies], axis=1)

        # Gerekli tüm kolonlar
        required_cols = [
            'txn_POS', 'txn_FAST_IN', 'txn_FAST_OUT',
            'txn_EFT', 'txn_KK_ODEME', 'txn_ATM', 'txn_OTHER',
            'loc_IST', 'loc_BUR', 'loc_UNKNOWN'
        ]

        # Eksik kolon varsa 0 yap
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        return df


    def reset(self):
        """Episode başlangıcı: ilk state döner"""
        self.current_step = 0
        return self.states[self.current_step]


    def step(self, action):
        """
        action: 0 = normal, 1 = alarm
        return: next_state, reward, done, info
        """

        fraud = self.labels[self.current_step]

        # Reward hesaplama
        if fraud == 1 and action == 1:
            reward = 20      # doğru alarm
        elif fraud == 1 and action == 0:
            reward = -30     # fraud kaçtı
        elif fraud == 0 and action == 1:
            reward = -10     # yanlış alarm
        else:  # fraud == 0 and action == 0
            reward = 1       # doğru normal

        # Bir sonraki adıma 
        self.current_step += 1

        done = self.current_step >= (self.n_steps - 1)

        next_state = None if done else self.states[self.current_step]

        info = {
            "fraud": int(fraud),
            "step": int(self.current_step)
        }

        return next_state, reward, done, info
