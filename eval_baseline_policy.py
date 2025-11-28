import numpy as np
import joblib
from fraud_env import FraudEnv

# 1) Eğittiğimiz modeli yükle
clf = joblib.load("baseline_fraud_model.joblib")

# 2) Environmenti başlat
env = FraudEnv("hesap_hareketleri_features.csv")

state = env.reset()
total_reward = 0
step_count = 0

while True:
    
    state_input = state.reshape(1, -1)


    fraud_proba = clf.predict_proba(state_input)[0, 1]

    # Kontrol
    threshold = 0.5
    action = 1 if fraud_proba > threshold else 0

    # Adım yap
    next_state, reward, done, info = env.step(action)

    total_reward += reward
    step_count += 1

    print(
        f"Step: {step_count}, P(fraud)={fraud_proba:.3f}, "
        f"Action: {action}, Fraud: {info['fraud']}, Reward: {reward}"
    )

    if done:
        break

    state = next_state

print("\nToplam adım:", step_count)
print("Toplam reward:", total_reward)
