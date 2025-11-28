from fraud_env import FraudEnv
import numpy as np

env = FraudEnv("hesap_hareketleri_features.csv")

state = env.reset()
total_reward = 0
step_count = 0

while True:
    # %50 ihtimalle alarm, %50 normal
    action = np.random.randint(0, 2)

    next_state, reward, done, info = env.step(action)

    total_reward += reward
    step_count += 1

    print(f"Step: {step_count}, Action: {action}, Fraud: {info['fraud']}, Reward: {reward}")

    if done:
        break

print("\nToplam adım:", step_count)
print("Toplam reward:", total_reward)
