import torch
from agent import Agent
from game import SnakeGameAI
import matplotlib.pyplot as plt
import os


# Skor grafiğini çizmek için
def plot(scores, mean_scores):
    plt.figure(figsize=(10, 5))
    plt.title('Yılan Oyunu - Skor Grafiği')
    plt.xlabel('Oyun Sayısı')
    plt.ylabel('Skor')
    plt.plot(scores, label='Skor')
    plt.plot(mean_scores, label='Ortalama Skor')
    plt.legend()
    plt.grid()
    plt.show()


def train():
    scores = []
    mean_scores = []
    total_score = 0

    agent = Agent()
    game = SnakeGameAI()

    model_path = './model/model.pth'
    record_path = 'record.txt'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Rekoru yükle
    if os.path.exists(record_path):
        with open(record_path, "r") as f:
            try:
                record = int(f.read().strip())
            except:
                record = 0
    else:
        record = 0

    # Önceki model varsa yükle
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"Önceki model yüklendi. Başlangıç rekoru: {record}")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
    else:
        print("Eğitime sıfırdan başlandı.")

    # Ana oyun döngüsü
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.save(model_path)
                with open(record_path, "w") as f:
                    f.write(str(record))
                print(f"🏆 Yeni rekor: {record} — Model kaydedildi!")

            print(f"Oyun: {agent.n_games} | Skor: {score} | Rekor: {record}")

            scores.append(score)
            total_score += score
            mean_scores.append(total_score / agent.n_games)

            # Her 50 oyunda bir grafik göster
            #if agent.n_games % 50 == 0:
            #    plot(scores, mean_scores)


if __name__ == '__main__':
    train()
