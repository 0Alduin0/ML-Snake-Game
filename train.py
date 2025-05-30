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
    record = 0
    agent = Agent()

    # MODEL DOSYASI YOLU - hem load hem save aynı olsun
    model_path = './model/model.pth'

    # Model klasörü yoksa oluştur
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Daha önce kayıtlı model varsa yükle
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Model başarıyla yüklendi, eğitim kaldığı yerden devam edecek.")
    else:
        print("Kayıtlı model bulunamadı, yeni eğitim başlayacak.")

    game = SnakeGameAI()

    while True:
        # 1. Durum bilgisi al
        state_old = agent.get_state(game)

        # 2. Hareket seç (model ya da rastgele)
        final_move = agent.get_action(state_old)

        # 3. Adımı uygula (oyun ortamında)
        reward, done, score = game.play_step(final_move)

        # 4. Yeni durumu al
        state_new = agent.get_state(game)

        # 5. Kısa dönem öğrenme
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 6. Hafızaya kaydet (uzun dönem için)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Oyun bittiğinde reset ve uzun dönem öğrenme
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Yeni skor, rekoru geçerse modeli kaydet
            if score > record:
                record = score
                agent.save(model_path)
                print("Yeni rekor! Model kaydedildi.")

            print('Oyun:', agent.n_games, 'Skor:', score, 'Rekor:', record)

            # Skorları kaydet ve ortalamayı hesapla
            scores.append(score)
            total_score += score
            mean_scores.append(total_score / agent.n_games)

            # Her 50 oyunda bir grafik çiz
            if agent.n_games % 50 == 0:
                plot(scores, mean_scores)


if __name__ == '__main__':
    train()
