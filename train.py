import torch
from agent import Agent
from game import SnakeGameAI
import matplotlib.pyplot as plt

def plot(scores, mean_scores):
    plt.figure(figsize=(10,5))
    plt.title('Yılan Oyunu - Skor Grafiği')
    plt.xlabel('Oyun Sayısı')
    plt.ylabel('Skor')
    plt.plot(scores, label='Skor')
    plt.plot(mean_scores, label='Ortalama Skor')
    plt.legend()
    plt.show()

def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Durum bilgisi al
        state_old = agent.get_state(game)

        # Hareket seç
        final_move = agent.get_action(state_old)

        # Adımı uygula ve sonuç al
        reward, done, score = game.play_step(final_move)

        # Yeni durum bilgisi
        state_new = agent.get_state(game)

        # Kısa dönem öğren
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Hafızaya kaydet
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Oyun bittiğinde
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Oyun:', agent.n_games, 'Skor:', score, 'Rekor:', record)

            scores.append(score)
            total_score += score
            mean_scores.append(total_score / agent.n_games)

            # Grafik çiz (isteğe bağlı)
            if agent.n_games % 10 == 0:
                plot(scores, mean_scores)

if __name__ == '__main__':
    train()
