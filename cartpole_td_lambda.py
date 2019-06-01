import gym
import numpy as np
import matplotlib.pyplot as plt
from cartpole_RBF_q_learning import FeatureTransformer
from mountain_car_RBF_q_learning import plot_running_avg, plot_cost_to_go

class BaseModel:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, x, y, e, lr = 0.1):
        self.w += lr * (y - x.dot(self.w)) * e

    def predict(self, X):
        return X.dot(self.w)

class Model:
    def __init__(self, env, feature_transformer):
        self.env  = env
        self.feature_transformer = feature_transformer
        self.models = []

        D = feature_transformer.dimension
        for i in range(env.action_space.n):
            model = BaseModel(D)
            self.models.append(model)

        self.eligibilities = np.zeros((env.action_space.n, D))

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        return np.array([m.predict(X) for m in self.models])

    def update(self, s, a, G, gamma, lambda_):
        X = self.feature_transformer.transform([s])
        self.eligibilities *= gamma * lambda_
        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X, G, self.eligibilities[a])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

    def reset(self):
        self.eligibilities = np.zeros_like(self.eligibilities)


def play_one(model, eps, gamma, lambda_):
    observation = model.env.reset()
    done = False
    iters = 0
    total_reward = 0
    model.reset()

    while not done:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = model.env.step(action)

        if done:
            reward = -200

        # Update the model
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G, gamma, lambda_)


        iters += 1
        if reward == 1:
            total_reward += reward

    return total_reward

def main():
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.999
    lambda_ = 0.7

    N = 500
    total_rewards = np.empty(N)

    for n in range(N):
        eps = 1 / np.sqrt(n + 1)
        total_reward = play_one(model, eps, gamma, lambda_)
        total_rewards[n] = total_reward

        if (n + 1) % 50 == 0:
            print('episode:', (n + 1), 'total reward:', total_reward, 'avg reward (last 50):', total_rewards[max(0, n -49):n + 1].mean())

    print('avg reward for last 100 episodes:', total_rewards[-100].mean())

    plt.plot(total_rewards)
    plt.title('Total Rewards')
    plt.show()

    plot_running_avg(total_rewards)

#main()