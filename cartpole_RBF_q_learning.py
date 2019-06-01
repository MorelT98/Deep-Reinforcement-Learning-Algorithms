import numpy as np
import matplotlib.pyplot as plt
import gym
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from q_learning_bins import plot_running_avg

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1

    def predict(self, X):
        return X.dot(self.w)

    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

class FeatureTransformer:
    def __init__(self, n_components = 1000):
        observation_examples = np.random.random((20000, 4)) * 2 - 1

        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=1, n_components=n_components)),
            ('rbf2', RBFSampler(gamma=0.5, n_components=n_components)),
            ('rbf3', RBFSampler(gamma=0.1, n_components=n_components)),
            ('rbf4', RBFSampler(gamma=0.05, n_components=n_components))
        ])

        transformed = featurizer.fit_transform(observation_examples)
        self.dimension = transformed.shape[1]

        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, s):
        scaled = self.scaler.transform(s)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimension)
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        return np.array([m.predict(X) for m in self.models])

    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, eps):
    observation = model.env.reset()
    done = False
    total_reward = 0
    iters = 0
    gamma = 0.99

    while not done:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = model.env.step(action)

        if done:
            reward = -200

        # Update
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G)

        if reward == 1:
            total_reward += reward
        iters += 1

    return total_reward

def main():
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    N = 500
    total_rewards = np.empty(N)

    for n in range(N):
        eps = 1 / np.sqrt(n + 1)
        total_reward = play_one(model, eps)
        total_rewards[n] = total_reward

        if (n + 1) % 100 == 0:
            print('episode:', (n + 1), 'total reward:', total_reward, 'eps:', eps, 'avg_reward (last 100):', total_rewards[max(0, n - 100):n+1].mean())

    print('avg reward for the last 100 episodes:', total_rewards[-100].mean())

    plt.plot(total_rewards)
    plt.title('Total Rewards')
    plt.show()

    plot_running_avg(total_rewards)

#main()