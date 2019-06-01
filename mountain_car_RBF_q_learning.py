import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from q_learning_bins import plot_running_avg

class FeatureTransformer:
    def __init__(self, env, n_components = 500):
        observation_examples = [env.observation_space.sample() for _ in range(10000)]
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to convert a state to featurized representation
        # We use RBF kernels with different variances to cover different parts
        # of the space
        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=n_components)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=n_components)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=n_components)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=n_components))
        ])
        featurized = featurizer.fit_transform(scaler.transform(observation_examples))

        self.scaler = scaler
        self.featurizer = featurizer
        self.dimension = featurized.shape[1]

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        assert len(X.shape) == 2
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)

def play_one(model, eps, gamma):
    observation = model.env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = model.env.step(action)

        # Update the model
        G = reward + gamma * np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G)

        totalreward += reward
        iters += 1

    return totalreward

def plot_cost_to_go(env, estimator, num_tiles = 20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title('Cost-To-Go Function')
    fig.colorbar(surf)
    plt.show()

def main(show_plots = True):
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    N = 300
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 0.1 * (0.97 ** n)
        total_reward = play_one(model, eps, gamma)
        total_rewards[n] = total_reward

        if n % 10 == 0:
            print('episode:', n, 'total reward:', total_reward)

    print('avg reward for last 100 episodes:', total_rewards[-100].mean())
    print('total steps:', -total_rewards.sum())

    if show_plots:
        plt.plot(total_rewards)
        plt.title('Rewards')
        plt.show()

        plot_running_avg(total_rewards)

        # Plot the optimal state-value function
        plot_cost_to_go(env, model)

#main()
