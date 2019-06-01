import gym
import numpy as np
import matplotlib.pyplot as plt
import mountain_car_RBF_q_learning
from mountain_car_RBF_q_learning import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg

class SGDRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 10e-3

    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.random(D) / np.sqrt(D)
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

mountain_car_RBF_q_learning.SGDRegressor = SGDRegressor

def play_one(model, eps, gamma, n=5):
    observation = model.env.reset()
    done = False
    iters = 0
    total_reward = 0
    states = []
    actions = []
    rewards = []
    # array of [gamma^0, gamma^1, ..., gamma^(n-1)]
    multiplier = np.array([gamma] * n) ** np.arange(n)

    while not done:
        action = model.sample_action(observation, eps)

        states.append(observation)
        actions.append(action)

        observation, reward, done, info = model.env.step(action)

        rewards.append(reward)

        if len(states) >= n:
            # get R(t + 1) + gamma * R(t + 2) + ... + gamma^(n-1) * R(t + n)
            last_n_weighted_rewards_sum = multiplier.dot(rewards[-n:])
            # Add gamma^n * V(s(t + n))
            G = last_n_weighted_rewards_sum + (gamma ** n) * np.max(model.predict(observation)[0])
            # Update model
            model.update(states[-n], actions[-n], G)

        iters += 1
        total_reward += reward

    # empty the cach
    rewards = rewards[-n+1:]
    states = states[-n+1:]
    actions = actions[-n+1:]

    # According to documentation, we reach the goal when
    # the position >= 0.5
    # if observation[0] >= 0.5:
    #     while len(rewards) > 0:
    #         G = multiplier[:len(rewards)].dot(rewards)
    #         model.update(states[0], actions[0], G)
    #         rewards.pop(0)
    #         states.pop(0)
    #         actions.pop(0)
    # else:
    #     # If we didn't make it, we'll just
    #     # assume that the remaining rewards were all -1
    #     while len(rewards) > 0:
    #         guess_rewards = rewards + [-1] * (n - len(rewards))
    #         G = multiplier.dot(guess_rewards)
    #         model.update(states[0], actions[0], G)
    #         rewards.pop(0)
    #         states.pop(0)
    #         actions.pop(0)

    return total_reward

def main():
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    N = 300
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 1 / np.sqrt(n + 1)
        total_reward = play_one(model, eps, gamma)
        total_rewards[n] = total_reward
        print('episode:', n, 'total reward:', total_reward)
    print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
    print('total steps:', -total_rewards.sum())

    plt.plot(total_rewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(total_rewards)

    plot_cost_to_go(env, model)

#main()




