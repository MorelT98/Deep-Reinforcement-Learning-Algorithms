import gym
import numpy as np
import matplotlib.pyplot as plt
ALPHA = 10e-3


# turns a list of integers into an int
# Ex: build_state([1, 2, 3, 4, 5]) = 12345
def build_state(features):
    return int(''.join(map(lambda feature:str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x = [value], bins = bins)[0]

class FeatureTransformer:
    def __init__(self):
        # Note: to make this better you could look at how often each bin was
        # actually used while running the script
        # It's not clear from the high/low values nor sample() what values
        # we really expect to get
        self.cart_pos_bins = np.linspace(-2.4, 2, 4, 9)
        self.cart_vel_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_vel_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return build_state([
            to_bin(cart_pos, self.cart_pos_bins),
            to_bin(cart_vel, self.cart_vel_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.cart_vel_bins)
        ])


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        num_states = 10 ** env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

    def predict(self, s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]

    def update(self, s, a, G):
        x = self.feature_transformer.transform(s)
        self.Q[x, a] += ALPHA * (G - self.Q[x, a])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)

def play_one(model, eps, gamma):
    observation = model.env.reset()
    done = False
    total_reward = 0
    iters = 0
    while not done:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = model.env.step(action)

        total_reward += reward

        if done and iters < 199:
            reward = -300

        # Update the model
        # Q learning equation
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G)

        iters += 1

    return total_reward

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t - 100): t + 1].mean()
    plt.plot(running_avg)
    plt.title('Running Average')
    plt.show()

def main():
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9

    N = 5000
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        total_reward = play_one(model, eps, gamma)
        total_rewards[n] = total_reward
        if n % 100 == 0:
            print('episode:', n, 'total reward:', total_reward, 'eps:', eps)

    print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
    print('total steps:', total_rewards.sum())

    plt.plot(total_rewards)
    plt.title('Rewards')
    plt.show()

    # Since the total rewards varies a lot, it's better to
    # plot the running avg
    plot_running_avg(total_rewards)

#main()