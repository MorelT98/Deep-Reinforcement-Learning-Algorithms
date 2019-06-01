import gym
import numpy as np
import matplotlib.pyplot as plt

def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params, render = False):
    observation = env.reset()
    done = False
    t = 0

    while not done:
        t += 1
        if render:
            env.render()
        action = get_action(observation, params)
        observation, rewards, done, info = env.step(action)

    return t

def play_multiple_episodes(env, T, params):
    episode_lengths = np.zeros(T)

    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)

    avg_length = episode_lengths.mean()
    print('avg length', avg_length)
    return avg_length

def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4) * 2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)

        if avg_length > best:
            best = avg_length
            params = new_params

        episode_lengths.append(avg_length)

    plt.plot(episode_lengths)
    plt.title("Episode Lengths")
    plt.show()
    return params

env = gym.make('CartPole-v0')
params = random_search(env)
steps = play_one_episode(env, params)
print('best steps achieved:', steps)

