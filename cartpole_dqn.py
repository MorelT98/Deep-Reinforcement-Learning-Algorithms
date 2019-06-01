import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from cartpole_RBF_q_learning import plot_running_avg

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W = tf.Variable(tf.random_normal((M1, M2)))
        self.params = [self.W]
        self.f = f
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)

    def forward(self, X):
        if self.use_bias:
            Z = tf.matmul(X, self.W) + self.b
        else:
            Z = tf.matmul(X, self.W)
        return self.f(Z)

class DQN:
    def __init__(self, D, K, hidden_layers_sizes, gamma, max_experieces = 10000, min_experiences=100, batch_sz=32):
        self.K = K
        self.gamma = gamma
        self.max_experiences = max_experieces
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz

        # Create layers
        self.layers = []
        M1 = D
        for M2 in hidden_layers_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        # Output layer
        layer = HiddenLayer(M1, K, f=lambda x:x)
        self.layers.append(layer)

        # Save params
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # Inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # Computational graph
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        selected_actions_values = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions, K),
            reduction_indices=[1]
        )
        cost = tf.reduce_sum(tf.square(self.G - selected_actions_values))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

        # Replay memory
        self.experiences = {'s':[], 'a':[], 'r':[], 's2':[], 'done':[]}

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        my_params = self.params
        other_params = other.params
        ops = []
        # Collect all assignment operations
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        # Now run them all
        self.session.run(ops)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})

    def train(self, target_network):
        if len(self.experiences['s']) < self.min_experiences:
            return

        # Sample a random batch from experiences
        idx = np.random.choice(len(self.experiences['s']), size=self.batch_sz, replace=True)
        states = [self.experiences['s'][i] for i in idx]
        actions = [self.experiences['a'][i] for i in idx]
        rewards = [self.experiences['r'][i] for i in idx]
        next_states = [self.experiences['s2'][i] for i in idx]
        dones = [self.experiences['done'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self.gamma * next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        # Call optimizer
        self.session.run(self.train_op, feed_dict={
            self.X:states,
            self.G:targets,
            self.actions:actions
        })

    def add_experience(self, s, a, r, s2, done):
        # Create space if necessary
        if len(self.experiences['s']) >= self.max_experiences:
            self.experiences['s'].pop(0)
            self.experiences['a'].pop(0)
            self.experiences['r'].pop(0)
            self.experiences['s2'].pop(0)
            self.experiences['done'].pop(0)
        # Add new experience
        self.experiences['s'].append(s)
        self.experiences['a'].append(a)
        self.experiences['r'].append(r)
        self.experiences['s2'].append(s2)
        self.experiences['done'].append(done)

    def sample_action(self, X, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(X)
            return np.argmax(self.predict(X)[0])

def play_one(env, model, tmodel, eps, gamma, copy_period):
    observation = env.reset()
    done = False
    iters = 0
    total_reward = 0
    while not done:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            reward = -200

        # Update model
        model.add_experience(prev_observation, action, reward, observation, done)
        model.train(tmodel)

        # Copy the params
        if iters % copy_period == 0:
            tmodel.copy_from(model)

        iters += 1

    return total_reward, iters

def main():
    env = gym.make('CartPole-v0')
    D = len(env.observation_space.sample())
    K = env.action_space.n
    sizes = [200, 200]
    gamma = 0.95
    copy_period = 50

    model = DQN(D, K, sizes, gamma)
    tmodel = DQN(D, K, sizes, gamma)
    session = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    session.run(init)
    model.set_session(session)
    tmodel.set_session(session)

    N = 300
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        total_reward, iters = play_one(env, model, tmodel, eps, gamma, copy_period)
        total_rewards[n] = total_reward

        if n % 50 == 0:
            print("episode:", n, 'total reward:', total_reward, 'eps:', eps, 'avg reward (last 50):', total_rewards[max(0, n-50):n+1].mean())
    print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
    print('total steps:', total_rewards.sum())

    plt.plot(total_rewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(total_rewards)

main()