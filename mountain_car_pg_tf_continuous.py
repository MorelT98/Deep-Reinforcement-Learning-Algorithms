import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from mountain_car_RBF_q_learning import FeatureTransformer, plot_cost_to_go, plot_running_avg

class HiddenLayer:
    def __init__(self, M1, M2, f = tf.nn.tanh, use_bias = True, zeros = False):
        self.f = f
        self.use_bias = use_bias

        if zeros:
            self.W = tf.Variable(np.zeros((M1, M2)).astype(np.float32))
        else:
            self.W = tf.Variable(tf.random_normal(shape=(M1, M2)) / np.sqrt(M2))

        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))

    def forward(self, X):
        Z = tf.matmul(X, self.W)
        if self.use_bias:
            Z += self.b
        return self.f(Z)

class PolicyModel:
    def __init__(self, ft, D, hidden_layers_sizes = []):
        self.ft = ft

        # create layers
        self.layers = []
        M1 = D
        for M2 in hidden_layers_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # Output layer
        self.mean_layer = HiddenLayer(M1, 1, f=lambda x:x, use_bias=False)
        self.stdv_layer = HiddenLayer(M1, 1, f=tf.nn.softplus, use_bias=False)

        # Inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None,D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        # Build computational graph
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        mean = tf.reshape(self.mean_layer.forward(Z), [-1])
        stdv = tf.reshape(self.stdv_layer.forward(Z), [-1]) + 1e-6

        norm = tf.contrib.distributions.Normal(mean, stdv)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

        log_probs = norm.log_prob(self.actions)
        cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*norm.entropy())
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = self.ft.transform(np.atleast_2d(X))
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(self.train_op, feed_dict={
            self.X:X,
            self.actions:actions,
            self.advantages:advantages
        })

    def predict(self, X):
        X = self.ft.transform(np.atleast_2d(X))
        return self.session.run(self.predict_op, feed_dict={self.X:X})

    def sample_action(self, X):
        return self.predict(X)[0]

class ValueModel:
    def __init__(self, ft, D, hidden_layers_sizes = []):
        self.ft = ft
        self.costs = []

        # create layers
        self.layers = []
        M1 = D
        for M2 in hidden_layers_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # output layer
        layer = HiddenLayer(M1, 1, f=lambda x:x, use_bias=False)
        self.layers.append(layer)

        # Inputs and targets
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None,), name='Y')

        # Computational graph
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1])
        self.predict_op = Y_hat

        self.cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.train_op = tf.train.AdamOptimizer(1e-1).minimize(self.cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = self.ft.transform(np.atleast_2d(X))
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X:X, self.Y:Y})

        # save costs for debugging purposes
        cost = self.session.run(self.cost, feed_dict={self.X:X, self.Y:Y})
        self.costs.append(cost)

    def predict(self, X):
        X = self.ft.transform(np.atleast_2d(X))
        return self.session.run(self.predict_op, feed_dict={self.X:X})

def play_one_td(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    iters = 0
    total_reward = 0

    while not done:
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step([action])
        total_reward += reward

        # update the model
        G = reward + gamma * vmodel.predict(observation)
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)

        iters += 1

    return total_reward, iters

def main():
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env, n_components=100)
    D = ft.dimension
    pmodel = PolicyModel(ft, D, [10])
    vmodel = ValueModel(ft, D, [10])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.95

    N = 100
    total_rewards = np.empty(N)
    for n in range(N):
        total_reward, steps = play_one_td(env, pmodel, vmodel, gamma)
        total_rewards[n] = total_reward
        print(n, 'total reward:', total_reward, 'steps:', steps, 'avg reward so far:', total_rewards[:n+1].mean())

    print('avg total reward:', total_rewards.mean())

    plt.plot(vmodel.costs)
    plt.title('Value Model Cost Function')
    plt.show()

    plt.plot(total_rewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(total_rewards)

    plot_cost_to_go(env, vmodel)

main()
