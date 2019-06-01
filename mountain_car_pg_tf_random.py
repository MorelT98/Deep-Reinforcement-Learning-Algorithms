import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mountain_car_RBF_q_learning import FeatureTransformer

class HiddenLayer:
    def __init__(self, M1, M2, f = tf.nn.tanh, use_bias = True, zeros = False):
        if zeros:
            W = np.zeros((M1, M2)).astype(np.float32)
            self.W = tf.Variable(W)
        else:
            self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))

        self.params = [self.W]

        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)


# approximates pi(a | s)
class PolicyModel:
    def __init__(self, ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_std=[]):

        # save inputs for copy
        self.ft = ft
        self.D = D
        self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
        self.hidden_layer_sizes_std = hidden_layer_sizes_std

        ### model the mean ###
        self.mean_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_mean:
            layer = HiddenLayer(M1, M2)
            self.mean_layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, lambda x:x, use_bias=False, zeros=True)
        self.mean_layers.append(layer)

        ### model the standard deviation ###
        self.std_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_std:
            layer = HiddenLayer(M1, M2)
            self.std_layers.append(layer)

        # final layer
        layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
        self.std_layers.append(layer)

        # gather params
        self.params = []
        for layer in (self.mean_layers + self.std_layers):
            self.params += layer.params

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name = 'X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name ='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        def get_ouput(layers):
            Z = self.X
            for layer in layers:
                Z = layer.forward(Z)
            return tf.reshape(Z, [-1])

        # calculate output and cost
        mean = get_ouput(self.mean_layers)
        std = get_ouput(self.std_layers) + 1e-4 # smoothing
        norm = tf.contrib.distributions.Normal(mean, std)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

    def set_session(self, session):
        self.session = session

    # Since we need to save the best params but initialize the
    # current params we're testing, we no longer can use
    # global_variables_initializer
    def init_vars(self):
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return p

    def copy_from(self, other):
        # Collect all the operations
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        # Now run them all
        self.session.run(ops)

    def copy(self):
        # Create a new model with the same constants
        clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_std)
        clone.set_session(self.session)
        clone.init_vars()
        # Now copy all the variables
        clone.copy_from(self)
        return clone

    def perturb_params(self):
        ops = []
        for p in self.params:
            v = self.session.run(p)
            noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
            # With a 0.1 probability, start completely from scratch
            if np.random.random() < 0.1:
                op = p.assign(noise)
            else:
                op = p.assign(v + noise)
            ops.append(op)
        self.session.run(ops)

def play_one(env, pmodel):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0

    while not done:
        action = pmodel.sample_action(observation)
        # oddly the mountain car environment requires the action to be in
        # an object where the actual action is stored in object[0]
        observation, reward, done, info = env.step([action])

        total_reward += reward
        iters += 1
    return total_reward

def play_multiple_episodes(env, T, pmodel, print_iters=False):
    total_rewards = np.empty(T)

    for t in range(T):
        total_rewards[t] = play_one(env, pmodel)

        if print_iters:
            print(t, 'avg so far:', total_rewards[:t+1].mean())

    avg_total_rewards = total_rewards.mean()
    print('avg total rewards:', avg_total_rewards)
    return avg_total_rewards

def random_search(env, pmodel):
    total_rewards = []
    best_avg_total_reward = float('-inf')
    best_pmodel = pmodel
    num_episodes_per_param_test = 3
    for t in range(100):
        tmp_model = best_pmodel.copy()
        tmp_model.perturb_params()

        avg_total_rewards = play_multiple_episodes(env, num_episodes_per_param_test, tmp_model)
        total_rewards.append(avg_total_rewards)

        if avg_total_rewards > best_avg_total_reward:
            best_pmodel = tmp_model
            best_avg_total_reward = avg_total_rewards

    return total_rewards, best_pmodel

def main():
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env, n_components=100)
    D = ft.dimension
    pmodel = PolicyModel(ft, D, [], [])
    session = tf.InteractiveSession()
    pmodel.set_session(session)
    pmodel.init_vars()

    total_rewards, pmodel = random_search(env, pmodel)
    print('max reward:', np.max(total_rewards))

    # play 100 episodes and check the average
    avg_totalrewards = play_multiple_episodes(env, 100, pmodel, print_iters=True)
    print('avg reward over 100 episodes with best models:', avg_totalrewards)

    plt.plot(total_rewards)
    plt.title('Rewards')
    plt.show()

main()



