import numpy as np
import tensorflow as tf
import collections
import tqdm
import statistics
import gym
import env
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from MCTS import Node, State, monte_carlo_tree_search

env = gym.make("video_player-v0")
N = 1


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=1):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


set_global_determinism(1)

class DQN(tf.keras.Model):

    def __init__(self, a_num):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=int(16 * N), activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=int(32 * N), activation='relu')
        self.dnn1 = tf.keras.layers.Dense(units=int(128 * N), activation='relu')
        self.dr1 = tf.keras.layers.Dropout(0.2)
        self.dr2 = tf.keras.layers.Dropout(0.2)
        self.dnn2 = tf.keras.layers.Dense(units=int(128 * N), activation='relu')
        self.dnn3 = tf.keras.layers.Dense(units=int(64 * N), activation='relu')
        self.con1d = tf.keras.layers.Conv1D(int(32 * N), 3, activation='relu')
        self.con1d_buffer = tf.keras.layers.Conv1D(int(16 * N), 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.flatten_buffer = tf.keras.layers.Flatten()
        self.dense_a = tf.keras.layers.Dense(units=a_num)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, input):
        con1d_out = self.flatten(self.con1d(input[0]))
        buffer_feat = self.dense1(tf.reshape(input[1][0][5][0], [1, 1]))
        buffer_seq_feat = self.flatten_buffer(self.con1d_buffer(input[1]))
        in_merge = tf.concat([con1d_out, buffer_feat, buffer_seq_feat], 1)
        feat0 = self.dnn3(self.dr2(self.dnn2(self.dr1(self.dnn1(in_merge)))))
        Q = self.dense_a(feat0)
        return Q

    def predict(self, input):
        out_Q = self(input)
        return tf.argmax(out_Q, axis=1)

def generate_sequence(cur_seq, cur_item, num_t):
    if (len(cur_seq) == num_t):
        for i in range(1, num_t):
            cur_seq[i - 1] = cur_seq[i]
        cur_seq[num_t - 1] = cur_item
        return cur_seq, True
    elif ((len(cur_seq) == num_t - 1)):
        cur_seq.append(cur_item)
        return cur_seq, True
    else:
        cur_seq.append(cur_item)
        return cur_seq, False


def state_to_tensor(initial_non_ten, bs_seq, buffer_seq, seq_num=8):
    last_bit_rate_idx = initial_non_ten[4]
    if (last_bit_rate_idx < 5 and last_bit_rate_idx > 0):
        state_temp1 = tf.constant(initial_non_ten[1][last_bit_rate_idx - 1:last_bit_rate_idx + 2], dtype=tf.float32)
    elif (last_bit_rate_idx == 5):
        temp = []
        temp.append(initial_non_ten[1][4])
        temp.append(initial_non_ten[1][5])
        temp.append(initial_non_ten[1][5])
        state_temp1 = tf.constant(temp, dtype=tf.float32)
    elif (last_bit_rate_idx == 0):
        temp = []
        temp.append(initial_non_ten[1][0])
        temp.append(initial_non_ten[1][0])
        temp.append(initial_non_ten[1][1])
        state_temp1 = tf.constant(temp, dtype=tf.float32)
    bs_seq, OK_for_training = generate_sequence(bs_seq, initial_non_ten[0][0], seq_num)
    throughput = [i / 8 * 0.95 for i in bs_seq]
    buffer_seq, _ = generate_sequence(buffer_seq, initial_non_ten[2], seq_num)

    if (OK_for_training):
        temp = [1 / i for i in throughput]
        temp = tf.constant(temp, dtype=tf.float32)
        temp = tf.reshape(temp, [seq_num, 1])
        state_temp1 = temp * tf.reshape(state_temp1, [1, 3])
        state_temp2 = tf.constant(buffer_seq, dtype=tf.float32)
        state_temp2 = tf.reshape(state_temp2, [1, seq_num, 1])
        state = [tf.reshape(state_temp1, [1, seq_num, 3]), state_temp2]
        return state, bs_seq, buffer_seq, True
    else:
        state_temp1 = state_temp1 / throughput[-1]
        state_temp2 = tf.constant(initial_non_ten[2], dtype=tf.float32)
        state_temp2 = tf.reshape(state_temp2, [1, 1])
        state = [state_temp1, state_temp2]
        return state, bs_seq, buffer_seq, False


def run_episode(initial_state, bs_seq, buffer_seq, dqn: tf.keras.Model, max_steps: int, train, seq_num):
    Q_values = []
    target_values = []
    rewards = []
    state = initial_state
    chunk_num = 0
    OK_for_training = 0
    for t in tf.range(max_steps):
        if (OK_for_training):
            Q_logits = dqn(state)
            if (chunk_num % 40 == 0 and train):
                Q_values.append(Q_logits)
                root = Node()
                state_mcts = State(0, env, 0)
                root.set_state(state_mcts)
                root_value = monte_carlo_tree_search(root)
                target_values.append(root_value)
            action = tf.argmax(Q_logits, axis=1)
            state, reward, done, _ = env.step(action.numpy()[0] - 1)
            rewards.append(reward)
            state, bs_seq, buffer_seq, OK_for_training = state_to_tensor(state, bs_seq, buffer_seq, seq_num)
            reward = tf.constant(reward, dtype=tf.float32)
            chunk_num = chunk_num + 1
            if done:
                break
        else:
            action = 1
            state, reward, done, _ = env.step(action - 1)
            state, bs_seq, buffer_seq, OK_for_training = state_to_tensor(state, bs_seq, buffer_seq, seq_num)

    return target_values, Q_values, rewards

cce = tf.keras.losses.CategoricalCrossentropy()

def compute_loss(dqn, target_values, Q_values):
    target_values_ten = tf.reshape(target_values, [1, -1, 3])
    Q_values_ten = tf.reshape(Q_values, [1, -1, 3])
    loss = tf.keras.losses.MSE(target_values_ten, Q_values_ten)

    return loss

def train_step(initial_state, bs_seq, buffer_seq, dqn: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, gamma: float, max_steps_per_episode: int, seq_num):

    with tf.GradientTape() as tape:
        target_values, Q_values, rewards = run_episode(initial_state, bs_seq,
                                                       buffer_seq, dqn,
                                                       max_steps_per_episode,
                                                       True, seq_num)
        loss = compute_loss(dqn, target_values, Q_values)

    grads_actor = tape.gradient(loss, dqn.trainable_variables)
    optimizer.apply_gradients(zip(grads_actor, dqn.trainable_variables))
    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


def test_step(initial_state, bs_seq, buffer_seq, dqn: tf.keras.Model,
              max_steps_per_episode: int, seq_num):
    with tf.GradientTape() as tape:
        target_values, Q_values, rewards = run_episode(initial_state, bs_seq,
                                                       buffer_seq, dqn,
                                                       max_steps_per_episode,
                                                       False, seq_num)
    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


def main():

    train_set_path = "./env/video_size/"
    test_set_path = "./env/video_size/"
    test_bw_set_path = "./env/bw_data/"
    pretrain_path = ""
    a_num = 3
    seq_num = 6
    min_episodes_criterion = 100
    max_episodes = 500
    max_steps_per_episode = 60
    iter_per_test = 50
    running_reward = 0
    gamma = 0.99

    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
    dqn = DQN(a_num)
    best = -float('inf')
    with tqdm.trange(max_episodes) as t:
        for i in t:
            bs_seq = []
            buffer_seq = []
            initial_non_ten = env.reset(train_set_path, test_bw_set_path, train=True)
            initial_state, bs_seq, buffer_seq, _ = state_to_tensor(initial_non_ten, bs_seq, buffer_seq, seq_num)

            episode_reward = float(train_step(initial_state, bs_seq, buffer_seq, dqn, optimizer, gamma, max_steps_per_episode, seq_num))
            if (i == 0 and len(pretrain_path) > 0):
                dqn.load_weights(pretrain_path, by_name=True, skip_mismatch=True)
                continue
            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)
            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

            if (i + 1) % iter_per_test == 0:
                test_reward = []
                print(" start testing...")
                while (True):
                    bs_seq = []
                    buffer_seq = []
                    initial_non_ten, done = env.reset(test_set_path, test_bw_set_path, train=False)
                    if done:
                        break
                    initial_state, bs_seq, buffer_seq, _ = state_to_tensor(initial_non_ten, bs_seq, buffer_seq, seq_num)
                    episode_reward = float(test_step(initial_state, bs_seq, buffer_seq, dqn, max_steps_per_episode, seq_num))
                    test_reward.append(episode_reward)
                print("cur_rewad: ", np.mean(test_reward), " , Best_reward: ", best)
                if (np.mean(test_reward) > best):
                    best = np.mean(test_reward)
                    dqn.save_weights('./model_best.h5')
            
            with open("./log.txt", "a") as f:
                f.write(str(episodes_reward[-1]) + " " + str(running_reward) + '\n')


if __name__ == "__main__":
    main()
