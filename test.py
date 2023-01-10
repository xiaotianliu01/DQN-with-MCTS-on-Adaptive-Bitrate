import numpy as np
import tensorflow as tf
import collections
import tqdm
import statistics
import gym
import env
import os
from train import DQN

env = gym.make("video_player-v0")

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


def state_to_tensor(initial_non_ten, bs_seq, buffer_seq, seq_num=6):
    temp = []
    last_bit_rate_idx = initial_non_ten[4]
    if (last_bit_rate_idx < 5 and last_bit_rate_idx > 0):
        temp = initial_non_ten[1][last_bit_rate_idx - 1:last_bit_rate_idx + 2]
        state_temp1 = tf.constant(initial_non_ten[1][last_bit_rate_idx -1:last_bit_rate_idx + 2], dtype=tf.float32)
    elif (last_bit_rate_idx == 5):
        temp.append(initial_non_ten[1][4])
        temp.append(initial_non_ten[1][5])
        temp.append(initial_non_ten[1][5])
        state_temp1 = tf.constant(temp, dtype=tf.float32)
    elif (last_bit_rate_idx == 0):
        temp.append(initial_non_ten[1][0])
        temp.append(initial_non_ten[1][0])
        temp.append(initial_non_ten[1][1])
        state_temp1 = tf.constant(temp, dtype=tf.float32)
    bs_seq, OK_for_training = generate_sequence(bs_seq, initial_non_ten[0][0], seq_num)
    throughput = [i / 8 * 0.95 for i in bs_seq]
    buffer_seq, _ = generate_sequence(buffer_seq, initial_non_ten[2], seq_num)

    if (OK_for_training):
        temp_ = [1 / i for i in throughput]
        temp_ = tf.constant(temp_, dtype=tf.float32)
        temp_ = tf.reshape(temp_, [seq_num, 1])
        state_temp1 = temp_ * tf.reshape(state_temp1, [1, 3])
        state_temp2 = tf.constant(buffer_seq, dtype=tf.float32)
        state_temp2 = tf.reshape(state_temp2, [1, seq_num, 1])
        state = [tf.reshape(state_temp1, [1, seq_num, 3]), state_temp2]

        return state, bs_seq, buffer_seq, True, initial_non_ten[0][0], temp
    else:
        state_temp1 = state_temp1 / throughput[-1]
        state_temp2 = tf.constant(initial_non_ten[2], dtype=tf.float32)
        state_temp2 = tf.reshape(state_temp2, [1, 1])
        state = [state_temp1, state_temp2]
        return state, bs_seq, buffer_seq, False, initial_non_ten[0][0], temp


def write_state(buffer_size, curr_bit_rate, reward, bw_t, chunk, f):
    f.write(
        str(curr_bit_rate) + '\t' + str(buffer_size) + '\t' + str(chunk[0]) +
        '\t' + str(chunk[1]) + '\t' + str(chunk[2]) + '\t' + str(reward) +
        '\t' + str(bw_t) + '\n')


def bitrate_iter(action, curr_bit_rate):
    if (action == 0):
        if (curr_bit_rate > 0):
            output = curr_bit_rate - 1
        else:
            output = curr_bit_rate
    elif (action == 2):
        if (curr_bit_rate < 5):
            output = curr_bit_rate + 1
        else:
            output = curr_bit_rate
    else:
        output = curr_bit_rate
    return output


def run_episode(initial_state, bs_seq, buffer_seq, dqn: tf.keras.Model,
                max_steps: int, curr_bit_rate, f, seq_num):

    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    state = initial_state
    action_statistic = {-1: 0, 0: 0, 1: 0}
    chunk_num = 0
    OK_for_training = 0
    last_action = 0
    trumble = 0
    bw_t = 0
    chunk = []
    reward = 0
    for t in tf.range(max_steps):
        if (OK_for_training):
            value = dqn(state)
            action = tf.argmax(value, axis=1).numpy()[0] - 1
            action_statistic[action] += 1
            new_state, reward, done, _ = env.step(action)
            curr_bit_rate = bitrate_iter(action + 1, curr_bit_rate)
            write_state(state[1][0][1].numpy()[0], curr_bit_rate, reward, bw_t, chunk, f)
            state, bs_seq, buffer_seq, OK_for_training, bw_t, chunk = state_to_tensor(new_state, bs_seq, buffer_seq, seq_num)
            reward = tf.constant(reward, dtype=tf.float32)
            rewards = rewards.write(t, reward)
            chunk_num = chunk_num + 1
            if done:
                break
        else:
            action = 1
            curr_bit_rate = bitrate_iter(action, curr_bit_rate)
            state_new, reward, done, _ = env.step(action - 1)
            if done:
                break
            state, bs_seq, buffer_seq, OK_for_training, bw_t, chunk = state_to_tensor(state_new, bs_seq, buffer_seq, seq_num)
            if (OK_for_training == False):
                write_state(state[1][0][0].numpy(), curr_bit_rate, reward, bw_t, chunk, f)
            else:
                write_state(state[1][0][5].numpy()[0], curr_bit_rate, reward, bw_t, chunk, f)

    rewards = rewards.stack()
    return rewards, chunk_num, action_statistic


def test_step(initial_state, bs_seq, buffer_seq, dqn: tf.keras.Model,
              max_steps_per_episode: int, curr_bit_rate, f, seq_num):
    with tf.GradientTape() as tape:
        rewards, _, sta = run_episode(initial_state, bs_seq, buffer_seq, dqn,
                                      max_steps_per_episode, curr_bit_rate, f,
                                      seq_num)
    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward, sta


def main():

    test_set_path = "./env/video_size/"
    test_bw_path = "./env/bw_data/"
    test_log_path = "./test_log/"
    pretrain_path = "./model_best.h5"
    a_num = 3
    seq_num = 6
    min_episodes_criterion = 100
    max_steps_per_episode = 45

    dqn = DQN(a_num)
    test_reward = []
    print(" start testing...")
    iid = -1
    sta = []
    while (True):
        bs_seq = []
        buffer_seq = []
        curr_bit_rate = 0
        initial_non_ten, done = env.reset(test_set_path, test_bw_path, train=False)
        if (done):
            break
        with open(test_log_path + str(iid) + ".txt", 'a') as f:
            initial_state, bs_seq, buffer_seq, _, _, _ = state_to_tensor(initial_non_ten, bs_seq, buffer_seq, seq_num)

            episode_reward, sta_ = test_step(initial_state, bs_seq, buffer_seq,
                                             dqn, max_steps_per_episode,
                                             curr_bit_rate, f, seq_num)
        if (iid == -1):
            dqn.load_weights(pretrain_path)
        iid = iid + 1
        if (iid != -1):
            test_reward.append(episode_reward)
            sta.append(sta_)
    print(np.mean(test_reward))
    action_statistic = [0, 0, 0]
    for i in sta:
        for j in range(-1, 2):
            action_statistic[j + 1] += i[j]
    print(action_statistic)

if __name__ == "__main__":
    main()
