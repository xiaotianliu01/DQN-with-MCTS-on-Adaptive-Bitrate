from gym import spaces, core
import numpy as np
import tensorflow as tf
import random
import math
from . import merton_series
import os

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 2.8
SMOOTH_PENALTY = 1
DEFAULT_BITRATE_INDEX = 0
EPS = 1e-6
BITS_IN_BYTE = 8
NETWORK_DISCOUNT = 0.95
BUFFER_THRESH = 50
CHUNK_LEN = 4
EPISDOE_LEN = 100
TEST_VIDEO_NUM = 290
BITRATE_LEVEL = 6

class MyEnv(core.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, ))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, ))
        self.test_bw_id = 0
        self.test_done = 0

    def regenerate_bw(self, use_CRN=True):
        if (use_CRN):
            new_sequence = self.net.regenrate_bw(self.net[self.download_chunk_index], 200 - self.download_chunk_index)
            self.net.series_bw = self.net.series_bw[:self.download_chunk_index + 1] + new_sequence
        else:
            new_sequence = self.net.regenrate_bw(self.net[self.download_chunk_index - 1], 200 - self.download_chunk_index + 1)
            self.net.series_bw = self.net.series_bw[:self.download_chunk_index] + new_sequence

    def reset(self, video_path, bw_path, train):

        self.path = video_path
        self.train = train
        self._get_cur_video()
        self.buffer = 0
        self.rebuf_time = 0.0
        self.last_bit_rate = 0
        self.download_chunk_index = 0
        self.done = False
        self.watch_reward = 0
        self.rebuf_reward = 0
        self.smooth_reward = 0

        if (train == True):
            self.net = merton_series.merton(len(self.video_info['chunk_times']))
        else:
            self.bw_list = []
            with open(bw_path + str(self.test_bw_id) + '.txt', "rb") as f:
                for line in f:
                    self.bw_list.append(float(line))
            self.test_bw_id += 1
            if (self.test_bw_id == TEST_VIDEO_NUM):
                self.test_done = True

        obs = self._get_start()
        self._get_done()
        if (self.train == False):
            if (self.test_done == True):
                self.test_done = False
                self.test_bw_id = 0
                return obs, True
            else:
                return obs, False
        return obs

    def step(self, action_change, abs_level=False):

        if (abs_level == False):
            action = self.last_bit_rate + action_change
        else:
            action = action_change
        if action < 0:
            action = 0
        if action > BITRATE_LEVEL-1:
            action = BITRATE_LEVEL-1
        if self.done:
            return None, None, True, None
        self._get_cur_bw()
        reward = self._get_reward(action)
        self.last_bit_rate = action
        obs = [[self.cur_bw],
               self.video_info['chunk_sizes'][self.download_chunk_index + 1],
               self.buffer_size, self.download_chunk_index + 1,
               self.last_bit_rate]
        done = self._get_done()
        info = {}
        return obs, reward, done, info

    def _get_cur_video(self):
        VIDEO_DIR = self.path
        bit_rates = []
        chunk_times = []
        chunk_sizes = [[] for i in range(BITRATE_LEVEL)]
        chunk_brs = [[] for i in range(BITRATE_LEVEL)]
        for i in range(BITRATE_LEVEL):
            file_name = "video_size_" + str(i)
            with open(VIDEO_DIR + file_name, 'rb') as f:
                for line in f:
                    chunk_sizes[i].append(float(line.split()[0]))
                    chunk_times.append(CHUNK_LEN)
                    br = VIDEO_BIT_RATE[i]
                    chunk_brs[i].append(br)

        def change_order(li):
            length = len(li[0])
            res = [[] for i in range(length)]
            for i in range(length):
                for j in range(BITRATE_LEVEL):
                    res[i].append(li[j][i])
            return res

        chunk_sizes = change_order(chunk_sizes)
        chunk_brs = change_order(chunk_brs)

        chunk_sizes = chunk_sizes + chunk_sizes
        chunk_brs = chunk_brs + chunk_brs
        chunk_times = chunk_times + chunk_times

        video_info = {
            'bit_rates': bit_rates,
            'chunk_times': chunk_times,
            'chunk_sizes': chunk_sizes,
            'chunk_brs': chunk_brs
        }
        self.video_info = video_info

    def _get_cur_bw(self):
        if (self.train == True):
            self.cur_bw = self.net[self.download_chunk_index]
        else:
            self.cur_bw = self.bw_list[self.download_chunk_index]

    def _get_start(self):
        self._get_cur_bw()

        self.download_chunk_index = 0
        self.buffer_size = self.video_info['chunk_times'][self.download_chunk_index]
        self.last_bit_rate = DEFAULT_BITRATE_INDEX
        
        obs = [[self.cur_bw],
               self.video_info['chunk_sizes'][self.download_chunk_index + 1],
               self.buffer_size, self.download_chunk_index + 1,
               self.last_bit_rate]
        return obs

    def _download_one_chunk(self, br_index):
        self.rebuf_time = 0.0
        video_info = self.video_info
        video_chunk_time = video_info['chunk_times'][self.download_chunk_index]
        video_chunk_size = video_info['chunk_sizes'][self.download_chunk_index][br_index]

        throughput = self.cur_bw / BITS_IN_BYTE * NETWORK_DISCOUNT
        duration = video_chunk_size / throughput
        buffer_size = np.maximum(self.buffer_size - duration, 0.0)

        self.rebuf_time = np.maximum(duration - self.buffer_size, 0.0)

        buffer_size += video_chunk_time

        if self.download_chunk_index + 1 < EPISDOE_LEN:
            next_chunk_time = video_info['chunk_times'][self.download_chunk_index + 1]
            if buffer_size + next_chunk_time > BUFFER_THRESH:
                buffer_size = np.maximum(BUFFER_THRESH - next_chunk_time, 0.0)

        self.buffer_size = buffer_size

        rebuf_penalty = -REBUF_PENALTY * self.rebuf_time

        return rebuf_penalty

    def _watch_utility(self, bit_rate_index, chunk_index):
        utility = math.log(self.video_info['chunk_brs'][chunk_index][bit_rate_index] / 300)
        return utility

    def _smooth_penalty(self, bit_rate_index, last_bit_rate_index):
        smooth_penalty = -SMOOTH_PENALTY * np.abs(math.log(VIDEO_BIT_RATE[bit_rate_index]/VIDEO_BIT_RATE[last_bit_rate_index]))
        return smooth_penalty

    def _get_reward(self, action):
        bit_rate_index = action
        self.watch_reward = self._watch_utility(bit_rate_index, self.download_chunk_index)
        self.rebuf_reward = self._download_one_chunk(bit_rate_index)
        self.smooth_reward = self._smooth_penalty(bit_rate_index, self.last_bit_rate)

        reward = self.watch_reward + self.rebuf_reward + self.smooth_reward

        return reward

    def _get_done(self):
        self.download_chunk_index += 1
        if self.download_chunk_index >= EPISDOE_LEN:
            self.done = True
        return self.done
