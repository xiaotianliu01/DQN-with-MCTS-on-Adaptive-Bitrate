import sys
import cmath
import random
import numpy as np
import gym
import copy
import networkx as nx
import matplotlib.pyplot as plt

AVAILABLE_CHOICES = [-1, 0, 1]
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
SIGMA = 0.8
VIS = False
USE_CRN = True
SIM_LEN = 90
BUDGET = 27
C_IN_UCT = 1

class Node(object):

    def __init__(self, ID=0, level=0):
        self.parent = None
        self.children = []
        self.visit_times = 0
        self.state = None
        self.value = 0
        self.ID = ID
        self.level = level

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def set_children(self, children):
        self.children = children

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def value_renew(self, node):

        if (len(node.children) > 0):
            child_value = []
            total_visit_times = 0
            for child in node.children:
                child_value.append(child.value)
                total_visit_times += child.get_visit_times()
            node.visit_times = total_visit_times
            node.value = node.get_state().reward + SIGMA * np.mean(child_value)

    def is_all_expand(self):
        if len(self.children) == AVAILABLE_CHOICE_NUMBER:
            return True
        else:
            return False

    def add_child(self, sub_node):
        self.children.append(sub_node)


def get_upper_lower_reward_bound(player_env):

    player = copy.deepcopy(player_env)
    rebuf_lower_bound = []
    watching_upper_bound = []
    for i in range(SIM_LEN - player_env.download_chunk_index):
        player.step(1)
        rebuf_lower_bound.append(player.rebuf_reward)
        watching_upper_bound.append(player.watch_reward)
    del player

    player = copy.deepcopy(player_env)
    watching_lower_bound = []
    rebuf_upper_bound = []
    for i in range(SIM_LEN - player_env.download_chunk_index):
        player.step(-1)
        watching_lower_bound.append(player.watch_reward)
        rebuf_upper_bound.append(player.rebuf_reward)
    del player

    temp_a = 1
    player = copy.deepcopy(player_env)
    smooth_lower_bound = []
    for i in range(SIM_LEN - player_env.download_chunk_index):
        player.step(temp_a)
        temp_a = -temp_a
        smooth_lower_bound.append(player.smooth_reward)
    del player

    def compute_discounted(reward_list):
        res = []
        reward_list.reverse()
        for reward in reward_list:
            if (len(res) == 0):
                res.append(reward)
            else:
                dis = reward + res[-1] * SIGMA
                res.append(dis)
        res.reverse()
        return res

    reward_upper_bound = [watching_upper_bound[i] + rebuf_upper_bound[i] for i in range(len(watching_upper_bound))]
    reward_lower_bound = [watching_lower_bound[i] + rebuf_lower_bound[i] + smooth_lower_bound[i] for i in range(len(watching_lower_bound))]

    reward_upper_bound = compute_discounted(reward_upper_bound)
    reward_lower_bound = compute_discounted(reward_lower_bound)

    return reward_upper_bound, reward_lower_bound


class State(object):

    def __init__(self, reward, player, action):
        self.player = player
        self.reward = reward
        self.from_action = action

    def expand_random_state(self, tried_states, node_ID):

        while (True):
            random_choice = random.choice([choice for choice in AVAILABLE_CHOICES])
            if (random_choice not in tried_states):
                break
        new_player = copy.deepcopy(self.player)
        if (node_ID < AVAILABLE_CHOICE_NUMBER + 1):
            new_player.regenerate_bw(use_CRN=USE_CRN)
            upper, lower = get_upper_lower_reward_bound(new_player)
        _, reward, _, _ = new_player.step(random_choice)
        next_state = State(reward, new_player, random_choice)
        if (node_ID < AVAILABLE_CHOICE_NUMBER + 1):
            return next_state, upper, lower
        return next_state

    def sim_to_get_value(self):

        player = copy.deepcopy(self.player)
        rewards = []

        for i in range(SIM_LEN - self.player.download_chunk_index):
            random_choice = random.choice(AVAILABLE_CHOICES)
            _, reward, _, _ = player.step(random_choice)
            rewards.append(reward)

        rewards.reverse()
        dis_reward = 0
        for rew in rewards:
            dis_reward = dis_reward * SIGMA + rew
        del player

        return dis_reward


def expand(node, node_ID):

    tried_states = [sub_node.get_state().from_action for sub_node in node.get_children()]
    if (node_ID < AVAILABLE_CHOICE_NUMBER + 1):
        new_state, upper, lower = node.get_state().expand_random_state(
            tried_states, node_ID)
    else:
        new_state = node.get_state().expand_random_state(tried_states, node_ID)
    sub_node = Node(ID=node_ID, level=node.level + 1)
    exp_dis_reward = new_state.sim_to_get_value()
    sub_node.set_state(new_state)
    sub_node.set_value(new_state.reward + SIGMA * exp_dis_reward)
    node.add_child(sub_node)
    sub_node.set_parent(node)
    if (node_ID < AVAILABLE_CHOICE_NUMBER + 1):
        return sub_node, upper, lower
    return sub_node


def norm_UCT(node, upper, lower):
    best_score = -float('inf')
    best_sub_node = None
    visit_times_sum = 0
    c = C_IN_UCT

    for sub_node in node.get_children():
        visit_times_sum += sub_node.get_visit_times()

    for sub_node in node.get_children():
        score = (sub_node.value - lower[sub_node.level - 1]) / (upper[sub_node.level - 1] - lower[sub_node.level - 1]) + c * cmath.sqrt(2 * cmath.log(visit_times_sum) / sub_node.get_visit_times())
        if score > best_score:
            best_score = score
            best_sub_node = sub_node
    return best_sub_node


def best_child(node, upper=None, lower=None):
    best_sub_node = norm_UCT(node, upper, lower)
    return best_sub_node


def tree_policy(node, node_ID, upper=None, lower=None):

    while (True):
        if node.is_all_expand():
            if (node_ID < AVAILABLE_CHOICE_NUMBER + 1):
                node = best_child(node)
            else:
                node = best_child(node, upper, lower)
        else:
            if (node_ID < AVAILABLE_CHOICE_NUMBER + 1):
                sub_node, upper, lower = expand(node, node_ID)
                return sub_node, upper, lower
            sub_node = expand(node, node_ID)
            return sub_node


def process_upper_lower(uppers, lowers):

    processed_upper = [
        np.max([uppers[0][i], uppers[1][i], uppers[2][i]])
        for i in range(len(uppers[0]))
    ]
    processed_lower = [
        np.min([lowers[0][i], lowers[1][i], lowers[2][i]])
        for i in range(len(lowers[0]))
    ]

    return processed_upper, processed_lower


def monte_carlo_tree_search(node):

    computation_budget = BUDGET
    node_ID = 1
    uppers = []
    lowers = []
    processed_upper = None
    processed_lower = None

    for i in range(computation_budget):
        if (node_ID < AVAILABLE_CHOICE_NUMBER + 1):
            expand_node, upper, lower = tree_policy(node, node_ID)
            uppers.append(upper)
            lowers.append(lower)
        else:
            expand_node = tree_policy(node, node_ID, processed_upper, processed_lower)
        backup(expand_node)
        node_ID += 1
        if (node_ID == AVAILABLE_CHOICE_NUMBER + 1):
            processed_upper, processed_lower = process_upper_lower(
                uppers, lowers)
    values = [0 for i in range(AVAILABLE_CHOICE_NUMBER)]
    for child in node.get_children():
        values[child.get_state().from_action + 1] = child.value

    mean = np.mean(values)
    std = np.std(values)
    values = [(i - mean) / std for i in values]

    if (VIS == True):
        graph = nx.DiGraph()
        graph, pos = create_graph(graph, node)
        fig, ax = plt.subplots(figsize=(5, 5))
        nx.draw_networkx(graph,
                         pos,
                         ax=ax,
                         node_size=50,
                         with_labels=False,
                         font_size=5,
                         font_color='white')
        plt.show()
    return values


def backup(node):

    while node != None:
        node.visit_times_add_one()
        node.value_renew(node)
        node = node.parent


def create_graph(G, node, ID=0, pos={}, x=0, y=0, layer=1):
    pos[node.ID] = (x, y)
    if (len(node.children) > 0):
        for child in node.children:
            G.add_edge(node.ID, child.ID)
            if child.get_state().from_action == -1:
                l_x, l_y = x - 1 / 3**layer, y - 1
                l_layer = layer + 1
            if child.get_state().from_action == 0:
                l_x, l_y = x, y - 1
                l_layer = layer + 1
            if child.get_state().from_action == 1:
                l_x, l_y = x + 1 / 3**layer, y - 1
                l_layer = layer + 1
            create_graph(G, child, x=l_x, y=l_y, pos=pos, layer=l_layer)
    return (G, pos)
