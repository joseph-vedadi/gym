import gym, numpy as np
from gym import wrappers
import matplotlib.pyplot as plt
from copy import deepcopy

env = gym.make("CartPole-v0")

MAXSTATE = 10 ** 4
GAMMA = 0.9
ALPHA = 0.15


def max_dict(d):
    max_v = float("-inf")
    max_k = 0
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_k = key
    return max_k, max_v


def create_bins():
    bins = np.zeros((4, 10))
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-0.418, 0.418, 10)
    bins[3] = np.linspace(-5, 5, 10)
    return bins


def assign_bins(obs, bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(obs[i], bins[i])
    return state


def get_state_as_string(state):
    return "".join(str(int(e)) for e in state)


def get_all_states_as_string():
    return [str(i).zfill(4) for i in range(MAXSTATE)]


def init_Q():
    all_state = get_all_states_as_string()
    action_map = {action: 0 for action in range(env.action_space.n)}
    return {state: deepcopy(action_map) for state in all_state}


def play_one_game(bins, Q, eps=0.5):
    obs = env.reset()
    done = False
    cnt = 0
    state = get_state_as_string(assign_bins(obs, bins))
    total_reword = 0
    while not done:
        cnt += 1
        if np.random.uniform() < eps:
            act = env.action_space.sample()
        else:
            act = max_dict(Q[state])[0]
        obs, reward, done, _ = env.step(act)
        total_reword += reward
        if done and cnt < 200:
            reward = -300
        state_new = get_state_as_string(assign_bins(obs, bins))
        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA * (reward + GAMMA * max_q_s1a1 - Q[state][act])
        state, act = state_new, a1
    return total_reword, cnt


def play_many_games(bins, N):
    Q = init_Q()
    length = []
    reward = []
    for n in range(N):
        eps = 1.0 / np.sqrt(n - 1) if n > 1 else 0
        episode_reward, episode_length = play_one_game(bins, Q, eps)
        if n % 100:
            print(n, "%.4f" % eps, episode_reward)
        length.append(episode_length)
        reward.append(episode_reward)
    return length, reward


def plot_results(total_rewords):
    N = len(total_rewords)
    running = [np.mean(total_rewords[max(0, t - 100) : (t + 1)]) for t in range(N)]
    plt.plot(running)
    plt.show()


if __name__ == "__main__":
    bins = create_bins()
    episode_lengths, episode_rewards = play_many_games(bins, 500)
    plot_results(episode_rewards)

