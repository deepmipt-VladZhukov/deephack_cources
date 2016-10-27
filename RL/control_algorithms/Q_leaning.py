import gym
import numpy as np
from scipy import stats

game_id = 1
map_size =4
if (game_id == 1):
    map_size = 8
game = ['FrozenLake-v0', 'FrozenLake8x8-v0']
env = gym.make(game[game_id])
env.reset()
actionN = env.nA
stateN = env.nS
max_time_steps = 100000000
n_episode = 100000
history = []
alpha = 0.03
gamma = 0.99


env.monitor.start('recordings', force=True)
eps = 1
obs = stats.rv_discrete(values = ((0, 1), (1 - eps, eps)))
random_policy = stats.rv_discrete(values = ((0, 1, 2, 3), (1/4, 1/4, 1/4, 1/4)))
def eps_greedy(state, q): 
    is_observing = obs.rvs() 
    if (np.sum(q[state, :]) == 0):
        return random_policy.rvs()
    return is_observing * random_policy.rvs() + \
    (1 - is_observing) * np.argmax(q, axis=1)[state]

np.random.seed = 42
q = np.zeros((stateN, actionN))
for i_episode in range(n_episode):
    # if (i_episode %  == 0):
    if (i_episode % 100 == 0):
        print(eps)
    eps = 0.5*(1 - (1/ (1 + np.exp(3-6 * i_episode / n_episode))))
    obs = stats.rv_discrete(values = ((0, 1), (1 - eps, eps)))
    S = env.reset() #reset environment to beginning 
    # action = eps_greedy(S, q)#np.argmax(q, axis=1)[S]
    for t in range(max_time_steps):
        action = eps_greedy(S, q) 
        S_prime, reward, done, info = env.step(action)
        q[S, action]  = q[S, action] + alpha * (reward + gamma * np.max(q[S_prime]) - q[S, action])
        # action_prime = eps_greedy(S_prime, q)#np.argmax(q, axis=1)[S_prime]
        # q[S, action] = q[S, action] + \
        # alpha*(reward + gamma*q[S_prime, action_prime] - q[S, action])
        S = S_prime
        # action = action_prime
        if done:
            env.render()
            break


    #     print(policy)
          
def go(policy):
    acc = 0
    nn = 1000000
    # env.monitor.start('recordings', force=True)
    env.monitor.start('recordings', force=True)
    for i in range(nn):
        S = env.reset() 
        for t in range(max_time_steps): 
            action = policy[S]
            # print(S,action)
            S, reward, done, info = env.step(action)
            # env.render()
            if done:
                if (S == 63):
                    acc += 1
                # print(S)
                # print('-----------result----------')
                # print(t)
                # env.render()
                break
    env.monitor.close()
    print (acc / nn)
pls = np.argmax(q, axis=1)
go(pls)
gym.upload('/Users/vlad/Documents/PY_PROJECTS_FOLDER/DEEP_HACK/RL/MC_TD_TD(Lambd)/recordings', api_key='sk_4uOH7kK2R1SHIQlkRZotRw')




opt = [3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 0, 3, 0, 3, 2, 3, 2, 2, 0, 0, 0, 2, 0, 2, 2, 2, 0, 3, 0, 1, 2, 1, 3, 2, 0, 1, 0, 1, 3, 0, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
print (np.argmax(q, axis=1).reshape(8, 8) - np.array(opt).reshape(8, 8))
print (np.argmax(q, axis=1).reshape(8, 8))
env.monitor.close()
