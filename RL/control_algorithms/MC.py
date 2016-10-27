import gym
import numpy as np
from scipy import stats
# import matplotlib.pyplot as plt
# %matplotlib inline
game_id = 1
map_size =4
if (game_id == 1):
    map_size = 8
game = ['FrozenLake-v0', 'FrozenLake8x8-v0']
env = gym.make(game[game_id])
env.reset()
actionN = env.nA
stateN = env.nS
T = np.zeros([stateN, actionN, stateN])
R = np.zeros([stateN, actionN, stateN])
# print(env.P[0][0])
for s in range(stateN):
    for a in range(actionN):
        transitions = env.P[s][a]
        for p_trans,next_s,rew,done in transitions:
            T[s,a,next_s] += p_trans
            R[s,a,next_s] += rew
        T[s,a,:]/=np.sum(T[s,a,:])


#test optimal policy
max_time_steps = 100000000
n_episode = 100000
history = []
env.monitor.start('recordings', force=True)

np.random.seed = 42
# opt = np.random.randint(4, size=64)
opt = [3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 0, 3, 0, 3, 2, 3, 2, 2, 0, 0, 0, 2, 0, 2, 2, 2, 0, 3, 0, 1, 2, 1, 3, 2, 0, 1, 0, 1, 3, 0, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
N_vis_state = np.zeros(stateN)
value_estimated = np.zeros(stateN)
q_estimated = np.zeros((stateN, actionN))
N_vis_q = np.zeros((stateN, actionN))
count_positive = 0
eps = 0.05
obs = stats.rv_discrete(values = ((0, 1), (1 - eps, eps)))


for i_episode in range(n_episode):
    reward_episode = 0
    observation = env.reset() #reset environment to beginning 
    states_episode = {observation}
    sa_episode = set()
    # N_vis_state[observation] += 1 
    #run for several time-steps
    for t in range(max_time_steps): 
        #sample a random action 
        go_random = obs.rvs()
        action = (1 - go_random) * opt[observation] + go_random * np.random.randint(4)
        sa_episode.add((observation, action))
        #observe next step and get reward 
        prev_observ = observation
        observation, reward, done, info = env.step(action)
        reward_episode += reward
        states_episode.add(observation)
        # N_vis_state[observation] += 1
        
        if done or t == max_time_steps - 1:
            if (done and reward_episode==1):
                count_positive += 1
            for i in states_episode:
                N_vis_state[i] += 1
            for i in states_episode:
                value_estimated[i] = \
                value_estimated[i] + (1/N_vis_state[i])*(reward_episode - value_estimated[i])
            for i in sa_episode:
                N_vis_q[i[0], i[1]] += 1
            for i in sa_episode:
                q_p = q_estimated[i[0], i[1]]
                N_q = N_vis_q[i[0], i[1]]
                q_estimated[i[0], i[1]] = q_p + (1/N_q)*(reward_episode - q_p)
            # env.render()
            break


    #     print(policy)
policy = np.argmax(q_estimated, axis=1)                 
pls = np.zeros((stateN, actionN))
for i in range(stateN):
    pls[i][policy[i]] = 1   
policy = pls
            
env.monitor.close()
print((np.argmax(q_estimated, axis=1) - np.array(opt)).reshape(8, 8))
# for i in value_estimated.reshape(8, 8):
    # print (''.join([' '  + str(j) for j in i]))
# print(q_estimated)


