import random
import math
from collections import defaultdict
from environment import TennisEnv


def greedy_policy(Q, state, env):
    values = []

    for action in env.rally_actions:
        values.append((Q.get((state, action), 0.0), action))

    max_value = max(v for v, a in values)
    best_actions = [a for v, a in values if v == max_value]

    return random.choice(best_actions)

def epsilon_greedy(Q, state, env, epsilon):
    # Q will look like:
    # (state(phase, ball_position, player_balance, opponent_balance, stamina, pressure),
    # action("crosscourt",) : some value
    # (state, action) : value
    # eps percent of the time take random action
    # 1-eps of the time take greedy action

    # best action from state,action pairs
    if random.random() < epsilon: # random action
        return(random.choice(env.rally_actions))
    else:
        return greedy_policy(Q, state, env)
def run_episode(Q, env, mu, episode_num):
    trajectory = []

    state = env.reset()
    while True:
        # use 1/sqrt(n_episodes) to get GLIE
        action = mu(Q, state, env, epsilon = max(0.1, 1 / (episode_num ** 0.25))) # mu will almost always be e-greedy
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        if done:
            break
        state = next_state
    return trajectory




def mc_control(env, mu): #every-visit worked better in eval so ill just use it here
    N = defaultdict(float)
    Q = defaultdict(float)
    # training step, build Q
    for episode in range(1, 100000): # 100000 episodes
        trajectory = run_episode(Q, env, mu, episode)
        G = 0
        for state, action, reward in reversed(trajectory): #(state, action, reward)
            N[(state, action)] += 1
            G = reward + env.gamma * G
            Q[(state, action)] = Q[(state, action)] + (1/N[(state, action)]) * (G - Q[(state, action)]) 
    return Q







def sarsa(env, mu, alpha=0.005, num_episodes=100000): # build Q using sarsa(0)
    Q = defaultdict(float)
    for episode in range(1, num_episodes+1):
        epsilon = 1/(episode**0.20)
        state = env.reset()
        # must be declared here because otherwise we would choose an action twice in one state
        action = mu(Q, state, env, epsilon) # use e-greedy
        while True:
            next_state, reward, done = env.step(action)
            if done: # treat next state as 0
                Q[(state, action)] += alpha*(reward - Q[(state, action)]) 
                break
            next_action = mu(Q, next_state, env, epsilon) # next action also comes from mu behavorial policy
            Q[(state, action)] = Q[(state,action)] + alpha*(reward + 
                    env.gamma*Q[(next_state, next_action)] - Q[(state, action)])
            state = next_state
            action = next_action

    return Q


            
            





def Q_learning(env, mu, alpha=0.005, num_episodes=100000):
    Q = defaultdict(float)

    for episode in range(1, num_episodes+1):
        epsilon = 1/(episode**0.20)
        state = env.reset()
        
        while True:
            action = mu(Q, state, env, epsilon)  # action is chosen from mu
            next_state, reward, done = env.step(action)
            # the difference here, is that we're gonna get our next_action by taking the max from Q
            # this means this is off-policy learning, the policy used to collect experience
            # is different from the policy being learned/evaluated
            if done: # consider value next_state to be 0 if terminal
                delta = reward - Q[(state, action)] 
                Q[(state,action)] += alpha * delta
                break
            # next action gives max value possible
            best_next_action = greedy_policy(Q, next_state, env)
            delta = (reward + env.gamma*Q.get((next_state, best_next_action), 0.0) 
            - Q[(state, action)]
            )
            Q[(state, action)] += alpha*delta
            
            # transition to next step (off-policy)
            state = next_state
    return Q
            


def sarsa_lambda(env, mu, llambda, alpha=0.005):
    pass


env = TennisEnv()



Q = Q_learning(env, epsilon_greedy, num_episodes=1000000)
print(Q)

import statistics

print("mean:", statistics.mean(Q.values()))
print("abs max:", max(abs(v) for v in Q.values()))

# Todo: track win-rate and average return
def evaluate_policy(Q, env, num_episodes):
    pass