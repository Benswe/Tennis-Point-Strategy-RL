import random
from environment import TennisEnv
from collections import defaultdict




# implement first visit monte carlo eval/ epsilon greedy control 

# collect experience by choosing actions based on policy 
# should return a dictionary saying which states were visited
# should also return total return for each state in the rollout
def generate_episode(env, policy): 
    trajectory = [] # [(s0, a0, r1), (s1, a1, r2)...]
    state = env.reset()
    done = False
    while True:
        action = policy(state)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        if done:
            break
        state = next_state
    return trajectory

# every visit because states appear to be visited multiple times
def every_visit_monte_carlo(env, policy):
    # V(s) = V(s) + 1/N(s)(Gt - V(s))
    
    # initialize V
    V = {}
    N = {}
    for _ in range(10000): # 10000 episodes 
        trajectory = generate_episode(env, policy)
        
        G = 0
        for state, action, reward in reversed(trajectory):
            if state not in V:
                V[state] = 0
                N[state] = 0
            G = reward + env.gamma*G
            N[state] += 1
            V[state] = V[state] + (1/N[state])*(G - V[state])

    return V        
        





# implement td-learning and sarsa control
def td_0(env, policy, alpha=0.01):
    # V(s) = V(s) + alpha[r + gammaV(s') - V(s)]
    V = {}
    #N = {}
    for _ in range(100000):
        # online updates so 
        state = env.reset()
        while True:
            next_state, reward, done = env.step(policy(state))
            if state not in V:
                V[state] = 0
            if next_state not in V:
                V[next_state] = 0
            delta = reward + env.gamma * V[next_state] - V[state]
            if done:
                delta = reward - V[state] # V[next_state] is 0 if terminal
            else:
                delta = reward + env.gamma * V[next_state] - V[state]
            V[state] = V[state] + alpha * delta
            if done:
                break
            state = next_state
    return V
            




# implement backward view TD-lambda with eligibility traces 
def td_lambda(env, policy, llambda=0.6, alpha=0.005):
    V = defaultdict(float) # Eligibility traces
    for _ in range(100000):
        E = defaultdict(float)
        state = env.reset()
        while True:
            next_state, reward, done = env.step(policy(state))
            E[state] += 1
            if done:
                delta = reward - V[state] # treat V[next_state] as 0 if terminal
            else:
                delta = reward + env.gamma * V[next_state] - V[state]
            for s in E:
                V[s] += alpha * delta * E[s]
                E[s] *= llambda * env.gamma # decay the Eligibiity trace
            if done:
                break
            state = next_state
    return V





env = TennisEnv()
def baseline_policy(state):

    phase, ball_position, player_balance, opponent_balance, stamina, pressure = state

    if opponent_balance == "defensive":
        return "crosscourt"

    return "heavy_topspin"

#print(generate_episode(env, baseline_policy))
def print_states_by_value(label, V):
    print(label)
    for state, value in sorted(V.items(), key=lambda item: item[1], reverse=True):
        print(f"{value}: {state}")

# 2. print start state value
if __name__ == "__main__":
    print_states_by_value("Every visit V:", every_visit_monte_carlo(env, baseline_policy))
    print_states_by_value("TD(0):", td_0(env, baseline_policy))
    print_states_by_value("TD(0.6):", td_lambda(env, baseline_policy))
