from RLshortestmazepath import mazeEnv




# implement first visit monte carlo eval/ epsilon greedy control 

# collect experience by choosing actions based on policy 
# should return a dictionary saying which states were visited
# should also return total return for each state in the rollout
def mc_rollout(env, policy): 
    current_state = env.initial_state


def monte_carlo(env, policy, theta=1e-5):
    # V(s) = V(s) + 1/N(s)(Gt - V(s))
    
    # initialize V
    V = {s: 0.0 for s in env.states}
    N = {s: 0 for s in env.states}
    for _ in range(1000): # 1000 episodes 
        visited, total = mc_rollout(env, policy)
        for state, visits in visited.items(): # ex. (5, 1)
            N[state] += visits 
            if visits != 0:
                V[state] = V[state] + (1/N[state]*(total[state] - V[state]))  
    return V        
            
            







# implement td-learning and sarsa control
def td_sarsa_0(env, policy, theta=1e-5, alpha=0.1):
    pass


# implement backward view TD-lambda with eligibility traces 
def td_lambda(env, policy, theta=1e-5, alpha=0.1):
    pass



env = mazeEnv()
