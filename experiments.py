import random

import matplotlib.pyplot as plt

from control import epsilon_greedy, evaluate_policy, sarsa_lambda
from environment import TennisEnv


ALPHA = 0.01
TRAIN_EPISODES = 120000
EVAL_EPISODES = 1000
SEEDS = [1, 2, 3]
LLAMBDAS = [i / 10 for i in range(11)]


results = []

for llambda in LLAMBDAS:
    seed_results = []

    for seed in SEEDS:
        random.seed(seed)
        train_env = TennisEnv(seed=seed)

        Q = sarsa_lambda(
            train_env,
            epsilon_greedy,
            llambda=llambda,
            alpha=ALPHA,
            num_episodes=TRAIN_EPISODES,
        )

        random.seed(seed + 1000)
        eval_env = TennisEnv(seed=seed + 1000)
        win_rate, avg_return = evaluate_policy(
            Q,
            eval_env,
            num_episodes=EVAL_EPISODES,
        )
        seed_results.append((win_rate, avg_return))

    mean_win_rate = sum(result[0] for result in seed_results) / len(seed_results)
    mean_avg_return = sum(result[1] for result in seed_results) / len(seed_results)

    results.append((llambda, mean_win_rate, mean_avg_return))
    print(
        f"lambda={llambda:.1f}, "
        f"win_rate={mean_win_rate:.3f}, "
        f"avg_return={mean_avg_return:.3f}"
    )


best_lambda, best_win_rate, best_avg_return = max(
    results,
    key=lambda result: (result[1], result[2]),
)
print(
    f"\nBest lambda={best_lambda:.1f}, "
    f"win_rate={best_win_rate:.3f}, "
    f"avg_return={best_avg_return:.3f}"
)


plt.plot(
    [result[0] for result in results],
    [result[1] for result in results],
    marker="o",
)
plt.xlabel("Lambda")
plt.ylabel("Win rate")
plt.title(f"Sarsa(lambda), alpha={ALPHA}, averaged over {len(SEEDS)} seeds")
plt.xticks(LLAMBDAS)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
