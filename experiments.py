import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from control import epsilon_greedy, evaluate_policy, Q_learning, sarsa, sarsa_lambda
from environment import TennisEnv


ALPHAS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
LLAMBDAS = [0.0, 0.25, 0.5, 0.75, 0.9]
DEFAULT_TRAIN_EPISODES = 200000
DEFAULT_EVAL_EPISODES = 2000


def build_configs(train_episodes, eval_episodes, seed):
    configs = []
    run_id = 0

    for alpha in ALPHAS:
        configs.append(("Sarsa(0)", alpha, None, train_episodes, eval_episodes, seed + run_id))
        run_id += 1

        configs.append(("Q-learning", alpha, None, train_episodes, eval_episodes, seed + run_id))
        run_id += 1

        for llambda in LLAMBDAS:
            configs.append(("Sarsa(lambda)", alpha, llambda, train_episodes, eval_episodes, seed + run_id))
            run_id += 1

    return configs


def train_and_evaluate(config):
    policy, alpha, llambda, train_episodes, eval_episodes, seed = config
    random.seed(seed)
    env = TennisEnv(seed=seed)

    if policy == "Sarsa(0)":
        q_values = sarsa(env, epsilon_greedy, alpha=alpha, num_episodes=train_episodes)
    elif policy == "Q-learning":
        q_values = Q_learning(env, epsilon_greedy, alpha=alpha, num_episodes=train_episodes)
    else:
        q_values = sarsa_lambda(
            env,
            epsilon_greedy,
            llambda=llambda,
            alpha=alpha,
            num_episodes=train_episodes,
        )

    win_rate, avg_return = evaluate_policy(q_values, env, num_episodes=eval_episodes)
    return policy, alpha, llambda, win_rate, avg_return


def run_experiments(train_episodes, eval_episodes, workers, seed):
    configs = build_configs(train_episodes, eval_episodes, seed)
    workers = max(1, min(workers, len(configs)))

    if workers == 1:
        return [train_and_evaluate(config) for config in configs]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(train_and_evaluate, config) for config in configs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            policy, alpha, llambda, win_rate, avg_return = result
            lambda_text = "N/A" if llambda is None else llambda
            print(
                f"Finished {policy}: alpha={alpha}, lambda={lambda_text}, "
                f"win_rate={win_rate:.3f}, avg_return={avg_return:.3f}"
            )

    return results


def print_results(results):
    best_policy, best_alpha, best_lambda, best_win_rate, best_avg_return = max(
        results,
        key=lambda result: (result[3], result[4]),
    )
    lambda_text = "N/A" if best_lambda is None else best_lambda
    print(
        "\nBest combination: "
        f"policy={best_policy}, alpha={best_alpha}, lambda={lambda_text}, "
        f"win_rate={best_win_rate:.3f}, avg_return={best_avg_return:.3f}"
    )

    print("\nAll results:")
    for policy, alpha, llambda, win_rate, avg_return in sorted(results):
        lambda_text = "N/A" if llambda is None else llambda
        print(
            f"{policy}: alpha={alpha}, lambda={lambda_text}, "
            f"win_rate={win_rate:.3f}, avg_return={avg_return:.3f}"
        )


def plot_win_rates(results):
    import matplotlib.pyplot as plt

    sarsa_lambda_results = [
        result for result in results if result[0] == "Sarsa(lambda)"
    ]
    optimal_lambda = max(
        sarsa_lambda_results,
        key=lambda result: (result[3], result[4]),
    )[2]

    policies = ["Sarsa(0)", "Q-learning", "Sarsa(lambda)"]
    for policy in policies:
        policy_results = [
            result
            for result in results
            if result[0] == policy
            and (policy != "Sarsa(lambda)" or result[2] == optimal_lambda)
        ]
        policy_results.sort(key=lambda result: result[1])
        label = policy
        if policy == "Sarsa(lambda)":
            label = f"{policy}, lambda={optimal_lambda}"
        plt.plot(
            [result[1] for result in policy_results],
            [result[3] for result in policy_results],
            marker="o",
            label=label,
        )

    plt.xlabel("Alpha")
    plt.ylabel("Win rate")
    plt.title("Policy win rates by alpha")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment_results = run_experiments(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        workers=args.workers,
        seed=args.seed,
    )
    print_results(experiment_results)
    plot_win_rates(experiment_results)
