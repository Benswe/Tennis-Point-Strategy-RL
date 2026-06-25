import random


class TennisEnv:
    def __init__(self, gamma=0.9, seed=None):
        # Discount factor used by the learning algorithms outside this file.
        self.gamma = gamma

        # Local random generator makes experiments reproducible if a seed is passed.
        self.rng = random.Random(seed)

        # Rally length tracking prevents episodes from running forever.
        self.shot_count = 0
        self.max_shots = 28

        # These lists define the discrete values that can appear in the state tuple.
        self.ball_positions = [
            "wide",
            "body",
            "center",
            "deep",
            "short"
        ]

        self.balance_states = [
            "defensive",
            "neutral",
            "attacking"
        ]

        self.stamina_states = [
            "high",
            "medium",
            "low"
        ]

        self.pressure_states = [
            "low",
            "medium",
            "high"
        ]

        self.rally_actions = [
            "crosscourt",
            "down_the_line",
            "heavy_topspin",
            "drop_shot",
            "defensive_slice"
        ]

        # The current state is set by reset() and updated after every non-terminal step.
        self.state = None

    # every episode should start at a serving state
    def reset(self):
        # Start each point from a neutral rally situation.
        self.shot_count = 0
        self.state = (
            "rally",
            "center",
            "neutral",
            "neutral",
            "high",
            "low"
        )

        return self.state

    def step(self, action):
        # If step() is called before reset(), quietly start a new point.
        if self.state is None:
            self.reset()

        # Fail loudly if an algorithm tries an action the environment does not support.
        if action not in self.rally_actions:
            raise ValueError(f"Unknown action: {action}")

        # State layout:
        # (phase, ball_position, player_balance, opponent_balance, stamina, pressure)
        phase, ball_position, player_balance, opponent_balance, stamina, pressure = self.state
        self.shot_count += 1

        # Convert the chosen action plus current context into probabilities and transitions.
        spec = self._action_spec(action, ball_position, player_balance, opponent_balance, stamina, pressure)
        winner_prob = self._clamp(spec["winner"], 0.01, 0.75)
        error_prob = self._clamp(spec["error"], 0.01, 0.75)
        total_terminal = winner_prob + error_prob

        # Keep rallies alive often enough for tactical sequences to emerge.
        if total_terminal > 0.85:
            winner_prob *= 0.85 / total_terminal
            error_prob *= 0.85 / total_terminal

        outcome = self.rng.random()
        reward = spec["shape"]
        done = False
        next_state = None

        # Terminal outcomes: either the player wins the point or makes an error.
        if outcome < winner_prob:
            reward += 1.0
            done = True
        elif outcome < winner_prob + error_prob:
            reward -= 1.0
            done = True
        else:
            # If the point continues, the opponent gets a reply that may change our balance.
            next_opponent_balance = spec["next_opponent_balance"]
            next_player_balance = self._opponent_reply_balance(
                spec["next_player_balance"],
                next_opponent_balance,
                spec["next_ball"],
                action
            )
            next_state = (
                phase,
                spec["next_ball"],
                next_player_balance,
                next_opponent_balance,
                self._next_stamina(stamina, spec["energy"]),
                self._next_pressure(pressure, spec["pressure_delta"]),
            )
            reward += self._state_shape(next_state)

        # Long rallies are resolved by who has the better position when max_shots is reached.
        if not done and self.shot_count >= self.max_shots:
            reward += self._tiebreak_reward(next_state)
            done = True
            next_state = None

        # Terminal states are represented as None for the algorithm code.
        if done:
            self.state = None
        else:
            self.state = next_state

        return next_state, reward, done

    def _action_spec(self, action, ball, player, opponent, stamina, pressure):
        # Each action returns:
        # winner/error probabilities, small shaping reward, next ball and balances,
        # stamina cost/recovery, and pressure movement.
        if action == "crosscourt":
            # Crosscourt is the safer rally pattern: lower risk, modest setup value.
            winner = 0.05
            error = 0.04
            shape = 0.00
            pressure_delta = -1
            energy = 1

            if ball in ("wide", "body"):
                winner += 0.03
                shape += 0.02
            if player == "defensive":
                error -= 0.01
                shape += 0.01
            if opponent == "attacking":
                winner -= 0.02
                error += 0.02
            if pressure == "high":
                error -= 0.01

            return {
                "winner": winner,
                "error": error,
                "shape": shape,
                "next_ball": self.rng.choice(["wide", "deep", "body"]),
                "next_player_balance": self._balance_shift(player, +1),
                "next_opponent_balance": self._balance_shift(opponent, -1 if ball == "wide" else 0),
                "energy": energy,
                "pressure_delta": pressure_delta
            }

        if action == "down_the_line":
            # Down the line is a high-risk attacking shot, best from wide/attacking spots.
            winner = 0.13
            error = 0.15
            shape = -0.01
            pressure_delta = 1
            energy = 2

            if ball == "wide":
                winner += 0.10
                error -= 0.03
                shape += 0.04
            if ball == "short":
                winner += 0.06
            if player == "attacking":
                winner += 0.08
                error -= 0.04
            if player == "defensive":
                winner -= 0.05
                error += 0.08
            if opponent == "defensive":
                winner += 0.06
            if stamina == "low":
                error += 0.07
            if pressure == "high":
                winner += 0.04
                error += 0.04

            return {
                "winner": winner,
                "error": error,
                "shape": shape,
                "next_ball": self.rng.choice(["deep", "center", "wide"]),
                "next_player_balance": "attacking",
                "next_opponent_balance": self._balance_shift(opponent, -1),
                "energy": energy,
                "pressure_delta": pressure_delta
            }

        if action == "heavy_topspin":
            # Heavy topspin is a pressure-building shot that can push the opponent back.
            winner = 0.08
            error = 0.08
            shape = 0.01
            pressure_delta = 0
            energy = 2

            if ball in ("deep", "center"):
                winner += 0.03
                error -= 0.02
            if ball == "short":
                winner -= 0.03
                error += 0.03
            if player == "attacking":
                winner += 0.04
            if player == "defensive":
                error += 0.03
            if opponent == "defensive":
                shape += 0.02
            if stamina == "low":
                winner -= 0.03
                error += 0.06

            return {
                "winner": winner,
                "error": error,
                "shape": shape,
                "next_ball": self.rng.choice(["deep", "deep", "center", "wide"]),
                "next_player_balance": self._balance_shift(player, +1),
                "next_opponent_balance": self._balance_shift(opponent, -1),
                "energy": energy,
                "pressure_delta": pressure_delta
            }

        if action == "drop_shot":
            # Drop shots are strong after pulling the opponent defensive, but risky otherwise.
            winner = 0.06
            error = 0.13
            shape = -0.02
            pressure_delta = 1
            energy = 1

            if ball == "short":
                winner += 0.08
                error -= 0.04
            if opponent == "defensive":
                winner += 0.14
                error -= 0.02
                shape += 0.05
            if opponent == "attacking":
                winner -= 0.04
                error += 0.04
            if player == "attacking":
                winner += 0.05
                error -= 0.03
            if player == "defensive":
                winner -= 0.03
                error += 0.06
            if stamina == "low":
                error += 0.04
            if pressure == "high":
                winner += 0.03
                error += 0.04

            return {
                "winner": winner,
                "error": error,
                "shape": shape,
                "next_ball": "short",
                "next_player_balance": "attacking",
                "next_opponent_balance": self._balance_shift(opponent, -1),
                "energy": energy,
                "pressure_delta": pressure_delta
            }

        # defensive_slice
        # Slice is mostly a survival/recovery action: low winner chance, low energy cost.
        winner = 0.02
        error = 0.05
        shape = -0.01
        pressure_delta = -2
        energy = -1

        if player == "defensive":
            error -= 0.02
            shape += 0.04
        if stamina == "low":
            error -= 0.01
            shape += 0.02
        if ball in ("body", "deep"):
            error -= 0.01
        if opponent == "attacking":
            winner += 0.02
            shape += 0.01
        if pressure == "high":
            error -= 0.02

        return {
            "winner": winner,
            "error": error,
            "shape": shape,
            "next_ball": self.rng.choice(["deep", "center", "body"]),
            "next_player_balance": self._balance_shift(player, +1),
            "next_opponent_balance": self._balance_shift(opponent, +1),
            "energy": energy,
            "pressure_delta": pressure_delta
        }

    def _state_shape(self, state):
        # Small non-terminal reward nudges learning toward good tactical positions.
        _, ball, player, opponent, stamina, pressure = state
        reward = -0.01

        if player == "attacking":
            reward += 0.025
        elif player == "defensive":
            reward -= 0.025

        if opponent == "defensive":
            reward += 0.025
        elif opponent == "attacking":
            reward -= 0.025

        if ball == "short" and player == "attacking":
            reward += 0.02
        if stamina == "low":
            reward -= 0.02
        if pressure == "high":
            reward -= 0.015

        return reward

    def _tiebreak_reward(self, state):
        # If the rally hits max_shots, estimate who wins from the current position.
        if state is None:
            return 0.0

        _, ball, player, opponent, stamina, pressure = state
        score = 0.0

        if player == "attacking":
            score += 0.22
        elif player == "defensive":
            score -= 0.22

        if opponent == "defensive":
            score += 0.22
        elif opponent == "attacking":
            score -= 0.22

        if ball == "short":
            score += 0.08
        elif ball == "deep":
            score += 0.03

        if stamina == "low":
            score -= 0.12
        if pressure == "high":
            score -= 0.10

        win_prob = self._clamp(0.5 + score, 0.08, 0.92)
        return 1.0 if self.rng.random() < win_prob else -1.0

    def _next_stamina(self, stamina, energy):
        # Positive energy means fatigue risk; negative energy means possible recovery.
        index = self.stamina_states.index(stamina)

        if energy > 0:
            fatigue_chance = 0.22 * energy + 0.025 * self.shot_count
            if self.rng.random() < fatigue_chance:
                index = min(index + 1, len(self.stamina_states) - 1)
        elif energy < 0 and self.rng.random() < 0.35:
            index = max(index - 1, 0)

        return self.stamina_states[index]

    def _next_pressure(self, pressure, delta):
        # Risky attacking shots tend to raise pressure; safer shots can lower it.
        index = self.pressure_states.index(pressure)

        # Very long rallies add pressure over time even without a risky action.
        long_rally_pressure = 1 if self.shot_count >= 8 and self.shot_count % 4 == 0 else 0
        index += delta + long_rally_pressure
        index = max(0, min(index, len(self.pressure_states) - 1))
        return self.pressure_states[index]

    def _balance_shift(self, balance, direction):
        # Move between defensive -> neutral -> attacking while staying inside the list.
        index = self.balance_states.index(balance)
        index = max(0, min(index + direction, len(self.balance_states) - 1))
        return self.balance_states[index]

    def _opponent_reply_balance(self, player_balance, opponent_balance, ball, action):
        # The opponent's reply can undo some of the advantage created by our shot.
        pressure_roll = self.rng.random()

        if opponent_balance == "attacking":
            # An attacking opponent is dangerous and often puts us on defense.
            if action == "defensive_slice":
                return "defensive" if pressure_roll < 0.70 else "neutral"
            if pressure_roll < 0.45:
                return self._balance_shift(player_balance, -2)
            if pressure_roll < 0.75:
                return self._balance_shift(player_balance, -1)
            return player_balance

        if opponent_balance == "neutral":
            # Neutral opponents sometimes pressure deep/body balls.
            if ball == "body" and pressure_roll < 0.35:
                return self._balance_shift(player_balance, -1)
            if ball == "deep" and pressure_roll < 0.20:
                return self._balance_shift(player_balance, -1)
            return player_balance

        # Defensive opponents usually give us time, but a bad drop shot can backfire.
        if ball == "short" and action == "drop_shot" and pressure_roll < 0.20:
            return "defensive"
        if pressure_roll < 0.20:
            return self._balance_shift(player_balance, +1)

        return player_balance

    def _clamp(self, value, low, high):
        # Keep probabilities and computed scores inside reasonable bounds.
        return max(low, min(value, high))
