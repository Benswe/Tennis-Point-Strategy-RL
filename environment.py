import random


class TennisEnv:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.shot_count = 0
        self.max_shots = 20

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
    # every episode should start at a serving state
    def reset(self):
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
    # action selection, this will be very useful for MC control, sarsa and Q-learning
    def action_selection(self, state):
        return self.rally_actions

    def step(self, action):
        phase, ball_position, player_balance, opponent_balance, stamina, pressure = self.state

        reward = -0.01
        done = False

    # -------
    # Heavy topspin
    # -------
        if action == "heavy_topspin":

            outcome = random.random()

            # 10% winner
            if outcome < 0.10:
                reward = 1
                done = True
                next_state = None
            # 10% error
            elif outcome < 0.20:
                reward = -1
                done = True
                next_state = None
            else:
                next_state = ("rally",
                    random.choice(["deep", "center"]),
                    player_balance,
                    "defensive",
                    stamina,
                    pressure)
            self.shot_count += 1
        
    # ------
    # Crosscourt 
    # ------
        elif action == "crosscourt":

            outcome = random.random()

            if outcome < 0.08: # winner
                reward = 1
                done = True
                next_state = None
            
            elif outcome < 0.13: # error
                reward = -1
                done = True
                next_state = None
            
            else:
                next_state = (
                    "rally",
                    random.choice(["wide", "body"]),
                    player_balance,
                    opponent_balance,
                    stamina,
                    pressure
                )
            self.shot_count += 1
        if self.shot_count >= self.max_shots:
            done = True

            if random.random() < 0.5:
                reward = 1
            else:
                reward = -1
            next_state = None


        if not done:
            self.state = next_state
        return next_state, reward, done
env = TennisEnv()

state = env.reset()