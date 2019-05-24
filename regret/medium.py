import numpy as np
import pandas as pd

class RPS:
    actions = ["rock", "paper", "scissors"]
    n_actions = 3
    payoff = pd.DataFrame([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ], columns=actions, index=actions)

class Player:
    def __init__(self, name):
        self.name = name
        self.strategy, self.avg_strategy, self.strategy_sum, self.regret_sum = np.zeros((4, RPS.n_actions))

    def __repr__(self):
        return self.name

    def update_strategy(self):
        """
        changes strategy via regret matching algorithm
        """
        self.strategy = np.copy(self.regret_sum)
        