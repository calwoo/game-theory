import numpy as np
import matplotlib.pyplot as plt

class RPSPlayer:
    """
    a rock-paper-scissors player
    """
    def __init__(self):
        self.num_actions = 3
        self.regret_sum = np.array([0.0, 0.0, 0.0])
        self.strategy = np.array([0.0, 0.0, 0.0])
        self.strategy_sum = np.array([0.0, 0.0, 0.0])

    def utility(self, p1, p2):
        payoff = [
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]]
        return payoff[p1][p2]

    def get_strategy(self):
        """
        get current (mixed) strategy via regret-matching
        """
        self.strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(self.strategy)
        for i in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[i] /= normalizing_sum
            else:
                self.strategy[i] = 1.0 / self.num_actions
        self.strategy_sum += self.strategy
        return self.strategy

    def get_action(self):
        """
        choose an action randomly using weights in strategy
        """
        prob = np.random.random()
        cumulative_prob = 0
        for idx in range(self.num_actions):
            cumulative_prob += self.strategy[idx]
            if prob < cumulative_prob:
                return idx

    def get_average_strategy(self):
        """
        compute average strategy from strategy_sum
        """
        avg_strategy = np.zeros(self.num_actions)
        normalizing_sum = np.sum(self.strategy_sum)
        for i in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[i] = self.strategy_sum[i] / normalizing_sum
            else:
                avg_strategy[i] = 1.0 / self.num_actions
        return avg_strategy

##########
# play RPS

def train_2p(player1, player2, iterations):
    num_actions = player1.num_actions
    action_utility_1 = np.zeros(num_actions)
    action_utility_2 = np.zeros(num_actions)
    for i in range(iterations):
        # get regret-matched strategy actions
        strategy = player1.get_strategy()
        action = player1.get_action()
        opp_strategy = player2.get_strategy()
        opp_action = player2.get_action()
        # compute action utilities
        for a in range(num_actions):
            action_utility_1[a] = player1.utility(a, opp_action)
            action_utility_2[a] = player2.utility(a, action)
        # accumulate action regrets
        for a in range(num_actions):
            player1.regret_sum[a] += action_utility_1[a] - action_utility_1[action]
            player2.regret_sum[a] += action_utility_2[a] - action_utility_2[opp_action]
        print(i, player1.get_average_strategy(), player2.get_average_strategy())

if __name__ == "__main__":
    player1 = RPSPlayer()
    player2 = RPSPlayer()
    train_2p(player1, player2, 10000)
    print(player1.get_average_strategy(), player2.get_average_strategy())