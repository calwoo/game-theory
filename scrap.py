"""
stupid implementation of a normal form game, and eventual
computations of nash equilibria
"""

import numpy as np

class game:
    """
    2 player normal form game
    """
    def __init__(self, payoff_A, payoff_B=None):
        self.payoff_A = np.array(payoff_A)
        if payoff_B is None:
            # zero-sum game
            self.payoff_B = -self.payoff_A
        else:
            self.payoff_B = np.array(payoff_B)
        
    def __repr__(self):
        return "player 1 payoff:\n" + str(self.payoff_A) + "\n\n" + \
               "player 2 payoff:\n" + str(self.payoff_B)
    
    def payoff(self, sigma_1, sigma_2):
        expected_payoff_1 = sigma_1.T @ self.payoff_A @ sigma_2
        expected_payoff_2 = sigma_1.T @ self.payoff_B @ sigma_2
        return np.array([expected_payoff_1, expected_payoff_2])


################
# testing ground

if __name__ == "__main__":
    A = [[3, 0], [5, 1]]
    B = [[3, 5], [0, 1]]

    prison = game(A, B)
    print(prison)