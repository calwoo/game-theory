"""
stupid implementation of a normal form game, and eventual
computations of nash equilibria
"""

import numpy as np

from utils import *

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

    def equilibria(self):
    """
    find all Nash equilibria of 2-player normal form game via 
    support enumeration.
    """
    n, m = self.payoff_A.shape
    # create supports
    for support_u in (s for s in get_powerset(n) if len(s) > 0):
        for support_v in (s for s in get_powerset(m) if len(s) == len(support_u)):
            # get candidate for nash equilibrium
            try:
                candidate_A = equilibrium_candidate(self.payoff_A, support_u, support_v)
                candidate_B = equilibrium_candidate(self.payoff_B.T, support_v, support_u)
                # check if best response to one another
                if is_best_response(self.payoff_B.T, candidate_A, candidate_B) and \
                    is_best_response(self.payoff_A, candidate_B, candidate_A):
                        return [candidate_A, candidate_B]
            except:
                continue
    return "no Nash equilibria found"


################
# testing ground

if __name__ == "__main__":
    A = [[3, 0], [5, 1]]
    B = [[3, 5], [0, 1]]

    prison = game(A, B)
    print(prison)