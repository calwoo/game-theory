import numpy as np
import itertools

def support(sigma, tol=1e-10):
    """
    get support of a strategy
    """
    sigma = np.array(sigma)
    return np.where(sigma > tol)[0]

def is_best_response(A, sigma_A, sigma_B):
    """
    check if a given strategy sigma_A is the best response to opponent
    strategy sigma_B
    """
    # set as np.arrays
    sigma_A, sigma_B = np.array(sigma_A), np.array(sigma_B)
    spt = support(sigma_A)
    best = True
    Ay = A @ sigma_B
    max_Ay = max(Ay)
    for i in spt:
        if Ay[i] != max_Ay:
            best = False
    return best

def indifference(payoff, support_u, support_v):
    """
    compute indifference matrix for a given support.
    for reference, just check out "algorithmic game theory"
    by nisan et. al
    """
    n = len(payoff[0])
    M = np.zeros((len(support_u) + n - len(support_v), n), dtype=np.float64)
    # condition 1: best response condition for given support
    for i in range(len(support_u) - 1):
        M[i] = payoff[support_u[i]] - payoff[support_u[i+1]]
    # condition 2: ensure support is in S(v)
    complement_Sv = [j for j in range(n) if j not in support_v]
    for k, i in enumerate(range(len(support_u) - 1, len(support_u) + n - len(support_v) - 1)):
        M[i][complement_Sv[k]] = 1
    # condition 3: last row is all 1's to ensure solution is a probability vector
    M[-1] = np.ones(n)
    return M

def equilibrium_candidate(payoff, support_u, support_v):
    """
    using indifference, compute a candidate for the nash equilibrium
    given fixed supports
    """
    M = indifference(payoff, support_u, support_v)
    b = np.zeros(M.shape[1])
    b[-1] = 1
    # return np.linalg.inv(M) @ b
    return np.linalg.solve(M, b)

def get_powerset(n):
    """
    get powerset of n using itertools
    """
    return itertools.chain.from_iterable(
        itertools.combinations(range(n), r) for r in range(n + 1))