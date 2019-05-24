import numpy as np

def draw(weights):
    """
    randomly sample an index based on weights
    """
    # sample random number uniformly from total weight
    rnd_num = np.random.uniform(low=0.0, high=sum(weights))
    # find index that random number selects
    for i, w in enumerate(weights):
        rnd_num -= w
        if rnd_num <= 0:
            return i

def MWUA(objects, observer, payoffs, lr, num_epochs):
    num_objects = len(objects)
    # set initial weights
    weights = np.ones(num_objects)
    for t in range(num_epochs):
        # sample according to weight proportionality
        obj_idx = draw(weights)
        obj = objects[obj_idx]
        # observe outcome of object
        outcome = observer(t, weights, obj)
        payoff = payoffs(obj, outcome)
        # update weights
        for i in range(len(weights)):
            weights[i] *= 1 + learning_rate * reward(objects[i], outcome)

