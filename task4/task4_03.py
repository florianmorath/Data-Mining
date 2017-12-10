import numpy as np

d = 6   # number of dimensions of user feature vector

# running statistics of LinUCB algorithm
A = {}
b = {}

Ainv = {}
weight_vectors = {}

# LinUCB parameters
delta = 0.06
alpha = 1 + np.sqrt(np.log(2/delta)/2)


x = np.zeros((d, 1))

# recommend article that maximes UCB 
def recommend(time, user_features, choices):
    global recommendation, x
    x = np.array(user_features).reshape(d, 1)
    
    max_set = False
    for c in choices:
        if c not in A:
            b[c] = np.zeros((d, 1))
            A[c] = np.identity(d)
            Ainv[c] = np.linalg.inv(A[c])
            weight_vectors[c] = (Ainv[c].dot(b[c])).T

        # compute UCB for action c
        ucb_score = weight_vectors[c].dot(x) + alpha * np.sqrt(x.T.dot(Ainv[c]).dot(x))

        # find article with the maximum UCB
        if max_set == False or ucb_score > max_ucb_score:
            max_set = True
            max_ucb_score = ucb_score
            recommendation = c

    # recommend article with maximum UCB
    return recommendation

# update running statistics based on observation/reward
def update(reward):
    if reward == -1:
        return
    else:
        # adjust the rewards
        if reward == 0:
            reward = -10
        else:
            reward = 271

        # LinUCB algorithm
        b[recommendation] += reward*x
        A[recommendation] += x.dot(x.T)
        Ainv[recommendation] = np.linalg.inv(A[recommendation])
        weight_vectors[recommendation] = (Ainv[recommendation].dot(b[recommendation])).T

# disjoint LinUCB doesn't care about article features
def set_articles(articles):
    return