import numpy as np

d = 6
A = {}
b = {}
# store infrequently updated computationally expensive arrays
Ainv = {}
theta = {}

#LinUCB parameters
delta = 0.01
alpha = 1 + np.sqrt(np.log(2/delta)/2)
#alpha = 5
# reward scaling paramenters
r0 = -1
r1 = 20
print "alpha = {}, r0 = {}, r1 = {}".format(alpha, r0, r1)

# store current recommended article and current user features
chosen_arm = 0
x = np.zeros((d, 1))

# LinUCB does not use article features
def set_articles(articles):
    return


# update recommended article arrays according to observed reward
def update(reward):
    if reward == -1:
        return
    else:
        # scale the reward from 0/1 to r0/r1
        if reward == 0:
            reward = r0
        else:
            reward = r1
        # update arrays
        A[chosen_arm] += x.dot(x.T)
        b[chosen_arm] += reward*x
        Ainv[chosen_arm] = np.linalg.inv(A[chosen_arm])
        theta[chosen_arm] = (Ainv[chosen_arm].dot(b[chosen_arm])).T


# recommend article from the available choices
def recommend(time, user_features, choices):
    global chosen_arm, x
    x = np.array(user_features).reshape(d, 1)

    first = 1
    for arm in choices:
        # create arrays for new article 
        if arm not in A:
            A[arm] = np.identity(d)
            b[arm] = np.zeros((d, 1))
            Ainv[arm] = np.linalg.inv(A[arm])
            theta[arm] = (Ainv[arm].dot(b[arm])).T

        # compute UCB
        p = theta[arm].dot(x) + alpha*np.sqrt(x.T.dot(Ainv[arm]).dot(x))

        # find article with the maximum UCB
        if first == 1 or p > max_p:
            first = 0
            max_p = p
            chosen_arm = arm

    # recommend article with maximum UCB
    return chosen_arm

