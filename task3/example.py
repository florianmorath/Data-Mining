import numpy as np
from numpy.random import randint, choice, shuffle
from numpy.linalg import norm

# Task parameters
D = 250
N = 3000
k = 200

# Coreset construction parameters
alpha = 16*(np.log(k) + 2)
m = 5000 # coreset size per mapper

# k-means parametes
restarts = 5
epochs = 15
chunk_size = 100


def mapper(key, value):
    # key: None
    # value: array of 10000 training images, 250 features each

    X = value

    # D^2-sampling
    distances = np.empty((0, N)) # 2D array containing the distances of each sampled point to every point in X
    b = X[randint(N)] # sample the first point with uniform probability
    for i in range(1, k):
        distances = np.vstack((distances, norm(X - b, axis=1)**2)) # add distance of every point in X to the new sampled point b
        min_distances = distances.min(axis=0) # compute the current minimum distance of every point in X to the sampled points
        p = min_distances/min_distances.sum()
        b = X[choice(N, p=p)] # sample point with probability proportional to the minimum distance to the already sampled points

    # Coreset construction using importance sampling
    Bi_indexes = distances.argmin(axis=0) # array containing the index of the closest sample point for each point in X
    Bi = [[] for i in range(k)] # Bi[j] set of indexes of points from X closest to the jth sampled point
    for i in range(N):
        Bi[Bi_indexes[i]].append(i)
    # Compute the sensitivity for every point in X
    c_phi = min_distances.sum()/N
    min_distances_sum_Bi = [sum([min_distances[x] for x in Bi[i]]) for i in range(k)]
    Bi_part = [2.0*alpha*min_distances_sum_Bi[i]/(len(Bi[i])*c_phi) + 4.0*N/len(Bi[i]) if Bi[i] else 0 for i in range(k)]
    s = np.array([alpha*min_distances[x]/c_phi + Bi_part[Bi_indexes[x]] for x in range(N)])
    p = s/s.sum() # compute sampling probability for every point in X
    w = 1.0/(m*p) # compute weight for every point in X
    coreset = [(X[s], w[s]) for s in choice(N, size=m, p=p)] # sample m points using the calculated probabilities and construct the coreset

    print len(coreset)
    yield 0, coreset
    #yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    coreset = values

    # Perform k-means with multiple random restarts
    # Initialize the centers for each restart by sampling from the coreset with uniform probability
    centers = np.array([[coreset[i][0] for i in choice(len(coreset), size=k)] for r in range(restarts)])

    for e in range(epochs):
        point_sum = [[0 for i in range(k)] for r in range(restarts)] # sum of the points closest to each center for each restart
        point_count = [[0 for i in range(k)] for r in range(restarts)] # number of points closest to each center for each restart
        for x,w in coreset:
            c = norm(centers - x, axis=2).argmin(axis=1) # find the closest center to the point x for each restart
            for r in range(restarts):
                point_sum[r][c[r]] += w*x # update the sum
                point_count[r][c[r]] += w # update the count
        centers = np.array([[point_sum[r][i]/point_count[r][i] if point_count[r][i] != 0 else centers[r][i] for i in range(k)] for r in range(restarts)]) # update the centers

    # Split coreset points and weights into chunks to limit memory usage in NQE()
    coreset_chunks = [coreset[i:i + chunk_size] for i in range(0, len(coreset), chunk_size)]
    X_chunks = np.array([[v[0]for v in c] for c in coreset_chunks])
    W_chunks = np.array([[v[1]for v in c] for c in coreset_chunks])

    # Compute the normalized quantization error on the coreset for a centers configuration
    def NQE(centers):
        score = 0.0
        for i in range(len(coreset_chunks)):
            score += (W_chunks[i]*(norm(X_chunks[i][:,np.newaxis,:] - centers, axis=2)**2).min(axis=1)).sum()
        return score/(N*10)

    # Select the result of the repetition with the smallest error
    scores = [NQE(centers[r]) for r in range(restarts)]
    print "Restart scores:", scores
    print "Best score:", min(scores)
    min_index = np.array(scores).argmin()

    yield centers[min_index]
    #yield np.random.randn(200, 250)
