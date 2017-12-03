import numpy as np
from numpy.random import randint, choice, shuffle
from numpy.linalg import norm

# Fixed params based on input size to a mapper
D = 250     # each image has 250 dimensions
N = 3000    # each mapper receives 3000 images

# Coreset construction parameters (perfomed in mapper)
alpha = (np.log2(k) + 1)
m = 1000    # size of the coreset each mapper produces

# k-means parameters (performed in reducer)
k = 200     # number of centers of k-means
restarts = 5
epochs = 15
chunk_size = 100

# make algorithm deterministic: easyier to test based on scores received
np.random.seed(seed = 3333) # 2017, 1

def mapper(key, value):
    # key: None
    # value: array of 3000 training images, 250 features each

    images = value

    # do D^2-sampling -> results in a bicriteria approximation
    distances = np.empty((0, N)) # 2D array containing the distances of each sampled point to every point in images
    # first row of distances contains the distances of each point in images to the first sampled point
    # second row contains distances of each point to the second sampled point and so on
    b = images[randint(N)] # sample the first center u.a.r (is a row vector)
    for i in range(1, k):
        # for each data point (row in the images matrix), subtract point b and calculate the squared norm of the distance
        # add distance of every point in images to the new sampled point b
        distances = np.vstack((distances, norm(images - b, axis=1)**2))
        min_distances = distances.min(axis=0) # compute for every point of the data set the current minimum distance to the samled points (centers)
        p = min_distances/min_distances.sum()
        b = images[choice(N, p=p)] # sample point with probability proportional to the minimum distance to the already sampled points

    # Coreset construction using importance sampling
    Bi_indexes = distances.argmin(axis=0) # array containing the index of the closest sample point for each point in images
    Bi = [[] for i in range(k)] # Bi[j] set of indexes of points from images closest to the jth sampled point
    for i in range(N):
        Bi[Bi_indexes[i]].append(i)
    # Compute the sensitivity for every point in images
    c_phi = min_distances.sum()/N
    min_distances_sum_Bi = [sum([min_distances[x] for x in Bi[i]]) for i in range(k)]
    Bi_part = [2.0*alpha*min_distances_sum_Bi[i]/(len(Bi[i])*c_phi) + 4.0*N/len(Bi[i]) if Bi[i] else 0 for i in range(k)] # second summand of sampling probability
    s = np.array([alpha*min_distances[x]/c_phi + Bi_part[Bi_indexes[x]] for x in range(N)])
    p = s/s.sum() # compute sampling probability for every point in images
    w = 1.0/(m*p) # compute weight for every point in images
    coreset = [(images[s], w[s]) for s in choice(N, size=m, p=p)] # sample m points using the calculated probabilities and construct the coreset

    print len(coreset)
    yield 0, coreset


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    coreset = values    # union of all coresets received from reducers

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
