import numpy as np
from numpy.random import randint, choice, shuffle
from numpy.linalg import norm

# Fixed params based on input size to a mapper
D = 250     # each image has 250 dimensions
N = 3000    # each mapper receives 3000 images

# k-means parameters (performed in reducer)
k = 200     # number of centers of k-means
restarts = 5
epochs = 15

# Coreset construction parameters (perfomed in mapper)
alpha = (np.log2(k) + 1)
m = 1000    # size of the coreset each mapper produces

# make algorithm deterministic: easyier to test based on scores received
np.random.seed(seed = 3333) # 2017, 1

def mapper(key, value):
    # key: None
    # value: array of 3000 training images, 250 features each

    images = value

    # do D^2-sampling -> results in a bicriteria approximation
    n = randint(N) # random index between 0 and N (u.a.r)
    b = images[n] # sample the first point (row vector)
    dist = np.empty((0, N)) # contains the distances of each sampled point to every point in images
    # first row of distances contains the distances of each point in images to the first sampled point
    # second row contains distances of each point to the second sampled point and so on
    b = images[randint(N)] # sample the first point u.a.r (is a row vector)
    for i in range(1, k):
        # for each data point (row in the images matrix), subtract point b and calculate the squared norm of the distance
        # add distance of every point in images, as a new row to the dist matrix
        norms = norm(images - b, axis=1)**2
        dist = np.vstack((dist, norms))
        current_min_dist = dist.min(axis=0) # holds for every point of the data set the current min distance to the sampled points (centers)
        probabilities = current_min_dist/current_min_dist.sum()
        b = images[choice(N, p=probabilities)] # sample a new point with probability proportional to the minimum distance to the already sampled points

    # use importance sampling to construct a coreset
    Bi_indexes = dist.argmin(axis=0) # array containing the index of the closest sampled center for each point in images
    Bi = [[] for i in range(k)] # Bi[j] contains the indexes of the points from images which are closest to the jth sampled point
    for i in range(N):
        #fill Bi
        Bi[Bi_indexes[i]].append(i)
    # Compute the sensitivity for every point in images
    c_phi = current_min_dist.sum()/N
    sum_Bi_min_dist = [sum([current_min_dist[x] for x in Bi[i]]) for i in range(k)]
    Bi_part = [2.0*alpha*sum_Bi_min_dist[i]/(len(Bi[i])*c_phi) + 4.0*N/len(Bi[i]) if Bi[i] else 0 for i in range(k)] # second summand of sampling probability
    sensitivity = np.array([alpha*current_min_dist[x]/c_phi + Bi_part[Bi_indexes[x]] for x in range(N)])
    probabilities = sensitivity/sensitivity.sum() # compute sampling probability for every point in images
    weights = 1.0/(m*probabilities) # compute weight for every point in images
    coreset = [(images[s], weights[s]) for s in choice(N, size=m, p=probabilities)] # sample m points using the calculated probabilities and construct the coreset

    print len(coreset)
    yield 0, coreset


def reducer(key, values):
    # key: key from mapper used to aggregate (= 0 for all mappers)
    # values: list of all value for that key (list of (image, weight) pairs)
    coreset = values    # union of all coresets received from reducers

    # k-means algorithm performed on coreset
    # each restart has a different initial guess of the k centers (chosen u.a.r)
    # centers.shape = (restarts, k, D)
    centers = np.array([[coreset[i][0] for i in choice(len(coreset), size=k)] for r in range(restarts)])

    for e in range(epochs):
        # sum of the points closest to each center for each restart
        point_sum = [[0 for i in range(k)] for r in range(restarts)]
        # number of points closest to each center for each restart
        point_count = [[0 for i in range(k)] for r in range(restarts)]
        for x,w in coreset:
            # find the closest center to the point x for each restart
            closests_centers = norm(centers - x, axis=2).argmin(axis=1)
            for r in range(restarts):
                # update the sum
                point_sum[r][closests_centers[r]] += w*x
                # update the count
                point_count[r][closests_centers[r]] += w
        # update the centers
        centers = np.array([[point_sum[r][i]/point_count[r][i] if point_count[r][i] != 0 else centers[r][i] for i in range(k)] for r in range(restarts)])

    # Compute the normalized quantization error on the coreset for a specific set of centers
    X = [v[0]for v in coreset]
    W = [v[1]for v in coreset]

    def NQE(centers):
        score = 0
        for i in range(len(X)):
            distance = 10000 # np.inf
            for c in centers:
                distance = min(distance, norm(X[i]-c)**2)
            score += distance * W[i]
        return score/(N*10)

    # Choose the centers which result in the best NQE
    score_candidates = [NQE(centers[r]) for r in range(restarts)]
    print "Scores candidates:", score_candidates
    print "Best score:", min(score_candidates)
    i_min = np.array(score_candidates).argmin()

    yield centers[i_min]
