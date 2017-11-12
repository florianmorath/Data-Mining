import random
import numpy as np

input_dim = 400  # Number of features of every image
N = 2000    # Each mapper receives 2000 images
epochs = 100    # Number of epochs

alpha = 0.05    # step-size/learning rate

# exponential decay rates for the moment estimates (set to default values)
beta1 = 0.9
beta2 = 0.999

epsilon = 10**(-9)

RFF_count = 2500  # Number of random fourier features = number of samples

# Omega and beta samples for Random Fourier Features with Laplacian kernel
sigma = 2
np.random.seed(seed = 1000)
omega_samples = np.random.laplace(scale=sigma, size=(RFF_count, input_dim))
beta_samples = np.random.uniform(low=0.0, high=2.0*np.pi, size=RFF_count)


def transform(X):
    """Transform the original features.

    :param X: 2D numpy array representing all the images
    :return: transformed features
    """

    x_transformed = np.sqrt(2.0 / RFF_count) * np.cos(np.dot(X, omega_samples.T) + beta_samples)

    # Add one dimension to the data with a value of 1
    if len(np.shape(x_transformed)) == 1:
        return np.append(x_transformed, [1])
    else:
        shape = np.shape(x_transformed)
        out = np.zeros((shape[0], shape[1] + 1))
        for i in range(shape[0]):
            out[i] = np.append(x_transformed[i], [1])
        return out


def mapper(key, value):
    """The mapper optimizes over the hinge loss objective using ADAM to obtain a classifier w.

    :param key: None
    :param value: 2D numpy array which represents a subset of the images (2000 images)
    :return: a tuple (0, w) where w is the weight vector computed by this mapper
    """

    # The matrix represents the 400 features of the 2000 images provided in every mapper call
    matrix = transform(np.array([image.split()[1:] for image in value]).astype(float))
    new_shape = matrix.shape[1]
    # Label for each image
    y = np.array([float(image.split()[0]) for image in value])
    # Initialization of the weight vector
    w = np.zeros(new_shape)
    # Initialization of the first moment vector
    moment_vector_1 = np.zeros(new_shape)
    # Initialization of the second moment vector
    moment_vector_2 = np.zeros(new_shape)
    # Perform ADAM algorithm
    indexes = range(N)
    for e in range(epochs):
        random.shuffle(indexes)
        timestep = 0
        for t in range(N):
            timestep += 1
            index = indexes[t]
            # Get the gradient of the hinge loss function
            if y[index]*np.inner(w, matrix[index]) < 1:
                gradient = -y[index]*matrix[index]
            else:
                gradient = 0
            # Update biased first moment estimate
            moment_vector_1 = beta1*moment_vector_1 + (1.0 - beta1)*gradient
            # Update biased second raw moment estimate
            moment_vector_2 = beta2*moment_vector_2 + (1.0 - beta2)*(gradient**2)
            # Compute biased-corrected first moment estimate
            moment_vector_1_corr = moment_vector_1/(1.0 - beta1**timestep)
            # Compute bias-corrected second raw moment estimate
            moment_vector_2_corr = moment_vector_2/(1.0 - beta2**timestep)
            # Update weight vector
            w = w - alpha*moment_vector_1_corr/(np.sqrt(moment_vector_2_corr) + epsilon)
    yield 0, w


def reducer(key, values):
    """

    :param key: 0
    :param values: list of weight vectors
    :return: Output the average of the weight vectors
    """
    w = 0
    for value in values:
        w += value

    yield w/float(len(values))

