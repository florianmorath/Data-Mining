# Data Mining: Learning from Large Data Sets

Data mining is about gaining insights from massive and high-dimensional data sets.  
This repository contains large scale machine learning tasks.

## Task 1
Find similar pages among a large collection of pages.

Model implemented: The overall idea is that a mapper outputs a page with his band hashes. The reducer receives pages with equal band hashes and takes all possible pairs over those candidates, computes the exact Jaccard similarity and compares it to the given threshold. Based on Locality-Sensitive Hashing (Min Hashing).


## Task 2
Classification of 400-dimensional images.

Model implemented: MapReduce program where each mapper runs SGD over hinge loss objective on the subset of images assigned to it and produces a weight vector. After that all the weight vectors are gathered by a single reducer. The reducer computes the average of the weight vectors which results in a single final weight vector that can be used to classify the images. In more details, the mapper does linear classification on nonlinearly transformed features. The nonlinear feature transformation is done by transforming features to random Fourier features (RFF). The feature space is transformed from 400 to 2000 dimensions. An extra dimension with a value of one is added s.t. the bias can be integrated in the weight vector. Since we have chosen the Cauchy kernel for the RFF the omega sample is sampled from a Laplace distribution. To stochastically optimise over the hinge loss we have chosen the ADAM algorithm which is based on adaptive estimates of lower-order moments. 

## Task 3
Clustering of 250-dimensional images.

Model implemented: MapReduce program where each mapper receives a subset of images and constructs a coreset of those images (using D^2 sampling and importance sampling). The reducer receives all the coresets of the mappers and takes the union. The union is the final coreset on which k-means is performed. The resulting centers are returned. 

## Task 4
Recommender system for articles. 

Model implemented: LinUCB with disjoint linear models.



