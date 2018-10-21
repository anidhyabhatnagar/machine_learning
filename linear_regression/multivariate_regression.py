# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:12:31 2018

@author: Anidhya Bhatnagar

"""

import pandas as pd
import numpy as np

# Initializing variables
learning_rate = 0.0005
bias = 0.50
iterations = 1000

# Reading the csv data file
ad_data = pd.read_csv('../datasets/advertising.csv', header=None)
features = ad_data[[1, 2, 3]]
features = np.array(features)
targets = ad_data[[4]]
targets = np.array(targets)


W1 = 0.0
W2 = 0.0
W3 = 0.0

weights = np.array([
    [W1],
    [W2],
    [W3]
])


def normalize(features):
    
    for feature in features.T:
        fmean = np.mean(feature)
        frange = np.amax(feature) - np.amin(feature)

        #Vector Subtraction
        feature -= fmean

        #Vector Division
        feature /= frange
    
    return features


def predict(features, weights):
  return np.dot(features, weights)


def cost_function(features, targets, weights):

    N = len(targets)

    predictions = predict(features, weights)

    # Matrix math lets use do this without looping
    sq_error = (predictions - targets)**2

    # Return average squared error among predictions
    return 1.0/(2*N) * sq_error.sum()


def update_weights_vectorized(X, targets, weights, learning_rate):

    companies = len(X)

    #1 - Get Predictions
    predictions = predict(X, weights)

    #2 - Calculate error/loss
    error = targets - predictions

    #3 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  error matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(-X.T,  error)

    #4 Take the average error derivative for each feature
    gradient /= companies

    #5 - Multiply the gradient by our learning rate
    gradient *= learning_rate

    #6 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights


def train(features, weights, bias, learning_rate, iterations):
    cost_history = []

    for i in range(iters):
        weights = update_weights_vectorized(features, targets, weights, 
                                            learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(features, targets, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print("iterations: " + str(i) + " cost: " + str(cost))

    return cost_history


features = normalize(features)

cost = train(features, weights, bias, learning_rate, iterations)
print(weights)

