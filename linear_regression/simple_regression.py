# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:12:31 2018

@author: Anidhya Bhatnagar

"""

import pandas as pd
import numpy as np

# Initializing variables
weight = 1
bias = 0.50
learning_rate = 0.0001
iterations = 1000

# Reading the csv data file
full_data = pd.read_csv('../datasets/advertisingandsales.csv', header=None)
adexpenses = full_data[[1]]
adexpenses = np.array(adexpenses)
sales = full_data[[2]]
sales = np.array(sales)


def predict_sales(adexpenses, weight, bias):
    return weight*adexpenses + bias

def cost_function(adexpenses, sales, weight, bias):
    years = len(adexpenses)
    total_error = 0.0
    for i in range(years):
        total_error += (sales[i] - (weight*adexpenses[i] + bias))**2
    return total_error / years

def update_weights(adexpenses, sales, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    years = len(adexpenses)

    for i in range(years):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += -2 * adexpenses[i] * (sales[i] 
                            - (weight * adexpenses[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -2 * (sales[i] - (weight * adexpenses[i] + bias))

    # We subtract because the derivatives point in direction of steepest 
    # ascent
    weight -= (weight_deriv / years) * learning_rate
    bias -= (bias_deriv / years) * learning_rate

    return weight, bias

def train(adexpenses, sales, weight, bias, learning_rate, iterations):

    for i in range(iterations):
        weight,bias = update_weights(adexpenses, sales, weight, bias, 
                                     learning_rate)

        # Calculating cost 
        cost = cost_function(adexpenses, sales, weight, bias)

        # Logging the Progress
        if i % 10 == 0:
            print("iteration: " + str(i) + " weight: " + str(weight) 
            + " bias: " + str(bias) + " cost: " + str(cost))

    return weight, bias, cost

fw, fb, fc = train(adexpenses, sales, weight, bias, learning_rate, iterations)
print("Final Weight: " + str(fw) + " Final Bias: " + str(fb) 
+ " Final Cost: " + str(fc))
