#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 3

import math

def read_data(filename):
    import csv, numpy as np
    instances = []
    with open(filename, 'r') as csvfile:
        dataset = csv.reader(csvfile, delimiter=',')
        for row in dataset:
            instances.append(float(row[0]))
    size = len(instances)
    return instances, size

if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt

    # read the data
    x, size = read_data('poisson.csv')
    print(x)

    # initialize the lambdas with values between 0 and 50, step 0.1
    L = np.linspace(0.1, 50, 500)

    # compute the value of the density function for every lambda
    densities = []
    for lambd in L:
        sum = 0
        for xi in x:
            sum += (math.log(math.factorial(xi)) + xi * math.log(lambd) - lambd)
        densities.append(sum)


    max_density = max(densities)  # Find the maximum density
    max_lambda = L[densities.index(max_density)]  # Find the lambda value corresponding to the maximum density

    mean = np.mean(x)
    print("Mean is "+str(mean)+" and lambda for which max is attained is "+str(max_lambda))

    plt.plot(L,densities)
    # Mark the maximum in the plot
    plt.text(max_lambda+1, max_density, "Maximum")
    plt.plot(max_lambda, max_density, 'ro')
    # Add labels
    plt.xlabel("Lambda")
    plt.ylabel("Density")
    plt.show()
