import numpy as np


# sigmoid function # I thnk it's wrong, as the derivative is not like this
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))


iter=1

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
weights = 2 * np.random.random((3, 1)) - 1
weights2=weights


# forward propagation
l0 = X
l1 = nonlin(np.dot(l0, weights))

    # how much did we miss?
l1_error = y - l1


print 'Loop num: %d. The output are %0.5f, %0.5f, %0.5f, %0.5f' % (iter,l1[0], l1[1], l1[2], l1[3])

print 'Loop num: %d. The target are %0.5f, %0.5f, %0.5f, %0.5f' % (iter, y[0], y[1], y[2], y[3])

print 'Loop num: %d. The error are %0.5f, %0.5f, %0.5f, %0.5f' % (iter, l1_error[0], l1_error[1], l1_error[2], l1_error[3])

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
l1_delta = l1_error * nonlin(l1, True)

old_der=nonlin(l1, True)
der=sigmoid_deriv(l1)

print 'The old derivative are %0.5f, %0.5f, %0.5f, %0.5f' %(old_der[0],old_der[1],old_der[2],old_der[3])
print 'The new derivatives are %0.5f, %0.5f, %0.5f, %0.5f' %(der[0],der[1],der[2],der[3])

productErrorDer=l1_error*der

print 'The old error products are %0.5f, %0.5f, %0.5f, %0.5f' % (l1_delta[0], l1_delta[1], l1_delta[2], l1_delta[3])
print 'The new error products are %0.5f, %0.5f, %0.5f, %0.5f' % (productErrorDer[0], productErrorDer[1], productErrorDer[2], productErrorDer[3])


    # update weights
weights += np.dot(l0.T, l1_delta)
weights2 += np.dot(l0.T, productErrorDer)

print 'The old updated weights are %0.5f, %0.5f, %0.5f' % (weights[0], weights[1], weights[2])
print 'The new updated  weights are %0.5f, %0.5f, %0.5f' % (weights2[0], weights2[1], weights2[2])

print '####################'


