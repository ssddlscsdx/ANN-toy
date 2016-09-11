
import numpy as np

# this is a new change
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

input=2


#strHello = "the length of (%s) is %d" %('Hello World',len('Hello World'))
#print strHello


for num in range(-5,5,1):
    result = sigmoid(num);
    result_de = sigmoid_deriv(num)
    print 'The simgmoid of %d is %.2f, The derivative of sigmoid of %d is %.2f' %(num,result, num,result_de)


print 'This is when all comes to an end'
#output="%s wocao is it this %s, %.5f" %("ganniniang","hahahhahah",result_de)
#print output