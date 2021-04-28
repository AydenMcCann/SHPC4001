import numpy as np


def XOR(x1, x2):

    def sig(x):
        return 1 / (1 + np.exp(-(np.dot(w,x)+b)))

    xvals = np.array([[x1, x2]]).T
    w = np.array([[1, 0], [1, 13]])
    b = np.array([1, 0]).T
    w1 = np.array([[1, 1]])

    b1 = np.array([-0.680])
    f = np.sum(np.outer(w1, sig(xvals)) + b1)


# rounding inputs
# comment out this code and uncomment return f to obtain original inputs
    if round(f) == 1:
        return 1
    else:
        return 0

# # uncomment to obtain original, unrounded outputs
#     return f


print("y expected: [[0]] y Network: [[{}]]".format(XOR(0, 0)))
print("y expected: [[1]] y Network: [[{}]]".format(XOR(0, 1)))
print("y expected: [[1]] y Network: [[{}]]".format(XOR(1, 0)))
print("y expected: [[0]] y Network: [[{}]]".format(XOR(1, 1)))
