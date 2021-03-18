
def f(x):
    return x**2+2*x-3


def forward_d(f,x,h):
    df = (f(x+h)-f(x))/h
    return df

def backward_d(f,x,h):
    df = (f(x)-f(x-h))/h
    return df

def central_d(f,x,h):
    df = (f(x+h)-f(x-h))/(2*h)
    return df

p = 1 # the x value
h = 0.01 # step size

print("Forward method f'({}) ={}".format(p,forward_d(f,p,h)))
print("Backward method f'({}) = {}".format(p,backward_d(f,p,h)))
print("Central method f'({}) = {}".format(p, central_d(f,p,h)))
