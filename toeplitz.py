def forward(size):
    """
    returns a toeplitz matrix for forward differences
    """
    r = zeros(size)
    c = zeros(size)
    r[0] = -1
    r[-1] = 1
    c[1] = 1
    return toeplitz(r,c)

def backward(size):
    """
    returns a toeplitz matrix for forward differences
    """
    r = zeros(size)
    c = zeros(size)
    r[0] = 1
    r[-1] = -1
    c[1] = -1
    return toeplitz(r,c).T

def central(size):
    """
    returns a toeplitz matrix for central differences
    """
    r = zeros(size)
    c = zeros(size)
    r[1] = .5
    r[-1] = -.5
    c[1] = -.5
    c[-1] = .5
    return toeplitz(r,c).T

def example_finite_difference(self):
    x = linspace(0,10,15)
    y = cos(x) # recall, the derivative of cos(x) is sin(x)
    # we need the step h to compute f'(x) 
    # because the product gives h*f'(x)
    h = x[1]-x[2]
    # generating the matrices
    Tf = forward(15)/h 
    Tb = backward(15)/h
    Tc = central(15)/h

    pylab.subplot(211)
    # approximation and plotting
    pylab.plot(x,dot(Tf,y),'g',x,dot(Tb,y),'r',x,dot(Tc,y),'m')
    pylab.plot(x,sin(x),'b--',linewidth=3)
    pylab.axis([0,10,-1,1])

    # the same experiment with more samples (h is smaller)
    x = linspace(0,10,50)
    y = cos(x)
    h = x[1]-x[2]
    Tf = forward(50)/h
    Tb = backward(50)/h
    Tc = central(50)/h

    pylab.subplot(212)
    pylab.plot(x,dot(Tf,y),'g',x,dot(Tb,y),'r',x,dot(Tc,y),'m')
    pylab.plot(x,sin(x),'b--',linewidth=3)
    pylab.axis([0,10,-1,1])
    pylab.legend(['Forward', 'Backward', 'Central', 'True f prime'],loc=4)
    pylab.show()

