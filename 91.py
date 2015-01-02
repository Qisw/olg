"""
December 31, 2014, Hyun Chang Yi
OLG value function approximation, a Python version of Rch91d.g, Rch91v.g 
and Rch91p.g by Burkhard Heer
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar, bisect, root
from scipy.linalg import toeplitz
from numpy import linspace, mean, array, zeros, cos, dot, sin, ones, concatenate, split, vectorize
from random import random
from matplotlib import pyplot as plt
from datetime import datetime
from math import fabs, pi
import time
import pylab


class OLG:
    """
    This class is just a "struct" to hold  the collection of primitives defining
    a economy with a fixed demographic structure.
    """
    def __init__(self,beta=0.96, sigma=2.0, alpha=0.3, zeta0=0.3, zeta1=0.2, delta=0.06,
        R=20, W=40, gamma=2.0, Kmax=5, Kmin=0, Na=51, psi=0.001, phi=0.8, Nq=30,
        tol=0.005, tol10=1e-10, neg=-1e10, ncheb = 3, Kinit=0.7, Ninit=0.3):
        self.beta, self.sigma, self.alpha, self.zeta0 = beta, sigma, alpha, zeta0
        self.delta, self.R, self.W, self.T, self.gamma = delta, R, W, W+R, gamma
        self.gamma, self.Kmax, self.Kmin, self.Na, self.Nq = gamma, Kmax, Kmin, Na, Nq
        self.psi, self.phi, self.tol, self.tol10, self.zeta1 = psi, phi, tol, tol10, zeta1
        self.neg = neg              # the worst value
        self.ncheb = ncheb          # order of Chebyshev polynomial in projection method
        self.tau = zeta0/(2.0+zeta0)  # tax rate that supports replacement rate zeta0 in PAYG
        self.agrid = linspace(Kmin,Kmax,Na)
        # aggregate labor supply and capital
        self.N = Ninit
        self.K = Kinit
        # market prices and pension benefit from PAYG scheme
        self.w = self.setwage(self.K, self.N)
        self.r = self.setrate(self.K, self.N)
        self.b = self.benefit(self.N)
        # value function and its interpolation
        self.v = array([[0 for l in range(self.Na)] for y in range(self.T)], dtype=float)
        self.vtilde = [[] for y in range(self.T)]
        # policy functions used in value function method
        self.a = array([[0 for l in range(self.Na)] for y in range(self.T)], dtype=float)
        self.c = array([[0 for l in range(self.Na)] for y in range(self.T)], dtype=float)
        self.n = array([[0 for l in range(self.Na)] for y in range(self.T)], dtype=float)
        # the following paths for a, c, n and u are used in direct and value function methods
        # In direct method, those paths are directly calculated, while in the value function
        # method the paths are calculated from value and policy functions
        self.apath = array([0 for y in range(self.T)], dtype=float)
        self.cpath = array([0 for y in range(self.T)], dtype=float)
        self.npath = array([0 for y in range(self.T)], dtype=float)
        self.upath = array([0 for y in range(self.T)], dtype=float)
        self.Converged = False
        # ac, an and anode are used in projection method
        self.ac = array([[0 for l in range(self.ncheb+1)] for y in range(self.T)], dtype=float)
        self.an = array([[0 for l in range(self.ncheb+1)] for y in range(self.T)], dtype=float)
        self.anode = (cos((2*linspace(1,ncheb+1,ncheb+1)-1)/(2*(ncheb+1))*pi)+1)*(Kmax-Kmin)/2.0+Kmin


    def SetPrices(self):
        """ Update market prices, w and r, and pension benefit according to new
        aggregate capital and labor from last iteration """
        self.w = self.setwage(self.K, self.N)
        self.r = self.setrate(self.K, self.N)
        self.b = self.benefit(self.N)


    def IterateValues(self):
        """ Given K and N, r, w and b are calculated
        Then, given r, w and b, all generations' value and decision functions
        are calculated ***BACKWARD*** """
        # y = -1 : the oldest generation
        for l in range(self.Na):            
            self.c[-1][l] = self.agrid[l]*(1+self.r) + self.b
            self.v[-1][l] = self.util(self.c[-1][l],0)
        self.vtilde[-1] = interp1d(self.agrid,self.v[-1], kind='cubic')
        # y = -2, -3,..., -60
        for y in range(-2,-(self.T+1),-1):
            m0 = 0
            for i in range(self.Na):    # l = 0, 1, ..., 50
                # Find a bracket within which optimal a' lies
                m = max(0, m0)  # Rch91v.g uses m = max(0, m0-1)
                m0, a, b, c = self.GetBracket(y, i, m, self.agrid)
                # Find optimal a' using Golden Section Search
                if a == b:
                    self.a[y][i] = 0
                elif b == c:
                    self.a[y][i] = self.agrid[-1]
                else:
                    def objfn(a1): # Define objective function for optimal a'
                        return -self.value(y, self.agrid[i], a1)
                    result = minimize_scalar(objfn, bracket=(a,b,c), method='Golden')
                    #‘Brent’,‘Bounded’,‘Golden’
                    self.a[y][i] = result.x
                # Computing consumption and labor
                if y >= -self.R:
                    self.c[y][i] = (1+self.r)*self.agrid[i] + self.b - self.a[y][i]
                    self.n[y][i] = 0
                else:
                    self.c[y][i], self.n[y][i] = self.solve(self.agrid[i], self.a[y][i])
                self.v[y][i] = self.util(self.c[y][i],self.n[y][i]) \
                                + self.beta*self.vtilde[y+1](self.a[y][i])
            self.vtilde[y] = interp1d(self.agrid, self.v[y], kind='cubic')


    def GetBracket(self, y, l, m, agrid):
        """
        Find a bracket (a,b,c) such that policy function for next period asset level, 
        a[x;asset[l],y] lies in the interval (a,b)
        """
        a, b, c = agrid[0], agrid[0]-agrid[1], agrid[0]-agrid[2]
        m0 = m
        v0 = self.neg
        while (a > b) or (b > c):
            v1 = self.value(y, agrid[l], agrid[m])
            if v1 > v0:
                if m == 0:
                    a, b = agrid[m], agrid[m]
                else:
                    b, a = agrid[m], agrid[m-1]
                v0, m0 = v1, m
            else:
                c = agrid[m]
            if m == self.Na - 1:
                a, b, c = agrid[m-1], agrid[m], agrid[m]
            m = m + 1
        return m0, a, b, c


    def CalculatePaths(self):
        """ Compute the aggregate capital stock and employment, K and N **FORWARD**
        """
        self.apath = array([0 for y in range(self.T)], dtype=float)
        self.cpath = array([0 for y in range(self.T)], dtype=float)
        self.npath = array([0 for y in range(self.T)], dtype=float)
        # generate each generation's asset, consumption and labor supply forward
        for y in range(self.T-1):    # y = 0, 1,..., 58
            self.apath[y+1] = min(5,max(0,interp1d(self.agrid,self.a[y],kind='cubic')(self.apath[y])))
            if y >= self.W:
                self.cpath[y] = (1+self.r)*self.apath[y] + self.b - self.apath[y+1]
                self.npath[y] = 0
            else:
                self.cpath[y], self.npath[y] = self.solve(self.apath[y], self.apath[y+1])
            self.upath[y] = self.util(self.cpath[y], self.npath[y])
        # the oldest generation's consumption and labor supply
        self.cpath[self.T-1] = (1+self.r)*self.apath[self.T-1]+self.b
        self.npath[self.T-1] = 0
        self.upath[self.T-1] = self.util(self.cpath[self.T-1], self.npath[self.T-1])


    def Aggregate(self):
        # Aggregate all generations' capital and labor supply
        Knew = mean(self.apath)
        Nnew = mean(self.npath)*self.W/(self.T*1.0)
        print 'At r:',"%2.1f" %(self.r*100),'At w:',"%2.1f" %(self.w),\
                    'Ks:',"%1.3f" %(Knew),'and Ls:',"%1.3f" %(Nnew), \
                    'and Ks/Ys',"%1.3f" %(Knew/(Knew**(self.alpha)*Nnew**(1-self.alpha)))
        self.Converged = (fabs(Knew-self.K) < self.tol*self.K)
        self.K = self.phi*self.K + (1-self.phi)*Knew
        self.N = self.phi*self.N + (1-self.phi)*Nnew
        # whether aggregate asset has converged, i.e., no change from the last iteration


    def util(self, c, n):
        # calculate utility value with given consumption and labor
        return (((c+self.psi)*(1-n)**self.gamma)**(1-self.sigma)-1)/(1-self.sigma*1.0)


    def setrate(self, K, N):
        # interest rate is at least 0.
        return max(self.alpha*K**(self.alpha-1)*N**(1-self.alpha) - self.delta, 0)


    def setwage(self, K, N):
        return (1-self.alpha)*K**self.alpha*N**(-self.alpha)


    def benefit(self, N):
        return self.zeta0*(1-self.tau)*self.w*N*self.T/(self.W*1.0)


    def uc(self, c, n):
        # marginal utility w.r.t. consumption
        return (c+self.psi)**(-self.sigma)*(1-n)**(self.gamma*(1-self.sigma))


    def un(self, c, n):
        # marginal utility w.r.t. labor
        return -self.gamma*(c+self.psi)**(1-self.sigma)*(1-n)**(self.gamma*(1-self.sigma)-1)


    def IteratePaths(self):
        """
        Directly solve each generation's optimal a', c and n
        In this case, apath, cpath and npath store agents' policy functions
        In the other case of value interation, the above three paths are calculated from
        each generation's value functions.
        """
        a1, aT = [-1,], []
        for q in range(self.Nq):
            if q == 0:
                self.apath[-1] = 0.2
            elif q == 1:
                self.apath[-1] = 0.3
            else:
                self.apath[-1] = max(0,aT[-1]-(aT[-1]-aT[-2])*a1[-1]/(a1[-1]-a1[-2]))
                
            self.npath[-1] = 0
            self.cpath[-1] = self.apath[-1]*(1+self.r) + self.b

            for y in range(-2,-(self.T+1),-1):     # y = -2, -3,..., -60
                self.apath[y], self.npath[y], self.cpath[y] = self.DirectSolve(y)

            aT.append(self.apath[-1])
            a1.append(self.apath[-self.T])
            if (fabs(self.apath[-self.T])<self.tol):
                break
        for y in range(-1,-(self.T+1),-1):
            self.upath[y] = self.util(self.cpath[y],self.npath[y])


    def DirectSolve(self, y):
        """
        Directly solve capital and labor supply given next two periods capitals
        y is given as -2, -3, ..., -60, i.e., through the next-to-last to the first
        """
        if y >= -self.R:
            a1 = self.apath[y+1]
            if y == -2:
                a2 = 0
            else:
                a2 = self.apath[y+2]                
            def constraints(a):
                c0 = (1+self.r)*a + self.b - a1
                c1 = (1+self.r)*a1 + self.b - a2
                return self.uc(c0,0)-self.beta*self.uc(c1,0)*(1+self.r)
            a, n = fsolve(constraints, a1), 0
            c = (1+self.r)*a + self.b - a1
        else:
            a1 = self.apath[y+1]
            a2 = self.apath[y+2]
            if y == -(self.R+1):
                n1 = 0
                c1 = (1+self.r)*a1 + self.b - a2
            else:
                n1 = self.npath[y+1]
                c1 = (1+self.r)*a1 + (1-self.tau)*self.w*n1 - a2
            def constraints((a0,n0)):
                c0 = (1+self.r)*a0 + (1-self.tau)*self.w*n0 - a1
                return self.uc(c0,n0) - self.beta*self.uc(c1,n1)*(1+self.r),\
                    (1-self.tau)*self.w*self.uc(c0,n0) + self.un(c0,n0)
            a, n = fsolve(constraints,(a1,n1))
            c = (1+self.r)*a + (1-self.tau)*self.w*n - a1
        return a, n, c


    def value(self, y, a0, a1):
        """
        Return the value at the given generation and asset a0 and corresponding 
        consumption and labor supply when the agent chooses his 
        next period asset a1, current period consumption c and labor n
        a1 is always within Kmin and Kmax
        """
        if y >= -self.R:    # y = -2, -3, ..., -60
            c, n = (1+self.r)*a0 + self.b - a1, 0
        else:
            c, n = self.solve(a0,a1)

        v = self.util(c,n) + self.beta*self.vtilde[y+1](a1)
        return v if c >= 0 else self.neg


    def solve(self, a0, a1):
        # FOC for workers to optimize on consumption and labor supply
        def constraints((c,n)):
            return (1+self.r)*a0+(1-self.tau)*self.w*n - a1 - c,\
                (1-self.tau)*self.w*self.uc(c,n) + self.un(c,n)
                # self.gamma*(c+self.psi)-(1-n)*self.w*(1-self.tau)
        c, n = fsolve(constraints,(0.3,0.3))
        """ or, we can analytically solve for c and n
        c = ((1+self.r)*a0-a1+(1-self.tau)*self.w-self.gamma*self.psi)/(1+self.gamma)
        n = (a1-(1+self.r)*a0+c)/(self.w*(1-self.tau))
        """
        return c, n


    def IterateCheby(self):
        """ Update Chebyshev polynomial
        """
        Kmax, Kmin = self.Kmax, self.Kmin
        # numpy.vectorize creates an elementwise function like "ufunc" in Numpy
        # vuc = vectorize(self.uc)
        # vun = vectorize(self.un)
        # policy function in period T and W
        self.ac = array([[0 for l in range(self.ncheb+1)] for y in range(self.T)], dtype=float)
        self.an = array([[0 for l in range(self.ncheb+1)] for y in range(self.T)], dtype=float)        
        
        def cT(x):
            return (1+self.r)*x+self.b
        def nW(x):
            return 0.3*(1+x/(self.Kmax*1.0))
        def uT(x):
            return self.util(c0(x),0)
        self.ac[-1] = self.chebcoef(cT,self.ncheb,2*self.ncheb,Kmin,Kmax)
        self.an[-self.R-1] = self.chebcoef(nW,self.ncheb,2*self.ncheb,Kmin,Kmax)

        for y in range(-2,-(self.T+1),-1):     # y = -2, -3,..., -60            
            if y >= -self.R:
                def foc(ac0):
                    c = self.chebeval(self.anode,ac0,Kmin,Kmax)
                    a1 = (1+self.r)*self.anode + self.b - c
                    c1 = self.chebeval(a1,self.ac[y+1],Kmin,Kmax)
                    return self.uc(c,0) - self.uc(c1,0)*self.beta*(1+self.r)
                self.ac[y] = fsolve(foc,self.ac[y])
            else:
                acn = concatenate((self.ac[y],self.an[y]))
                def foc(acn):
                    ac0, an0 = split(acn,2)
                    c = self.chebeval(self.anode,ac0,Kmin,Kmax)
                    n = self.chebeval(self.anode,an0,Kmin,Kmax)
                    a1 = (1+self.r)*self.anode + (1-self.tau)*self.w*n - c
                    c1 = self.chebeval(a1,self.ac[y+1],Kmin,Kmax)
                    n1 = self.chebeval(a1,self.an[y+1],Kmin,Kmax)
                    v1 = self.uc(c,n) - self.uc(c1,n1)*self.beta*(1+self.r)
                    v2 = self.un(c,n) + self.uc(c,n)*self.w*(1-self.tau)
                    # v1 = vuc(c,n) - vuc(c1,n1)*self.beta*(1+self.r)
                    # v2 = vun(c,n) - vuc(c,n)*self.w*(1-self.tau)
                    return concatenate((v1, v2))
                # sol = root(foc, acn, method='broyden1')
                # print sol
                """
                for root function, the following methods are available: hybr, lm, broyden1, 
                broyden2, anderson, linearmixing, diagbroyden, excitingmixing, krylov
                """
                acn1 = fsolve(foc,acn)
                self.ac[y], self.an[y] = split(acn1,2)
                # self.ac[y], self.an[y] = split(fsolve(foc,acn),2)
            print self.ac[y], self.an[y]


    def CalculateChebyPaths(self):
        """
        Compute the aggregate capital stock and employment, K and N **FORWARD**
        from Chebyshev polynomials
        """
        Kmin, Kmax = self.Kmin, self.Kmax
        self.apath = array([0 for y in range(self.T)], dtype=float)
        self.cpath = array([0 for y in range(self.T)], dtype=float)
        self.npath = array([0 for y in range(self.T)], dtype=float)
        # generate each generation's asset, consumption and labor supply forward
        for y in range(self.T-1):    # y = 0, 1,..., 58
            self.cpath[y] = self.chebeval(array([self.apath[y]]),self.ac[y],Kmin,Kmax)
            # if self.cpath[y] < 0:
            #     self.cpath[y] = 0
            if y >= self.W:
                income = self.b
            else:
                self.npath[y] = self.chebeval(array([self.apath[y]]),self.an[y],Kmin,Kmax)
                income = (1-self.tau)*self.w*self.npath[y]
            self.apath[y+1] = (1+self.r)*self.apath[y] + income - self.cpath[y]
            self.upath[y] = self.util(self.cpath[y], self.npath[y])
        # the oldest generation's consumption and labor supply
        self.cpath[self.T-1] = (1+self.r)*self.apath[self.T-1] + self.b
        # self.cpath[self.T-1] = self.chebeval(array([self.apath[self.T-1]]),self.ac[self.T-1],Kmin,Kmax)
        self.upath[self.T-1] = self.util(self.cpath[self.T-1], self.npath[self.T-1])
        # print self.cpath, self.apath, self.npath


    def chebcoef(self,f,n,m,Kmin,Kmax):
        """ 
        returns coefficients of Chebyshev Regression of f with m sample points 
        on the interval [min,max], and n is the order of chebyshev polynomials
        """
        z = -cos((linspace(1,m,m)*2-1)*pi/(2*m*1.0))
        x = (z+1)*(Kmax-Kmin)/2.0 + Kmin
        y = f(x)
        # print 'x, f(x):', x, y
        T0 = ones(m)
        T1 = z
        a = zeros(n+1)
        a[0] = sum(y)/(m*1.0)
        a[1] = dot(y,T1)/dot(T1,T1)
        for i in range(2,n+1):
            T = 2*z*T1 - T0
            a[i] = dot(y,T)/(dot(T,T)*1.0)
            T0 = T1
            T1 = T
        return a


    def chebeval(self,x,a,Kmin,Kmax):
        # print x, x.shape
        p = x.shape[0]
        z = 2*(x-Kmin)/(Kmax-Kmin) - 1
        T0 = ones(p)
        T1 = z
        y = a[0] + a[1]*z
        # print 'x:',x,'\n a:',a,    '\n p:',p,'\n z:',z,'\n T0:',T0,'\n y:',y
        for i in range(2,self.ncheb + 1):
            T = 2*z*T1 - T0
            y = y + a[i]*T
            T0 = T1
            T1 = T
        return y


"""
The following are procedures to get steady state of the economy using three different
methods, value function iteration, direct age-profile iteration and projection method
"""

def value(e,N=20):
    start_time = datetime.now() # records the starting time
    for i in range(N):      # N is the maximum number of iteration
        e.SetPrices()       # From last iteration's K and N, update market prices
        e.IterateValues()   # Given prices, update value function
        e.CalculatePaths()  # Given value function, find life-cycle profiles
        e.Aggregate()       # Given profiles, calculate aggregate K and N
        if e.Converged:
            print 'Converged! in',i+1,'iterations with tolerance level',e.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    # return e


def direct(e,N=30):
    start_time = datetime.now() # records the starting time
    for i in range(N):      # N is the maximum number of iteration
        e.SetPrices()       # From last iteration's K and N, update market prices
        e.IteratePaths()    # Given prices, find life-cycle profiles s.t. a0=0
        e.Aggregate()       # Given profiles, calculate aggregate K and N
        if e.Converged:
            print 'Converged! in',i+1,'iterations with tolerance level',e.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    # return e


def projection(e,N=30):
    """
    This method does NOT work!! except for some parameter values like
    eco = projectionmethod(N=2,tol=0.01,R=5,W=10,beta=0.98,delta=0.05,ncheb=2)
    """
    start_time = datetime.now() # records the starting time
    for i in range(N):
        e.SetPrices()
        e.IterateCheby()
        e.CalculateChebyPaths()
        e.Aggregate()
        if e.Converged:
            print 'Converged! in',i+1,'iterations with tolerance level',e.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    # return e


plt.close("all")

def showpath(e):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fig.subplots_adjust(hspace=.5)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.plot(e.apath)
    ax2.plot(e.npath)
    ax3.plot(e.cpath)
    ax4.plot(e.upath)
    ax.set_xlabel('generation')
    ax1.set_title('Asset')
    ax2.set_title('Labor')
    ax3.set_title('Consumption')
    ax4.set_title('Utility')
    plt.show()
    time.sleep(1)
    plt.close() # plt.close("all")


"""
eco = valuemethod(N=15,R=20,W=40,Na=51) works well.
eco = directmethod(N=50,R=20,W=40,tol=0.001,delta=0.05,alpha=0.3,beta=0.965,phi=0.9)
eco = directmethod(N=50,R=20,W=40,tol=0.001,delta=0.05,alpha=0.3,beta=0.96,phi=0.9)

"""
