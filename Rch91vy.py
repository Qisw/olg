"""
December 19, 2014, Hyun Chang Yi
OLG value function approximation, a Python version of Rch91v.g by Burkhard Heer
Cubic Spline and Golden Section Search
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar, bisect
from numpy import linspace, mean, array
from random import random
from matplotlib import pyplot as plt
from datetime import datetime
from math import fabs


class SteadyPopulation:
    """
    This class is just a "struct" to hold  the collection of primitives defining
    a economy with a fixed demographic structure.
    """
    def __init__(self,beta=0.99, sigma=2.0, alpha=0.3, zeta=0.3, delta=0.1, R=20,
                    W=40, gamma=2.0, Kmax=5, Kmin=0, Na=51, psi=0.001, phi=0.8,
                    tol=0.005,tol10=1e-10, neg=-1e10):
        self.beta, self.sigma, self.alpha, self.zeta = beta, sigma, alpha, zeta
        self.delta, self.R, self.W, self.T, self.gamma = delta, R, W, W+R, gamma
        self.gamma, self.Kmax, self.Kmin, self.Na = gamma, Kmax, Kmin, Na
        self.psi, self.phi, self.tol, self.tol10 = psi, phi, tol, tol10
        self.neg = neg
        self.tau = zeta/(2.0+zeta)
        self.agrid = linspace(Kmin,Kmax,Na)
        self.N = 0.2
        self.K = 0.7
        self.w = 0
        self.r = 0
        self.b = 0
        self.v = array([[0 for l in range(self.Na)] for y in range(self.T)], dtype=float)
        self.vtilde = [[] for y in range(self.T)]
        self.a = array([[0 for l in range(self.Na)] for y in range(self.T)], dtype=float)
        self.c = array([[0 for l in range(self.Na)] for y in range(self.T)], dtype=float)
        self.n = array([[0 for l in range(self.Na)] for y in range(self.T)], dtype=float)
        self.apath = array([0 for y in range(self.T)], dtype=float)
        self.cpath = array([0 for y in range(self.T)], dtype=float)
        self.npath = array([0 for y in range(self.T)], dtype=float)
        self.upath = array([0 for y in range(self.T)], dtype=float)
        self.Converged = False
        self.a0Converged = False


    def IterateValues(self):
        """
        Given K and N, r, w and b are calculated
        Then, given r, w and b, all generations' value and decision functions
        are calculated ***BACKWARD***
        """
        agrid = self.agrid
        self.w = self.mpl(self.K, self.N)
        self.r = self.mpk(self.K, self.N) - self.delta
        self.b = self.benefit(self.N)

        for l in range(self.Na):            
            self.c[-1][l] = agrid[l]*(1+self.r) + self.b
            self.v[-1][l] = self.util(self.c[-1][l],0)
        self.vtilde[-1] = interp1d(agrid,self.v[-1], kind='cubic')

        for y in range(-2,-(self.T+1),-1):     # y = -2, -3,..., -60
            m0 = 0 
            for l in range(self.Na):
                # Find a bracket within which optimal a' lies
                m = max(0, m0-1)
                m0, a, b, c = self.GetBracket(y, l, m, agrid)
                # Define objective function for optimal a'
                def objfn(a1):
                    v = self.value(y, agrid[l], a1)
                    return -v
                # Find optimal a' using Golden Section Search
                if a == b:
                    self.a[y][l] = 0
                elif b == c:
                    self.a[y][l] = agrid[-1]
                else:
                    result = minimize_scalar(objfn, bracket=(a,b,c), method='Golden')
                    #‘Brent’,‘Bounded’,‘Golden’
                    self.a[y][l] = result.x
                # Computing consumption and labor
                if y >= -self.R:
                    self.c[y][l], self.n[y][l] = (1+self.r)*agrid[l] + self.b - self.a[y][l], 0
                else:
                    self.c[y][l], self.n[y][l] = self.solve(agrid[l], self.a[y][l])
                self.v[y][l] = self.util(self.c[y][l],self.n[y][l]) + self.beta*self.vtilde[y+1](self.a[y][l])
            self.vtilde[y] = interp1d(agrid, self.v[y], kind='cubic')


    def GetBracket(self, y, l, m, agrid):
        a, b, c = agrid[0], agrid[0]-agrid[1], agrid[0]-agrid[2]
        m0 = m
        v0 = self.neg
        while a > b or b > c:
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
        """
        Compute the aggregate capital stock and employment, K and N **FORWARD**
        and return whether (abs(K - Kp)/K < tol) holds
        """
        agrid = self.agrid        
        self.apath = array([0 for y in range(self.T)], dtype=float)
        self.cpath = array([0 for y in range(self.T)], dtype=float)
        self.npath = array([0 for y in range(self.T)], dtype=float)
        # generate each generation's asset, consumption and labor supply forward
        for y in range(self.T-1):    # y = 0, 1,..., 58
            self.apath[y+1] = max(0,interp1d(agrid, self.a[y], kind='cubic')(self.apath[y]))
            if y >= self.W:
                self.cpath[y], self.npath[y] = (1+self.r)*self.apath[y] + self.b - self.apath[y+1], 0
            else:
                self.cpath[y], self.npath[y] = self.solve(self.apath[y], self.apath[y+1])
            self.upath[y] = self.util(self.cpath[y], self.npath[y])
        # the oldest generation's consumption and labor supply
        self.cpath[self.T-1], self.npath[self.T-1] = (1+self.r)*self.apath[self.T-1]+self.b, 0
        self.upath[self.T-1] = self.util(self.cpath[self.T-1], self.npath[self.T-1])


    def Aggregate(self):
        # Aggregate all generations' capital and labor supply
        Knew = mean(self.apath)
        Nnew = mean(self.npath)*self.W/(self.T*1.0)
        print 'At r=',"%1.4f" %(self.r),'Ks:',"%1.3f" %(Knew),'and Ls:',"%1.3f" %(Nnew), \
                    'and Ks/Ys',"%1.3f" %(Knew/(Knew**(self.alpha)*Nnew**(1-self.alpha)))
        self.Converged = (fabs(Knew-self.K)/(self.K*1.0) < self.tol)
        self.K = self.phi*self.K + (1-self.phi)*Knew
        self.N = self.phi*self.N + (1-self.phi)*Nnew
        # whether aggregate asset has converged, i.e., no change from the last iteration


    def util(self, c, n):
        # calculate utility value with given consumption and labor
        return (((c+self.psi)*(1-n)**self.gamma)**(1-self.sigma)-1)/(1-self.sigma*1.0)


    def mpk(self, K, N):
        return self.alpha*K**(self.alpha-1)*N**(1-self.alpha)


    def mpl(self, K, N):
        return (1-self.alpha)*K**self.alpha*N**(-self.alpha)


    def benefit(self, N):
        return self.zeta*(1-self.tau)*self.w*N*self.T/(self.W*1.0)


    def IteratePaths(self):
        """
        Directly solve each generation's optimal a', c and n
        In this case, apath, cpath and npath store agents' policy functions
        In the other case of value interation, the above three paths are calculated from
        each generation's value functions.
        """
        self.w = self.mpl(self.K, self.N)
        self.r = self.mpk(self.K, self.N)
        self.b = self.benefit(self.N)

        a1, aT = [-1,], []

        for q in range(self.Na):
            if q == 0:
                self.apath[-1] = 0.2# + random()*0.01
            elif q == 1:
                self.apath[-1] = 0.3# + random()*0.01
            else:
                # print 'a1-1-a1-2', (aT[-1]-aT[-2])*a1[-1]/(a1[-1]-a1[-2])
                self.apath[-1] = max(0,aT[-1]-(aT[-1]-aT[-2])/(a1[-1]-a1[-2])*a1[-1])
                
            self.npath[-1] = 0
            self.cpath[-1] = self.apath[-1]*(1+self.r) + self.b

            # print 'aT',self.apath[-1]
            for y in range(-2,-(self.T+1),-1):     # y = -2, -3,..., -60
                self.apath[y], self.npath[y], self.cpath[y] = self.DirectSolve(y)
            # print 'a0',self.apath[0]

            aT.append(self.apath[-1])
            a1.append(self.apath[-self.T])
            # if fabs(a1[-1]-a1[-2])<a1[-1]*self.tol:
            #     a1[-1] = a1[-1]+a1[-1]*self.tol
            if (fabs(self.apath[-self.T])<self.tol):
                break
        # print a1[-1], a1[-2], a1[-3]
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
                return self.uc(c0,0)/self.beta-self.uc(c1,0)*(1+self.r)
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
                return self.uc(c0,n0)/self.beta - self.uc(c1,n1)*(1+self.r),\
                    (1-self.tau)*self.w*self.uc(c0,n0) + self.un(c0,n0)
            a, n = fsolve(constraints,(a1,n1))
            c = (1+self.r)*a + (1-self.tau)*self.w*n - a1
        return a, n, c


    def uc(self, c, n):
        # marginal utility w.r.t. consumption
        return (c+self.psi)**(-self.sigma)*(1-n)**(self.gamma*(1-self.sigma))


    def un(self, c, n):
        # marginal utility w.r.t. labor
        return -self.gamma*(c+self.psi)**(1-self.sigma)*(1-n)**(self.gamma*(1-self.sigma)-1)


    def value(self, y, a0, a1):
        """
        Return the value at the given generation and asset a0 and corresponding 
        consumption and labor supply when the agent chooses his 
        next period asset a1, current period consumption c and labor n
        a1 is always within Kmin and Kmax
        """
        if y >= -self.R:    # y = -2, -3, ..., -60
            c, n = (1+self.r)*a0 + self.b - a1, 0
            #print self.r, a0, self.b, a1, c, n
        else:
            c, n = self.solve(a0,a1)

        v = self.util(c,n) + self.beta*self.vtilde[y+1](a1)
        #print 'v,util,beta*v',v,self.util(c,n),self.beta*self.vtilde[y+1](a1), self.beta, y+1
        #print self.util(c,n), self.beta*self.vtilde[y+1](a1)
        return v if c > 0 else self.neg


    def solve(self, a0, a1):
        # FOC for workers to optimize on consumption and labor supply
        def constraints((c,n)):
            return (1+self.r)*a0+(1-self.tau)*self.w*n-a1-c,\
                (1-self.tau)*self.w*self.uc(c,n) + self.un(c,n)
                # self.gamma*(c+self.psi)-(1-n)*self.w*(1-self.tau)
        c, n = fsolve(constraints,(0.3,0.3))
        # c = ((1+self.r)*a0-a1+(1-self.tau)*self.w-self.gamma*self.psi)/(1+self.gamma)
        # n = (a1-(1+self.r)*a0+c)/(self.w*(1-self.tau))
        return c, n


def valuemethod(N=10,R=10,W=20,Na=51,tol=0.005,Kmin=0,beta=0.99,
                    sigma=2.0,alpha=0.3,zeta=0.3,delta=0.1):
    start_time = datetime.now() # records the starting time
    eco = SteadyPopulation(R=R,W=W,Na=Na,tol=tol,Kmin=Kmin,beta=beta,
                            sigma=sigma,alpha=alpha,zeta=zeta,delta=delta)
    for i in range(N):
        eco.IterateValues()
        eco.CalculatePaths()
        eco.Aggregate()
        if eco.Converged:
            print 'Converged! in',i+1,'iterations with tolerance level',eco.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return eco


def directmethod(N=30,R=10,W=20,tol=0.005,beta=0.99,
                    sigma=2.0,alpha=0.3,zeta=0.3,delta=0.1,gamma=2):
    start_time = datetime.now() # records the starting time
    eco = SteadyPopulation(R=R,W=W,tol=tol,beta=beta,sigma=sigma,
                            alpha=alpha,zeta=zeta,delta=delta,gamma=gamma)
    for i in range(N):
        eco.IteratePaths()
        eco.Aggregate()
        if eco.Converged:
            print 'Converged! in',i+1,'iterations with tolerance level',eco.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return eco


def plotpath(ss):
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
    ax1.plot(ss.apath)
    ax2.plot(ss.npath)
    ax3.plot(ss.cpath)
    ax4.plot(ss.upath)
    ax.set_xlabel('generation')
    ax1.set_title('Asset')
    ax2.set_title('Labor')
    ax3.set_title('Consumption')
    ax4.set_title('Utility')
    plt.show()

"""
eco = valuemethod(N=15,R=20,W=40,Na=51) works well.
"""
