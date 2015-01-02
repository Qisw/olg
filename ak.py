"""
Jan. 2, 2015, Hyun Chang Yi
OLG value function approximation, a Python version of Rch92AK6.g by Burkhard Heer
This algorithm solves the Auerbach-Kotlikoff 6-period model, transition dynamics
unexpected change in the replacement ratio of pensions from 0.3 to 0.2
direct computation of the OLG model in section 9.1
This code separates 'generation' class from an economy that functions 
as firm, government and market.
A representative generation lives for T years in the economy.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar, bisect, root
from scipy.linalg import toeplitz
from numpy import linspace, mean, array, zeros, cos, dot, sin, ones, concatenate, split, vectorize, absolute
from random import random
from matplotlib import pyplot as plt
from datetime import datetime
from math import fabs, pi
import time
import pylab


class economy:
    """ This class is just a "struct" to hold  the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, alpha=0.3, zeta=0.3, delta=0.06, phi=0.8,
        tol=0.005, tol10=1e-10, Kinit=0.7, Ninit=0.3, TG=4, W=40, R=20):
        self.alpha, self.zeta, self.delta = alpha, zeta, delta
        self.phi, self.tol, self.tol10 = phi, tol, tol10
        self.TS, self.T, self.W, self.R = (W+R)*TG, (W+R), W, R
        # aggregate labor supply and capital
        self.N = array([0 for y in range(self.TS)], dtype=float)
        self.K = array([0 for y in range(self.TS)], dtype=float)
        # tax rate that supports replacement rate zeta in PAYG
        self.zeta = array([zeta for y in range(self.TS)], dtype=float)
        self.tau = array([zeta/(2.0+zeta) for y in range(self.TS)], dtype=float)
        # market prices and pension benefit from PAYG scheme for T years
        self.w = array([0 for y in range(self.TS)], dtype=float)
        self.r = array([0 for y in range(self.TS)], dtype=float)
        self.b = array([0 for y in range(self.TS)], dtype=float)
        # whether the capital stock has converged
        self.Converged = False


    def Aggregate(self, gs):
        # Aggregate all generations' capital and labor supply at each year
        K1, N1 = array([[0 for y in range(self.TS)] for i in range(2)], dtype=float)
        for y in range(self.TS):
            K1[y] = mean([gs[y+i].apath[self.T-1-i] for i in range(self.T)])
            N1[y] = mean([gs[y+i].npath[self.T-1-i]*self.W/(self.T*1.0) for i in range(self.W)])
        for y in [0, self.T, 2*self.T, 3*self.T]:
            print '\n At r:',self.r[y],'and w:',self.w[y],'Ks:', K1[y],'Ls:', N1[y], \
                        'and Ks/Ys:',K1[y]/(K1[y]**(self.alpha)*N1[y]**(1-self.alpha))
        # whether aggregate asset has converged, i.e., no change from the last iteration        
        self.Converged = (sum(absolute(K1-self.K)) < self.tol)
        self.K = self.phi*self.K + (1-self.phi)*K1
        self.N = self.phi*self.N + (1-self.phi)*N1
        self.Converged = False


    def UpdatePrices(self):
        """ Update market prices, w and r, and pension benefit according to new
        aggregate capital and labor paths for years 0,...,TS from last iteration """
        def CapitalReturn(K, N):    # interest rate is at least 0.
            return max(self.alpha*K**(self.alpha-1)*N**(1-self.alpha)-self.delta, 0)
        def Wage(K, N):
            return (1 - self.alpha)*K**self.alpha*N**(-self.alpha)
        def Benefit(zeta, tau, w, N):
            return zeta*(1 - tau)*w*N*self.T/(self.W*1.0)
        self.w = array([Wage(self.K[y], self.N[y]) 
                                        for y in range(self.TS)], dtype=float)
        self.r = array([CapitalReturn(self.K[y], self.N[y]) 
                                        for y in range(self.TS)], dtype=float)
        self.b = array([Benefit(self.zeta[y], self.tau[y], self.w[y], self.N[y])
                                        for y in range(self.TS)], dtype=float)


class generation:
    """ This class is just a "struct" to hold  the collection of primitives defining
    a generation """
    def __init__(self, e, beta=0.96, sigma=2.0, gamma=2.0, aH=5, aL=0,
        aN=51, Nq=50, psi=0.001, tol=0.005, tol10=1e-10, neg=-1e10):
        self.beta, self.sigma, self.gamma, self.psi = beta, sigma, gamma, psi
        self.R, self.W, self.T = e.R, e.W, e.W + e.R
        self.aH, self.aL, self.aN, self.aa = aH, aL, aN, linspace(aL,aH,aN)
        self.tol, self.tol10, self.Nq, self.neg = tol, tol10, Nq, neg
        """ value function and its interpolation """
        self.v = array([[0 for l in range(self.aN)] for y in range(self.T)], dtype=float)
        self.vtilde = [[] for y in range(self.T)]
        """ policy functions used in value function method """
        self.a = array([[0 for l in range(self.aN)] for y in range(self.T)], dtype=float)
        self.c = array([[0 for l in range(self.aN)] for y in range(self.T)], dtype=float)
        self.n = array([[0 for l in range(self.aN)] for y in range(self.T)], dtype=float)
        """ the following paths for a, c, n and u are used in direct and value function methods
        In direct method, those paths are directly calculated, while in the value function
        method the paths are calculated from value and policy functions """
        self.apath = array([0 for y in range(self.T)], dtype=float)
        self.cpath = array([0 for y in range(self.T)], dtype=float)
        self.npath = array([0 for y in range(self.T)], dtype=float)
        self.upath = array([0 for y in range(self.T)], dtype=float)


    def IterateValues(self, r, w, tau, b):
        """ Given prices r, w, tau and b over one's life-cycle, value and decision 
        functions are calculated ***BACKWARD*** """
        # y = -1 : the oldest generation
        for l in range(self.aN):
            self.c[-1][l] = self.aa[l]*(1 + r[-1]) + b[-1]
            self.v[-1][l] = self.util(self.c[-1][l], 0)
        self.vtilde[-1] = interp1d(self.aa, self.v[-1], kind='cubic')
        # y = -2, -3,..., -60
        for y in range(-2, -(self.T+1), -1):
            m0 = 0
            for i in range(self.aN):    # l = 0, 1, ..., 50
                # Find a bracket within which optimal a' lies
                m = max(0, m0)  # Rch91v.g uses m = max(0, m0-1)
                m0, a, b, c = self.GetBracket(y, i, m, r, w, tau, b)
                # Find optimal a' using Golden Section Search
                if a == b:
                    self.a[y][i] = 0
                elif b == c:
                    self.a[y][i] = self.aa[-1]
                else:
                    def objfn(a1): # Define objective function for optimal a'
                        return -self.OptimalValue(y, self.aa[i], a1, r, w, tau, b)
                    result = minimize_scalar(objfn, bracket=(a,b,c), method='Golden')
                    #‘Brent’,‘Bounded’,‘Golden’
                    self.a[y][i] = result.x
                # Computing consumption and labor
                if y >= -self.R:
                    self.c[y][i] = (1 + r[y])*self.aa[i] + b[y] - self.a[y][i]
                    self.n[y][i] = 0
                else:
                    self.c[y][i], self.n[y][i] = self.SolveForCN(y, self.aa[i],
                                                        self.a[y][i], r, w, tau, b)
                self.v[y][i] = self.util(self.c[y][i], self.n[y][i]) \
                                + self.beta*self.vtilde[y+1](self.a[y][i])
            self.vtilde[y] = interp1d(self.aa, self.v[y], kind='cubic')


    def GetBracket(self, y, l, m, r, w, tau, b):
        """ Find a bracket (a,b,c) such that policy function for next period asset level, 
        a[x;asset[l],y] lies in the interval (a,b) """
        aa = self.aa
        a, b, c = aa[0], aa[0]-aa[1], aa[0]-aa[2]
        m0 = m
        v0 = self.neg
        while (a > b) or (b > c):
            v1 = self.OptimalValue(y, aa[l], aa[m], r, w, tau, b)
            if v1 > v0:
                if m == 0:
                    a, b = aa[m], aa[m]
                else:
                    b, a = aa[m], aa[m-1]
                v0, m0 = v1, m
            else:
                c = aa[m]
            if m == self.aN - 1:
                a, b, c = aa[m-1], aa[m], aa[m]
            m = m + 1
        return m0, a, b, c


    def PathsFromValues(self, r, w, tau, b):
        """ Compute the aggregate capital stock and employment, K and N **FORWARD**
        """
        self.apath[0] = 0
        # generate each generation's asset, consumption and labor supply forward
        for y in range(self.T-1):    # y = 0, 1,..., 58
            self.apath[y+1] = self.clip(interp1d(self.aa, self.a[y],
                                                kind='cubic')(self.apath[y]))
            if y >= self.W:
                self.cpath[y] = (1 + r[y])*self.apath[y] + b[y] - self.apath[y+1]
                self.npath[y] = 0
            else:
                self.cpath[y], self.npath[y] = self.SolveForCN(y, self.apath[y], 
                                                        self.apath[y+1], r, w, tau, b)
            self.upath[y] = self.util(self.cpath[y], self.npath[y])
        # the oldest generation's consumption and labor supply
        self.cpath[self.T-1] = (1 + r[y])*self.apath[self.T-1] + b[y]
        self.npath[self.T-1] = 0
        self.upath[self.T-1] = self.util(self.cpath[self.T-1], self.npath[self.T-1])


    def clip(self, a):
        return self.aL if a <= self.aL else self.aH if a >= self.aH else a


    def util(self, c, n):
        # calculate utility value with given consumption and labor
        return (((c+self.psi)*(1-n)**self.gamma)**(1-self.sigma)-1)/(1-self.sigma*1.0)


    def uc(self, c, n):
        # marginal utility w.r.t. consumption
        return (c+self.psi)**(-self.sigma)*(1-n)**(self.gamma*(1-self.sigma))


    def un(self, c, n):
        # marginal utility w.r.t. labor
        return -self.gamma*(c+self.psi)**(1-self.sigma)*(1-n)**(self.gamma*(1-self.sigma)-1)


    def IteratePaths(self, RT, a0, r, w, tau, b):
        """ This function numerically finds optimal choices over RT years,
        from T-RT+1 yo to T yo such that asset level at T-RT+1 equals to a0,
        which is the amount of asset that T-RT+1 yo agent holds in the old SS. """
        if RT == 1:
            self.apath[-1] = a0
            self.cpath[-1] = self.apath[-1]*(1 + r[-1]) + b[-1]
            self.npath[-1] = 0
            self.upath[-1] = self.util(self.cpath[-1], self.npath[-1])
        else:
            a1, aT = [-1,], []
            for q in range(self.Nq):
                if q == 0:
                    self.apath[-1] = 0.2
                elif q == 1:
                    self.apath[-1] = 0.3
                else:
                    self.apath[-1] = self.clip(aT[-1]-(aT[-1]-aT[-2])*a1[-1]/(a1[-1]-a1[-2]))
                self.npath[-1] = 0
                self.cpath[-1] = self.apath[-1]*(1 + r[-1]) + b[-1]
                for y in range(-2,-(RT+1),-1):     # y = -2, -3,..., -RT
                    self.apath[y], self.npath[y], self.cpath[y] = \
                                                    self.DirectSolve(y, r, w, tau, b)
                aT.append(self.apath[-1])
                a1.append(self.apath[-RT])
                if (fabs(self.apath[-RT] - a0) < self.tol):
                    break
            for y in range(-1, -(RT+1), -1):
                self.upath[y] = self.util(self.cpath[y], self.npath[y])



    def DirectSolve(self, y, r, w, tau, b):
        """ analytically solve for capital and labor supply given next two periods 
        capital. y is from -2 to -60, i.e., through the next-to-last to the first """
        if y >= -self.R:
            a1 = self.apath[y+1]
            if y == -2:
                a2 = 0
            else:
                a2 = self.apath[y+2]                
            def foc(a):         # FOC for the retired
                c0 = (1 + r[y])*a + b[y] - a1
                c1 = (1 + r[y+1])*a1 + b[y+1] - a2
                return self.uc(c0,0) - self.beta*self.uc(c1,0)*(1 + r[y+1])
            a, n = fsolve(foc, a1), 0
            c = (1 + r[y])*a + b[y] - a1
        else:
            a1 = self.apath[y+1]
            a2 = self.apath[y+2]
            if y == -(self.R+1):
                n1 = 0
                c1 = (1 + r[y+1])*a1 + b[y+1] - a2
            else:
                n1 = self.npath[y+1]
                c1 = (1 + r[y+1])*a1 + (1 - tau[y+1])*w[y+1]*n1 - a2
            def foc((a0,n0)):   # FOC for the workers
                c0 = (1 + r[y])*a0 + (1 - tau[y])*w[y]*n0 - a1
                return self.uc(c0,n0) - self.beta*self.uc(c1,n1)*(1 + r[y+1]),\
                    (1 - tau[y])*w[y]*self.uc(c0,n0) + self.un(c0,n0)
            a, n = fsolve(foc,(a1,n1))
            c = (1 + r[y])*a + (1 - tau[y])*w[y]*n - a1
        return a, n, c


    def OptimalValue(self, y, a0, a1, r, w, tau, b):
        """ Return the value at the given generation and asset a0 and 
        corresponding consumption and labor supply when the agent chooses his 
        next period asset a1, current period consumption c and labor n
        a1 is always within aL and aH """
        if y >= -self.R:    # y = -2, -3, ..., -60
            c, n = (1 + r[y])*a0 + b[y] - a1, 0
        else:
            c, n = self.SolveForCN(y, a0, a1, r, w, tau, b)
        v = self.util(c,n) + self.beta*self.vtilde[y + 1](a1)
        return v if c >= 0 else self.neg


    def SolveForCN(self, y, a0, a1, r, w, tau, b):
        """ Given economy E.prices and next two periods' asset levels
        a generation optimizes on consumption and labor supply at year y """
        def foc((c,n)):
            return (1 + r[y])*a0+(1 - tau[y])*w[y]*n - a1 - c,\
                (1 - tau[y])*w[y]*self.uc(c,n) + self.un(c,n)
                # self.gamma*(c+self.psi)-(1-n)*self.w*(1-self.tau)
        c, n = fsolve(foc,(0.3,0.3))
        """ or, we can analytically solve for c and n
        c = ((1+E.r[y])*a0-a1+(1-E.tau[y])*E.w[y]-self.gamma*self.psi)/(1+self.gamma)
        n = (a1-(1+E.r[y])*a0+c)/(E.w[y]*(1-E.tau[y])) """
        return c, n


"""The following are procedures to get steady state of the economy using direct 
age-profile iteration and projection method"""

""" e = economy(alpha=0.3,delta=0.03,phi=0.9,tol=0.01,TG=4,W=4,R=2)
    gs = [generation(e,beta=0.965,sigma=2,gamma=2,tol=0.001) for y in range(e.TS)]
    direct(e,gs) """


def transition(zeta=0.3, zeta1=0.2, N=40):
    start_time = datetime.now()
    e0 = economy(alpha=0.3,delta=0.03,phi=0.9,tol=0.01,TG=1,W=4,R=2,zeta=zeta)
    e1 = economy(alpha=0.3,delta=0.03,phi=0.9,tol=0.01,TG=1,W=4,R=2,zeta=zeta1)
    
    e0, g0 = direct(e0, generation(e0,beta=0.965,sigma=2,gamma=2,tol=0.001))
    e1, g1 = direct(e1, generation(e1,beta=0.965,sigma=2,gamma=2,tol=0.001))
    
    e1.K[0:e.TS-e.T] = linspace(e0.K[-1],e1.K[0],e.TS-e.T)
    e1.N[0:e.TS-e.T] = linspace(e0.N[-1],e1.N[0],e.TS-e.T)

    gs = [generation(e1,beta=0.965,sigma=2,gamma=2,tol=0.001) for y in range(e.TS)]

    for n in range(N):
        e1.UpdatePrices()
        for t in range(-1, -(e.TS+1), -1):
            if t >= -(e.TS+1) + e.T:
                gs[t].IteratePaths(e.T, 0, r[t-(e.T-1):t+1], w[t-(e.T-1):t+1], tau[t-(e.T-1):t+1], b[t-(e.T-1):t+1])
            else:
                gs[t].IteratePaths((e.TS+1)+t, g0.apath[-(e.TS+1)-t], r[:t+1], w[:t+1], tau[:t+1], b[:t+1])
        e1.Aggregate(gs)
        if e1.Converged:
            print 'Converged! in',i+1,'iterations with tolerance level', e.tol
            break
    print('Duration: {}'.format(end_time - start_time))
    return e1


def direct(e, g, N=40):
    # start_time = datetime.now()
    for i in range(N):
        e.UpdatePrices()
        g.IteratePaths(e.T, 0, e.r, e.w, e.tau, e.b)
        e.Aggregate(g)
        if e.Converged:
            # print 'Converged! in',i+1,'iterations with tolerance level', e.tol
            break
    # end_time = datetime.now()
    # print('Duration: {}'.format(end_time - start_time))
    return e, g


plt.close("all")


def TransitionPath(e):
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
    ax1.plot(e.K)
    ax2.plot(e.N)
    ax3.plot(e.r)
    ax4.plot(e.w)
    ax.set_xlabel('generation')
    ax1.set_title('Capital')
    ax2.set_title('Labor')
    ax3.set_title('Interest Rate')
    ax4.set_title('Wage')
    plt.show()
    # time.sleep(1)
    # plt.close() # plt.close("all")


"""
eco = valuemethod(N=15,R=20,W=40,aN=51) works well.
eco = directmethod(N=50,R=20,W=40,tol=0.001,delta=0.05,alpha=0.3,beta=0.965,phi=0.9)
eco = directmethod(N=50,R=20,W=40,tol=0.001,delta=0.05,alpha=0.3,beta=0.96,phi=0.9)
"""
