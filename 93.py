"""
Jan. 12, 2015, Hyun Chang Yi
Computes the model of Section 9.3. in Heer/Maussner using 
direct method from Secion 9.1.

HOUSEHOLD'S UTILITY FUNCTION IS DIFFERENT FROM THAT OF SECTION 9.1. AND 9.2.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod
from matplotlib import pyplot as plt
from datetime import datetime
import time

class world:
    """ This class is just a "struct" to hold  the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, alpha=0.35, delta=0.08, phi=0.8, tol=0.01, tol10=1e-10,
        tr = 0.429, tw = 0.248, zeta=0.15, gy = 0.195,
        k=3.5, l=0.3, TG=4, W=45, R=30, ng = 0.01):
    # tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195,
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.alpha, self.zeta, self.delta = alpha, zeta, delta
        self.phi, self.tol, self.tol10 = phi, tol, tol10
        self.W, self.R = W, R
        self.T = T = (W+R)
        self.TS = TS = (W+R)*TG
        self.sp = sp = loadtxt('sp.txt', delimiter='\n')
        self.ef = ef = loadtxt('ef1.txt', delimiter='\n')
        """populations of each cohort with and without early death
                                pop[t][y] is the number of y-yrs-old agents in t"""
        self.mass = array([prod(sp[0:t+1])/(1+ng)**t for t in range(T)], dtype=float)
        self.mass0 = array([1/(1+ng)**t for t in range(T)], dtype=float)
        # self.mass = self.mass/sum(self.mass)
        # self.mass0 = self.mass0/sum(self.mass0)
        self.pop = array([self.mass*(1+ng)**t for t in range(TS)], dtype=float)
        self.pop0 = array([self.mass0*(1+ng)**t for t in range(TS)], dtype=float)
        """populations of workers and non-workers when the number of 1-yrs-old workers
                                is normailsed to 1"""
        self.Pw = Pw = sum([self.mass[t] for t in range(0,W)])
        self.Pr = Pr = sum([self.mass[t] for t in range(W,T)])
        self.Pt = Pt = Pw + Pr
        """Initialize variables"""
        L = sum([ef[t]*self.mass[t]*l for t in range(W)]) # aggregate effcient labor
        # k = ((r+delta)/alpha)**(1/(1-alpha))  # capital per 1 unit of efficient labor
        K = k*L                                 # aggregate capital
        r = (alpha)*k**(alpha-1) - delta
        w = (1-alpha)*k**alpha
        y = k**alpha
        Beq = (k*L/Pt)*(sum(self.mass0)-sum(self.mass)) # aggregate Bequest
        tb = zeta*(1-tw)*Pr/(L+zeta*Pr)          # pension contribution
        b = zeta*(1-tw-tb)*w                               # pension benefit
        Tax = tw*w*L+tr*r*K                              # aggregate Tax income
        G = gy*y*L                                    # aggregate Govt. expenditure
        Tr = (Tax+Beq-G)/Pt        # per-capita Transfer
        """Construct containers for market prices, tax rates, transfers, aggregate variables"""
        self.tr = array([tr for t in range(TS)], dtype=float)
        self.tw = array([tw for t in range(TS)], dtype=float)
        self.gy = array([gy for t in range(TS)], dtype=float)
        self.L = array([L for t in range(TS)], dtype=float)
        self.K = array([K for t in range(TS)], dtype=float)
        self.Beq = array([Beq for t in range(TS)], dtype=float)
        self.k = array([k for t in range(TS)], dtype=float)
        self.y = array([y for t in range(TS)], dtype=float)
        self.r = array([r for t in range(TS)], dtype=float)
        self.w = array([w for t in range(TS)], dtype=float)
        self.tb = array([tb for t in range(TS)], dtype=float)
        self.b = array([b for t in range(TS)], dtype=float)
        self.Tax = array([Tax for t in range(TS)], dtype=float)
        self.G = array([G for t in range(TS)], dtype=float)
        self.Tr = array([Tr for t in range(TS)], dtype=float)
        # container for r, w, tb, b
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, self.tb, self.Tr])
        # whether the capital stock has converged
        self.Converged = False


    def aggregate(self, gs):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        W, T, TS = self.W, self.T, self.TS
        if TS == T:
            K1, L1 = array([[0 for t in range(TS)] for i in range(2)], dtype=float)
            for t in range(T):
                K1[t] = sum([gs.apath[y]*self.mass[y] for y in range(T)])
                L1[t] = sum([gs.lpath[y]*self.mass[y]*self.ef[y] for y in range(T)])
                """worker's labor supply is product of working hour and efficiency"""
                self.Beq[t] = sum([gs.apath[y]*self.mass0[y]-gs.apath[y]*self.mass[y]
                                        for y in range(T)])
        else:
            """Aggregate all generations' capital and labor supply at each year"""
            K1, L1 = array([[0 for t in range(TS)] for i in range(2)], dtype=float)
            for t in range(TS):
                if t <= TS-T-1:
                    K1[t] = sum([gs[t+y].apath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                    L1[t] = sum([gs[t+y].lpath[-(y+1)]*self.pop[t,-(y+1)]*self.ef[-(y+1)] for y in range(T)])
                    self.Beq[t] = sum([(gs[t+y].apath[-(y+1)]*self.pop0[t,-(y+1)]-
                                            gs[t+y].apath[-(y+1)]*self.pop[t,-(y+1)])
                                                for y in range(T)])
                else:
                    K1[t] = sum([gs[TS-T].apath[-(y+1)]*self.pop[t][-(y+1)]
                                    for y in range(T)])
                    L1[t] = sum([gs[TS-T].lpath[-(y+1)]*self.pop[t][-(y+1)]
                                    *self.ef[-(y+1)] for y in range(T)])
                    self.Beq[t] = sum([(gs[TS-T].apath[-(y+1)]*self.pop0[t][-(y+1)]-
                                            gs[TS-T].apath[-(y+1)]*self.pop[t][-(y+1)])
                                                for y in range(T)])
        self.Converged = (max(absolute(K1-self.K)) < self.tol*max(absolute(self.K)))
        """ Update the economy's aggregate K and N with weight phi on the old """
        self.K = self.phi*self.K + (1-self.phi)*K1
        self.L = self.phi*self.L + (1-self.phi)*L1
        self.k = self.K/self.L
        print "K=%2.2f," %(self.K[0]),"L=%2.2f," %(self.L[0]),"K/L=%2.2f" %(self.k[0])

    def update(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,TS from last iteration """
        for t in range(self.TS):
            # self.k[t] = self.K[t]/self.L[t]
            self.y[t] = self.k[t]**self.alpha
            self.r[t] = self.alpha*self.k[t]**(self.alpha-1)-self.delta
            self.w[t] = (1-self.alpha)*self.k[t]**self.alpha
            self.Tax[t] = self.tw[t]*self.w[t]*self.L[t] + self.tr[t]*self.r[t]*self.K[t]
            self.G[t] = self.gy[t]*self.y[t]*self.L[t]
            self.Tr[t] = (self.Tax[t] + self.Beq[t] - self.G[t])/self.Pt
            self.tb[t] = self.zeta*(1-self.tw[t])*self.Pr/(self.L[t]+self.zeta*self.Pr)
            self.b[t] = self.zeta*(1-self.tw[t]-self.tb[t])*self.w[t]
        print "for r=%2.2f," %(self.r[0]*100), "w=%2.2f," %(self.w[0]), \
                "Tr=%2.2f," %(self.Tr[0]), "b=%2.2f," %(self.b[0]), "beq.=%2.2f," %(self.Beq[0])
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, self.tb, self.Tr])


class cohort:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, beta=0.96, sigma=2.0, gamma=0.4, aH=5.0, aL=0.0, y=100,
        aN=51, Nq=50, psi=0.001, tol=0.005, tol10=1e-10, neg=-1e10, W=45, R=30,
        ng = 0.01, a0 = 0):
        self.beta, self.sigma, self.gamma, self.psi = beta, sigma, gamma, psi
        self.R, self.W, self.y = R, W, y
        self.T = (y+1 if y <= W+R-2 else W+R)
        self.aH, self.aL, self.aN, self.aa = aH, aL, aN, linspace(aL,aH,aN)
        self.a0 = a0
        self.tol, self.tol10, self.Nq, self.neg = tol, tol10, Nq, neg
        self.sp = loadtxt('sp.txt', delimiter='\n')
        self.ef = loadtxt('ef1.txt', delimiter='\n')
        """ value function and its interpolation """
        self.v = array([[0 for i in range(self.aN)] for y in range(self.T)], dtype=float)
        self.vtilde = [[] for y in range(self.T)]
        """ policy functions used in value function method """
        self.a = array([[0 for i in range(self.aN)] for y in range(self.T)], dtype=float)
        self.c = array([[0 for i in range(self.aN)] for y in range(self.T)], dtype=float)
        self.l = array([[0 for i in range(self.aN)] for y in range(self.T)], dtype=float)
        """ the following paths for a, c, n and u are used in direct and value function methods
        In direct method, those paths are directly calculated, while in the value function
        method the paths are calculated from value and policy functions """
        self.apath = array([a0 for y in range(self.T)], dtype=float)
        self.cpath = array([0 for y in range(self.T)], dtype=float)
        self.lpath = array([0 for y in range(self.T)], dtype=float)
        self.upath = array([0 for y in range(self.T)], dtype=float)


    def findvfns(self, p):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle, 
        value and decision functions are calculated ***BACKWARD*** """
        [r, w, b, tr, tw, tb, Tr] = p
        T = self.T
        # y = -1 : the oldest generation
        for i in range(self.aN):
            self.c[-1,i] = self.aa[i]*(1 + (1-tr[-1])*r[-1]) + b[-1] + Tr[-1]
            self.v[-1,i] = self.util(self.c[-1,i], 0)
        self.vtilde[-1] = interp1d(self.aa, self.v[-1], kind='cubic')
        # y = -2, -3,..., -60
        for y in range(-2, -(T+1), -1):
            m0 = 0
            for i in range(self.aN):    # l = 0, 1, ..., 50
                # Find a bracket within which optimal a' lies
                m = max(0, m0)  # Rch91v.g uses m = max(0, m0-1)
                m0, a0, b0, c0 = self.GetBracket(y, i, m, p)
                # Find optimal a' using Golden Section Search
                if a0 == b0:
                    self.a[y,i] = 0
                elif b0 == c0:
                    self.a[y,i] = self.aa[-1]
                else:
                    def objfn(a1): # Define objective function for optimal a'
                        return -self.findv(y, self.aa[i], a1, p)
                    result = minimize_scalar(objfn, bracket=(a0,b0,c0), method='Golden')
                    #‘Brent’,‘Bounded’,‘Golden’
                    self.a[y,i] = result.x
                # Computing consumption and labor
                if y >= -self.R:
                    self.c[y,i] = (1+(1-tr[y])*r[y])*self.aa[i] + b[y] + Tr[y] - self.a[y,i]
                    self.l[y,i] = 0
                else:
                    self.c[y,i], self.l[y,i] = self.findcl(y, self.aa[i], self.a[y,i], p)
                self.v[y,i] = self.util(self.c[y,i], self.l[y,i]) \
                                + self.beta*self.vtilde[y+1](self.a[y,i])
            self.vtilde[y] = interp1d(self.aa, self.v[y], kind='cubic')
        """ find asset and labor supply profiles over life-cycle from value function"""
        # generate each generation's asset, consumption and labor supply forward
        for y in range(T-1):    # y = 0, 1,..., 58
            self.apath[y+1] = self.clip(interp1d(self.aa, self.a[y],
                                            kind='cubic')(self.apath[y]))
            if y >= T-self.R:
                self.cpath[y] = (1+(1-tr[y])*r[y])*self.apath[y] + b[y] + Tr[y] - self.apath[y+1]
                self.lpath[y] = 0
            else:
                self.cpath[y], self.lpath[y] = self.findcl(y, self.apath[y], self.apath[y+1], p)
            self.upath[y] = self.util(self.cpath[y], self.lpath[y])
        # the oldest generation's consumption and labor supply
        self.cpath[T-1] = (1+(1-tr[T-1])*r[T-1])*self.apath[T-1] + b[T-1] + Tr[T-1]
        self.lpath[T-1] = 0
        self.upath[T-1] = self.util(self.cpath[T-1], self.lpath[T-1])


    def GetBracket(self, y, l, m, p):
        """ Find a bracket (a,b,c) such that policy function for next period asset level, 
        a[x;asset[l],y] lies in the interval (a,b) """
        [r, w, b, tr, tw, tb, Tr] = p
        aa = self.aa
        a, b, c = aa[0], aa[0]-aa[1], aa[0]-aa[2]
        m0 = m
        v0 = self.neg
        while (a > b) or (b > c):
            v1 = self.findv(y, aa[l], aa[m], p)
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


    def findv(self, y, a0, a1, p):
        """ Return the value at the given generation and asset a0 and 
        corresponding consumption and labor supply when the agent chooses his 
        next period asset a1, current period consumption c and labor n
        a1 is always within aL and aH """
        [r, w, b, tr, tw, tb, Tr] = p
        if y >= -self.R:    # y = -2, -3, ..., -60
            c, l = (1+(1-tr[y])*r[y])*a0 + b[y] + Tr[y] - a1, 0
        else:
            c, l = self.findcl(y, a0, a1, p)
        v = self.util(c,l) + self.beta*self.vtilde[y+1](a1)
        return v if c >= 0 else self.neg


    def findcl(self, y, a0, a1, p):
        """ Given economy E.prices and next two periods' asset levels
        a generation optimizes on consumption and labor supply at year y """
        [r, w, b, tr, tw, tb, Tr] = p
        def foc((c,l)):
            return (1+(1-tr[y])*r[y])*a0 + (1-tw[y]-tb[y])*w[y]*self.ef[y]*l + Tr[y] - a1 - c,\
                (1-tw[y]-tb[y])*w[y]*self.ef[y]*self.uc(c,l) + self.ul(c,l)
        c, l = fsolve(foc,(0.3,0.3))
        return c, l


    # def findpath(self, p):
    #     """ find asset and labor supply profiles over life-cycle from value function
    #     """
    #     [r, w, b, tr, tw, tb, Tr] = p
    #     T = self.T
    #     # generate each generation's asset, consumption and labor supply forward
    #     for y in range(T-1):    # y = 0, 1,..., 58
    #         self.apath[y+1] = self.clip(interp1d(self.aa, self.a[y],
    #                                         kind='cubic')(self.apath[y]))
    #         if y >= T-self.W:
    #             self.cpath[y] = (1+(1-tr[y])*r[y])*self.apath[y] + b[y] + Tr[y] - self.apath[y+1]
    #             self.lpath[y] = 0
    #         else:
    #             self.cpath[y], self.lpath[y] = self.findcl(y, self.apath[y], self.apath[y+1], p)
    #         self.upath[y] = self.util(self.cpath[y], self.lpath[y])
    #     # the oldest generation's consumption and labor supply
    #     self.cpath[T-1] = (1+(1-tr[T-1])*r[T-1])*self.apath[T-1] + b[T-1] + Tr[T-1]
    #     self.lpath[T-1] = 0
    #     self.upath[T-1] = self.util(self.cpath[T-1], self.lpath[T-1])


    def clip(self, a):
        return self.aL if a <= self.aL else self.aH if a >= self.aH else a


    def util(self, c, l):
        # print 'c,l:', c, l, ((c+self.psi)**self.gamma*(1-l)**(1-self.gamma))
        # calculate utility value with given consumption and labor
        return (((c+self.psi)**self.gamma*(1-l)**(1-self.gamma))**(1-self.sigma))/(1-self.sigma)


    def uc(self, c, l):
        # marginal utility w.r.t. consumption
        return self.gamma*self.util(c, l)*(1-self.sigma)/(c*1.0)


    def ul(self, c, l):
        # marginal utility w.r.t. labor
        return -(1-self.gamma)*self.util(c, l)*(1-self.sigma)/(1-l)


    # def util(self, c, l):
    #     # calculate utility value with given consumption and labor
    #     return (((c+self.psi)*(1-l)**self.gamma)**(1-self.sigma)-1)/(1-self.sigma*1.0)


    # def uc(self, c, l):
    #     # marginal utility w.r.t. consumption
    #     return (c+self.psi)**(-self.sigma)*(1-l)**(self.gamma*(1-self.sigma))


    # def ul(self, c, l):
    #     # marginal utility w.r.t. labor
    #     return -self.gamma*(c+self.psi)**(1-self.sigma)*(1-l)**(self.gamma*(1-self.sigma)-1)


    def IteratePaths(self, RT, a0, p):
        """ This function numerically finds optimal choices over RT years,
        from T-RT+1 to T yrs-old such that asset level at T-RT+1 equals to a0,
        which is the amount of asset that T-RT+1 yrs-old agent holds in the old SS. """
        [r, w, b, tr, tw, tb, Tr] = p
        if RT == 1:
            self.apath[-1] = a0
            self.cpath[-1] = self.apath[-1]*(1+(1-tr[-1])*r[-1]) + Tr[-1] + b[-1]
            self.lpath[-1] = 0
            self.upath[-1] = self.util(self.cpath[-1], self.lpath[-1])
        else:
            a1, aT = [-1,], []
            for q in range(self.Nq):
                if q == 0:
                    self.apath[-1] = 0.23
                elif q == 1:
                    self.apath[-1] = 0.45
                else:
                    self.apath[-1] = self.clip(aT[-1]-(aT[-1]-aT[-2])*(a1[-1]-a0)/(a1[-1]-a1[-2]))
                self.lpath[-1] = 0
                self.cpath[-1] = self.apath[-1]*(1+(1-tr[-1])*r[-1]) + Tr[-1] + b[-1]
                for y in range(-2,-(RT+1),-1):     # y = -2, -3,..., -RT
                    self.apath[y], self.lpath[y], self.cpath[y] = self.DirectSolve(y, p)
                aT.append(self.apath[-1])
                a1.append(self.apath[-RT])
                print 'aT, a1:',aT,a1
                if (absolute(self.apath[-RT] - a0) < self.tol):
                    break
            for y in range(-1,-(RT+1),-1):
                self.upath[y] = self.util(self.cpath[y], self.lpath[y])
        return self.apath, self.cpath, self.lpath, self.upath


    def DirectSolve(self, y, p):
        """ analytically solve for capital and labor supply given next two periods 
        capital. y is from -2 to -60, i.e., through the next-to-last to the first """
        [r, w, b, tr, tw, tb, Tr] = p
        # print y, p.shape
        if y >= -self.R:
            a1 = self.apath[y+1]
            a2 = (0 if y == -2 else self.apath[y+2])
            def foc(a0):         # FOC for the retired
                c0 = (1+(1-tr[y])*r[y])*a0 + b[y] + Tr[y] - a1
                c1 = (1+(1-tr[y+1])*r[y+1])*a1 + b[y+1] + Tr[y+1] - a2
                return (self.beta*self.sp[y+1]*self.uc(c1,0)*(1+(1-tr[y+1])*r[y+1])
                         - self.uc(c0,0))
            a = fsolve(foc, a1)
            l = 0
            c = (1+(1-tr[y])*r[y])*a + b[y] + Tr[y] - a1
        else:
            a1, a2 = self.apath[y+1], self.apath[y+2]
            if y == -(self.R+1):
                l1 = 0
                c1 = (1+(1-tr[y+1])*r[y+1])*a1 + b[y+1] + Tr[y+1] - a2
            else:
                l1 = self.lpath[y+1]
                c1 = ((1-tw[y+1]-tb[y+1])*w[y+1]*self.ef[y+1]*l1 + Tr[y+1] - a2
                         + (1+(1-tr[y+1])*r[y+1])*a1)
            def foc((a0, l0)):   # FOC for the workers
                c0 = (1 + (1-tr[y])*r[y])*a0 + (1-tw[y]-tb[y])*w[y]*self.ef[y]*l0 - a1
                return (1-tw[y]-tb[y])*w[y]*self.ef[y]*self.uc(c0,l0) + self.ul(c0,l0), \
                        (self.beta*self.sp[y+1]*self.uc(c1,l1)*(1+(1-tr[y+1])*r[y+1])
                         - self.uc(c0,l0))
            a, l = fsolve(foc,(a1,l1))
            c = (1 + (1-tr[y])*r[y])*a + (1-tw[y]-tb[y])*w[y]*self.ef[y]*l + Tr[y] - a1
        return a, l, c


"""
The following are procedures to get steady state of the economy using direct 
age-profile iteration and projection method
"""


def transition(zeta=0.3, ng=0.01, ng1=0.0, N=3, W=45, R=30, alpha=0.35, delta=0.06, 
    phi=0.8, TG=4, beta=0.96, gamma=0.35, sigma=2.0, tol=0.001):
    start_time = datetime.now()
    T = W + R
    TS = T*TG
    """Find Old and New Steady States with population growth rates ng and ng1"""
    e0 = economy(alpha=alpha,delta=delta,phi=phi,tol=tol,TG=1,W=W,R=R,ng=ng)
    e1 = economy(alpha=alpha,delta=delta,phi=phi,tol=tol,TG=1,W=W,R=R,ng=ng1)   
    e0, g0 = direct(e0, generation(beta=beta,sigma=sigma,gamma=gamma,tol=tol,W=W,R=R))
    e1, g1 = direct(e1, generation(beta=beta,sigma=sigma,gamma=gamma,tol=tol,W=W,R=R))
    """Initialize Transition Dynamics of Economy for t = 0,...,TS-1"""
    et= economy(alpha=alpha,delta=delta,phi=phi,tol=tol,TG=TG,W=W,R=R,ng=ng1,
        k=e1.K[0],l=e1.L[0])
    et.K[0:TS-T] = linspace(e0.K[-1],e1.K[0],TS-T)
    et.L[0:TS-T] = linspace(e0.L[-1],e1.L[0],TS-T)
    """Construct TS generations who die in t = 0,...,TS-1, respectively"""
    gs = [generation(beta=beta,sigma=sigma,gamma=gamma,tol=tol,W=W,R=R,y=t)
            for t in range(TS)]
    """Iteratively Calculate all generations optimal consumption and labour supply"""
    for n in range(N):
        et.UpdateStates()
        for g in gs:
            if (g.y >= T-1) and (g.y <= TS-(T+1)):
                g.IteratePaths(T, 0, et.p[:,g.y-T+1:g.y+1])
            elif (g.y < T-1):
                g.IteratePaths(g.y+1, g0.apath[T-g.y-1], et.p[:,:g.y+1])
            else:
                g.apath, g.cpath, g.lpath = g1.apath, g1.cpath, g1.lpath
        et.Aggregate(gs)
        if et.Converged:
            print 'Transition Path Converged! in', n+1,'iterations with tolerance level', et.tol
            break
        if n >= N-1:
            print 'Transition Path Not Converged! in', n+1,'iterations with tolerance level', et.tol
            break        
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return et, gs, g0, g1


def direct(e, g, N=10):
    start_time = datetime.now()
    for n in range(N):
        e.UpdateStates()
        g.IteratePaths(e.T, 0, e.p)
        e.Aggregate(g)
        if e.Converged:
            print 'Economy Converged to SS! in',n+1,'iterations with', e.tol
            break
        if n >= N-1:
            print 'Economy Not Converged in',n+1,'iterations with', e.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return e, g


def value(e,g,N=10):
    start_time = datetime.now()
    for n in range(N):
        e.update()
        g.findvfns(e.p)
        e.aggregate(g)
        if e.Converged:
            print 'Economy Converged to SS! in',n+1,'iterations with', e.tol
            break
        if n >= N-1:
            print 'Economy Not Converged in',n+1,'iterations with', e.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return e, g


def steadypath(g):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, top=None, bottom=None)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.plot(g.apath)
    ax2.plot(g.lpath)
    ax3.plot(g.cpath)
    ax4.plot(g.upath)
    ax.set_xlabel('generation')
    ax1.set_title('Asset')
    ax2.set_title('Labor')
    ax3.set_title('Consumption')
    ax4.set_title('Utility')
    plt.show()
    # time.sleep(1)
    # plt.close() # plt.close("all")


def transitionpath(e):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, top=None, bottom=None)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.plot(e.K)
    ax2.plot(e.L)
    ax3.plot(e.r)
    ax4.plot(e.w)
    ax.set_xlabel('generation')
    ax.set_title('R:' + str(e.R) + 'W:' + str(e.W) + 'TS:' + str(e.TS), y=1.08)
    ax1.set_title('Capital')
    ax2.set_title('Labor')
    ax3.set_title('Interest Rate')
    ax4.set_title('Wage')
    plt.show()
    # time.sleep(1)
    # plt.close() # plt.close("all")