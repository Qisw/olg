"""
Jan. 7, 2015, Hyun Chang Yi
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

class economy:
    """ This class is just a "struct" to hold  the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, alpha=0.35, delta=0.08, phi=0.8, tol=0.005, tol10=1e-10,
        tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195,
        r = 0.02, k=0.7, l=0.3, TG=4, W=45, R=30, ng = 0.01):
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.alpha, self.zeta, self.delta = alpha, zeta, delta
        self.phi, self.tol, self.tol10 = phi, tol, tol10
        self.W, self.R = W, R
        self.T = T = (W+R)
        self.TS = TS = (W+R)*TG
        self.sp = loadtxt('sp.txt', delimiter='\n')
        self.ef = loadtxt('ef.txt', delimiter='\n')
        """populations of each cohort with and without early death
                                pop[t][y] is the number of y-yrs-old agents in t"""
        self.mass = array([prod(sp[0:t+1])/(1+ng)**t for t in range(T)], dtype=float)
        self.mass0 = array([1/(1+ng)**t for t in range(T)], dtype=float)
        self.pop = array([self.mass*(1+ng)**t for t in range(TS)], dtype=float)
        self.pop0 = array([self.mass0*(1+ng)**t for t in range(TS)], dtype=float)
        """populations of workers and non-workers when the number of 1-yrs-old workers
                                is normailsed to 1"""
        self.Pw = Pw = sum([self.mass[t] for t in range(0,W)])
        self.Pr = Pr = sum([self.mass[t] for t in range(W,T)])
        self.Pt = Pt = Pw + Pr
        """Initialize variables"""
        L = sum([ef[t]*self.mass[t]*l for t in range(W)]) # aggregate effcient labor
        k = (alpha/(r+delta))**(1/(1-alpha))  # capital per 1 unit of efficient labor
        K = k*L                                 # aggregate capital
        w = (1-alpha)*k**alpha
        y = k**alpha
        Beq = (k*L/Pt)*(sum(self.mass0)-sum(self.mass)) # aggregate Bequest
        tb = zeta*(1-tw)*Pr/(L+zeta*Pr)          # pension contribution
        b = zeta*(1-tw-tb)*w                               # pension benefit
        Tax = tw*w*L+tr*r*(k*L)                              # aggregate Tax income
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


    def Aggregate(self, gs):
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
                    K1[t] = sum([gs[t+y].apath[-(y+1)]*self.pop[t][-(y+1)]
                                    for y in range(T)])
                    L1[t] = sum([gs[t+y].lpath[-(y+1)]*self.pop[t][-(y+1)]
                                    *self.ef[-(y+1)] for y in range(T)])
                    self.Beq[t] = sum([(gs[t+y].apath[-(y+1)]*self.pop0[t][-(y+1)]-
                                            gs[t+y].apath[-(y+1)]*self.pop[t][-(y+1)])
                                                for y in range(T)])
                else:
                    K1[t] = sum([gs[TS-T].apath[-(y+1)]*self.pop[t][-(y+1)]
                                    for y in range(T)])
                    L1[t] = sum([gs[TS-T].lpath[-(y+1)]*self.pop[t][-(y+1)]
                                    *self.ef[-(y+1)] for y in range(T)])
                    self.Beq[t] = sum([(gs[TS-T].apath[-(y+1)]*self.pop0[t][-(y+1)]-
                                            gs[TS-T].apath[-(y+1)]*self.pop[t][-(y+1)])
                                                for y in range(T)])
        self.Converged = (max(absolute(K1-self.K)) < self.tol)
        """ Update the economy's aggregate K and N with weight phi on the old """
        self.K = self.phi*self.K + (1-self.phi)*K1
        self.L = self.phi*self.L + (1-self.phi)*L1


    def UpdateStates(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,TS from last iteration """
        for t in range(TS):
            self.k[t] = self.K[t]/self.L[t]
            self.y[t] = self.k[t]**alpha
            self.r[t] = self.alpha*self.k[t]**(self.alpha-1)-self.delta
            self.w[t] = (1-self.alpha)*self.k[t]**self.alpha
            self.Tax[t] = self.tw[t]*self.w[t]*self.L[t] + self.tr[t]*self.r[t]*self.K[t]
            self.G[t] = self.gy[t]*self.y[t]*self.L[t]
            self.Tr[t] = (self.Tax[t] + self.Beq[t] - self.G[t])/self.Pt
            self.tb[t] = self.zeta*(1-self.tw[t])*self.Pr/(self.L[t]+self.zeta*self.Pr)
            self.b[t] = self.zeta[t]*(1-self.tw[t]-self.tb[t])*self.w[t]
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, self.tb, self.Tr])


class generation:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, beta=0.96, sigma=2.0, gamma=0.32, aH=5.0, aL=0.0, y=0,
        aN=51, Nq=50, psi=0.001, tol=0.005, tol10=1e-10, neg=-1e10, W=45, R=30,
        ng = 0.01):
        self.beta, self.sigma, self.gamma, self.psi = beta, sigma, gamma, psi
        self.R, self.W, self.T, self.y = R, W, W + R, y
        self.aH, self.aL, self.aN, self.aa = aH, aL, aN, linspace(aL,aH,aN)
        self.tol, self.tol10, self.Nq, self.neg = tol, tol10, Nq, neg
        self.sp = loadtxt('sp.txt', delimiter='\n')
        self.ef = loadtxt('ef.txt', delimiter='\n')        
        """ value function and its interpolation """
        self.v = array([[0 for l in range(self.aN)] for y in range(self.T)], dtype=float)
        self.vtilde = [[] for y in range(self.T)]
        """ policy functions used in value function method """
        self.a = array([[0 for l in range(self.aN)] for y in range(self.T)], dtype=float)
        self.c = array([[0 for l in range(self.aN)] for y in range(self.T)], dtype=float)
        self.l = array([[0 for l in range(self.aN)] for y in range(self.T)], dtype=float)
        """ the following paths for a, c, n and u are used in direct and value function methods
        In direct method, those paths are directly calculated, while in the value function
        method the paths are calculated from value and policy functions """
        self.apath = array([0 for y in range(self.T)], dtype=float)
        self.cpath = array([0 for y in range(self.T)], dtype=float)
        self.lpath = array([0 for y in range(self.T)], dtype=float)
        self.upath = array([0 for y in range(self.T)], dtype=float)


    def clip(self, a):
        return self.aL if a <= self.aL else self.aH if a >= self.aH else a


    def util(self, c, l):
        # calculate utility value with given consumption and labor
        return ((c**self.gamma*(1-l)**(1-self.gamma))**(1-self.sigma))/(1-self.sigma)


    def uc(self, c, l):
        # marginal utility w.r.t. consumption
        return self.gamma*self.util(c, l)*(1-self.sigma)/(c*1.0)


    def ul(self, c, l):
        # marginal utility w.r.t. labor
        return -(1-self.gamma)*self.util(c, l)*(1-self.sigma)/(1-l)


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
                    self.apath[-1] = 0.2
                elif q == 1:
                    self.apath[-1] = 0.3
                else:
                    self.apath[-1] = self.clip(aT[-1]-(aT[-1]-aT[-2])*(a1[-1]-a0)/(a1[-1]-a1[-2]))
                self.lpath[-1] = 0
                self.cpath[-1] = self.apath[-1]*(1+(1-tr[-1])*r[-1]) + Tr[-1] + b[-1]
                for y in range(-2,-(RT+1),-1):     # y = -2, -3,..., -RT
                    self.apath[y], self.lpath[y], self.cpath[y] = self.DirectSolve(y, p)
                aT.append(self.apath[-1])
                a1.append(self.apath[-RT])
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
            c = (1 + (1-tr[y])*r[y])*a + (1-tw[y]-tb[y])*w[y]*ef[y]*l + Tr[y] - a1
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
        Kinit=e1.K[0],Ninit=e1.N[0])
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
            print 'Transition Path Converged! in',n+1,'iterations with tolerance level', et.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return et, gs, g0, g1


def direct(e, g, N=50):
    for i in range(N):
        e.UpdateStates()
        g.IteratePaths(e.T, 0, e.p)
        e.Aggregate(g)
        if e.Converged:
            print 'Economy Converged to SS! in',i+1,'iterations with', e.tol
            break
    return e, g


def TransitionPath(e):
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
    ax2.plot(e.N)
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