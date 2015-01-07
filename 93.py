"""
Jan. 7, 2015, Hyun Chang Yi
Computes the model of Section 9.3. in Heer/Maussner using 
direct method from Secion 9.1.

HOUSEHOLD'S UTILITY FUNCTION IS DIFFERENT FROM THAT OF SECTION 9.1. AND 9.2.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar
from numpy import linspace, mean, array, zeros, absolute, loadtxt
from matplotlib import pyplot as plt
from datetime import datetime
import time

class economy:
    """ This class is just a "struct" to hold  the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, alpha=0.35, delta=0.08, phi=0.8,
        tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195,
        tol=0.005, tol10=1e-10, r0 = 0.02, k0=0.7, n0=0.3, TG=4, W=45, R=30, ng = 0.01):
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.alpha, self.zeta, self.delta = alpha, zeta, delta
        self.phi, self.tol, self.tol10 = phi, tol, tol10
        self.T, self.W, self.R = (W+R), W, R
        self.TS = (W+R)*TG
        k = (alpha/(r0+delta))**(1/(1.0-alpha))
        # market prices and pension benefit from PAYG scheme for T years
        self.r = array([r0 for y in range(self.TS)], dtype=float)
        self.w = array([(1-alpha)*k**alpha for y in range(self.TS)], dtype=float)
        self.b = array([0 for y in range(self.TS)], dtype=float)
        self.Tr = array([0 for y in range(self.TS)], dtype=float)
        # aggregate labor supply and capital
        self.N = array([n0*W/(T*1.0) for y in range(self.TS)], dtype=float)
        self.K = array([K0 for y in range(self.TS)], dtype=float)
        self.k = array([k for y in range(self.TS)], dtype=float)
        self.n = array([n0 for y in range(self.TS)], dtype=float)
        self.y = array([k**alpha for y in range(self.TS)], dtype=float)
        self.c = array([((1-gy)*k**alpha-delta*k)*W/(T*1.0) 
                                for y in range(self.TS)], dtype=float)
        # tax rate that supports replacement rate zeta in PAYG
        self.zeta = array([zeta for y in range(self.TS)], dtype=float)
        self.tr = array([tr for y in range(self.TS)], dtype=float)
        self.tw = array([tw for y in range(self.TS)], dtype=float)
        self.tb = array([self.R*zeta/(self.W+self.R*zeta)
                                for y in range(self.TS)], dtype=float)
        self.sp = loadtxt('sp.txt', delimiter='\n')
        self.efage = loadtxt('efage.txt', delimiter='\n')
        self.mass = [1,]
        for i in range(T-1):
            self.mass.append(mass[-1]*self.sp[i+1]/(1+ng))


        # container for r, w, tb, b
        self.p = array([self.r, self.w, self.tr, self.tw, self.tb, self.Tr, self.b])
        # whether the capital stock has converged
        self.Converged = False


    def Aggregate(self, gs):
        W, T, TS = self.W, self.T, self.TS
        if TS == T:
            K1, N1 = array([[0 for t in range(TS)] for i in range(2)], dtype=float)
            for t in range(T):
                K1[t], N1[t] = mean(gs.apath), mean(gs.npath)
            self.Converged = (absolute(K1[0]-self.K[0]) < self.tol*self.K[0])
        else:
            # Aggregate all generations' capital and labor supply at each year
            K1, N1 = array([[0 for t in range(TS)] for i in range(2)], dtype=float)
            for t in range(TS):
                if t <= TS-T-1:
                    K1[t] = mean([gs[t+y].apath[-(y+1)] for y in range(T)])
                    N1[t] = mean([gs[t+y].npath[-(y+1)] for y in range(T)])
                else:
                    K1[t] = mean([gs[TS-T].apath[-(y+1)] for y in range(T)])
                    N1[t] = mean([gs[TS-T].npath[-(y+1)] for y in range(T)])
            self.Converged = (sum(absolute(K1-self.K)) < self.tol)
            """ Update the economy's aggregate K and N with weight phi on the old """
        self.K = self.phi*self.K + (1-self.phi)*K1
        self.N = self.phi*self.N + (1-self.phi)*N1


    def UpdatePrices(self):
        """ Update market prices, w and r, and pension benefit according to new
        aggregate capital and labor paths for years 0,...,TS from last iteration """
        def CapitalReturn(k):    # interest rate is at least 0.
            return max(self.alpha*k**(self.alpha-1)-self.delta, 0)
        def Wage(k):
            return (1 - self.alpha)*k**self.alpha
        def Benefit(zeta, tb, w, N):
            return zeta*(1 - tb)*w*N*self.T/(self.W*1.0)

        self.w = array([Wage(self.K[t], self.N[t]) 
                                        for t in range(self.TS)], dtype=float)
        self.r = array([CapitalReturn(self.K[t], self.N[t]) 
                                        for t in range(self.TS)], dtype=float)
        self.b = array([Benefit(self.zeta[t], self.tb[t], self.w[t], self.N[t])
                                        for t in range(self.TS)], dtype=float)
        self.p = array([self.r, self.w, self.tb, self.b])


class generation:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, beta=0.96, sigma=2.0, gamma=0.32, aH=5, aL=0, y=0,
        aN=51, Nq=50, psi=0.001, tol=0.005, tol10=1e-10, neg=-1e10, W=40, R=20,
        ng = 0.01):
        self.beta, self.sigma, self.gamma, self.psi = beta, sigma, gamma, psi
        self.R, self.W, self.T, self.y = R, W, W + R, y
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


    def clip(self, a):
        return self.aL if a <= self.aL else self.aH if a >= self.aH else a


    def util(self, c, n):
        # calculate utility value with given consumption and labor
        return ((c**self.gamma*(1-n)**(1-self.gamma))**(1-self.sigma))/(1-self.sigma*1.0)


    def uc(self, c, n):
        # marginal utility w.r.t. consumption
        return (c+self.psi)**(-self.sigma)*(1-n)**(self.gamma*(1-self.sigma))


    def un(self, c, n):
        # marginal utility w.r.t. labor
        return -self.gamma*(c+self.psi)**(1-self.sigma)*(1-n)**(self.gamma*(1-self.sigma)-1)


    def IteratePaths(self, RT, a0, p):
        """ This function numerically finds optimal choices over RT years,
        from T-RT+1 to T yrs-old such that asset level at T-RT+1 equals to a0,
        which is the amount of asset that T-RT+1 yrs-old agent holds in the old SS. """
        [r, w, tb, b] = p
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
                    self.apath[-1] = self.clip(aT[-1]-(aT[-1]-aT[-2])*(a1[-1]-a0)/(a1[-1]-a1[-2]))
                self.npath[-1] = 0
                self.cpath[-1] = self.apath[-1]*(1 + r[-1]) + b[-1]
                for y in range(-2,-(RT+1),-1):     # y = -2, -3,..., -RT
                    self.apath[y], self.npath[y], self.cpath[y] = self.DirectSolve(y, p)
                aT.append(self.apath[-1])
                a1.append(self.apath[-RT])
                if (absolute(self.apath[-RT] - a0) < self.tol):
                    break
            for y in range(-1, -(RT+1), -1):
                self.upath[y] = self.util(self.cpath[y], self.npath[y])
        return self.apath, self.cpath, self.npath, self.upath


    def DirectSolve(self, y, p):
        """ analytically solve for capital and labor supply given next two periods 
        capital. y is from -2 to -60, i.e., through the next-to-last to the first """
        [r, w, tb, b] = p
        # print y, p.shape
        if y >= -self.R:
            a1 = self.apath[y+1]
            a2 = (0 if y == -2 else self.apath[y+2])
            def foc(a):         # FOC for the retired
                c0 = (1 + r[y])*a + b[y] - a1
                c1 = (1 + r[y+1])*a1 + b[y+1] - a2
                return self.uc(c0,0) - self.beta*self.uc(c1,0)*(1 + r[y+1])
            a, n = fsolve(foc, a1), 0
            c = (1 + r[y])*a + b[y] - a1
        else:
            a1, a2 = self.apath[y+1], self.apath[y+2]
            if y == -(self.R+1):
                n1 = 0
                c1 = (1 + r[y+1])*a1 + b[y+1] - a2
            else:
                n1 = self.npath[y+1]
                c1 = (1 + r[y+1])*a1 + (1 - tb[y+1])*w[y+1]*n1 - a2
            def foc((a0,n0)):   # FOC for the workers
                c0 = (1 + r[y])*a0 + (1 - tb[y])*w[y]*n0 - a1
                return self.uc(c0,n0) - self.beta*self.uc(c1,n1)*(1 + r[y+1]),\
                    (1 - tb[y])*w[y]*self.uc(c0,n0) + self.un(c0,n0)
            a, n = fsolve(foc,(a1,n1))
            c = (1 + r[y])*a + (1 - tb[y])*w[y]*n - a1
        return a, n, c


    def OptimalValue(self, y, a0, a1, p):
        """ Return the value at the given generation and asset a0 and 
        corresponding consumption and labor supply when the agent chooses his 
        next period asset a1, current period consumption c and labor n
        a1 is always within aL and aH """
        [r, w, tb, b] = p
        if y >= -self.R:    # y = -2, -3, ..., -60
            c, n = (1 + r[y])*a0 + b[y] - a1, 0
        else:
            c, n = self.SolveForCN(y, a0, a1, r, w, tb, b)
        v = self.util(c,n) + self.beta*self.vtilde[y + 1](a1)
        return v if c >= 0 else self.neg


    def SolveForCN(self, y, a0, a1, p):
        """ Given economy E.prices and next two periods' asset levels
        a generation optimizes on consumption and labor supply at year y """
        [r, w, tb, b] = p
        def foc((c,n)):
            return (1 + r[y])*a0+(1 - tb[y])*w[y]*n - a1 - c,\
                (1 - tb[y])*w[y]*self.uc(c,n) + self.un(c,n)
        c, n = fsolve(foc,(0.3,0.3))
        return c, n


"""
The following are procedures to get steady state of the economy using direct 
age-profile iteration and projection method
"""


sp2 = loadtxt('sp2.txt', delimiter='\n')
efage = loadtxt('efage.txt', delimiter='\n')


def transition(zeta=0.3, zeta1=0.2, N=40, W=40, R=20, alpha=0.3, delta=0.05, 
    phi=0.8, TG=4, beta=0.96, gamma=2, sigma=2, tol=0.001):
    start_time = datetime.now()
    T = W + R
    TS = T*TG
    """Find Old and New Steady States with zeta and zeta1"""
    e0 = economy(alpha=alpha,delta=delta,phi=phi,tol=tol,TG=1,W=W,R=R,zeta=zeta)
    e1 = economy(alpha=alpha,delta=delta,phi=phi,tol=tol,TG=1,W=W,R=R,zeta=zeta1)   
    e0, g0 = direct(e0, generation(beta=beta,sigma=sigma,gamma=gamma,tol=tol,W=W,R=R))
    e1, g1 = direct(e1, generation(beta=beta,sigma=sigma,gamma=gamma,tol=tol,W=W,R=R))
    """Initialize Transition Dynamics of Economy for t = 0,...,TS-1"""
    et= economy(alpha=alpha,delta=delta,phi=phi,tol=tol,TG=TG,W=W,R=R,zeta=zeta1,
        Kinit=e1.K[0],Ninit=e1.N[0])
    et.K[0:TS-T] = linspace(e0.K[-1],e1.K[0],TS-T)
    et.N[0:TS-T] = linspace(e0.N[-1],e1.N[0],TS-T)
    """Construct TS generations who die in t = 0,...,TS-1, respectively"""
    gs = [generation(beta=beta,sigma=sigma,gamma=gamma,tol=tol,W=W,R=R,y=t) for t in range(TS)]
    """Iteratively Calculate all generations optimal consumption and labour supply"""
    for n in range(N):
        et.UpdatePrices()
        for g in gs:
            if (g.y >= T-1) and (g.y <= TS-(T+1)):
                g.IteratePaths(T, 0, et.p[:,g.y-T+1:g.y+1])
            elif (g.y < T-1):
                g.IteratePaths(g.y+1, g0.apath[T-g.y-1], et.p[:,:g.y+1])
            else:
                g.apath, g.cpath, g.npath = g1.apath, g1.cpath, g1.npath
        et.Aggregate(gs)
        if et.Converged:
            print 'Transition Path Converged! in',n+1,'iterations with tolerance level', et.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return et, gs, g0, g1


def direct(e, g, N=40):
    for i in range(N):
        e.UpdatePrices()
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