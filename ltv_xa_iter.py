# -*- coding: utf-8 -*-
"""
Jan. 12, 2015, Hyun Chang Yi
Computes the model of Section 9.3. in Heer/Maussner using 
direct method from Secion 9.1.

HOUSEHOLD'S UTILITY FUNCTION IS DIFFERENT FROM THAT OF SECTION 9.1. AND 9.2.
"""

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from numpy import linspace, array, absolute, loadtxt, prod, log, random, nonzero
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle
import os

from multiprocessing import Process, Lock, Manager


class state:
    """ This class is just a "struct" to hold  the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, alpha=0.3, delta=0.08, phi=0.8, tol=0.01, Hs = 60,
        tr = 0.15, tw = 0.24, zeta=0.35, gy = 0.195, qh = 10, qr = 0.3,
        k=3.5, l=0.3, TG=4, W=45, R=30, Tr=0.06, b=0.26):
        # tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195, in Section 9.3. in Heer/Maussner
        # tr = 0.01, tw = 0.11, zeta=0.15, gy = 0.195, qh = 0.1, qr = 0.09,
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.alpha, self.zeta, self.delta = alpha, zeta, delta
        self.phi, self.tol = phi, tol
        self.Hs = Hs
        self.W, self.R = W, R
        self.T = T = (W+R)
        self.TS = TS = (W+R)*TG
        self.sp = sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.pop = array([prod(sp[:t+1]) for t in range(T)], dtype=float)
        """Construct containers for market prices, tax rates, transfers, 
        other aggregate variables"""
        self.Pt = Pt = sum(self.pop)
        self.Pr = Pr = sum([self.pop[y] for y in range(W,T)])
        self.tr = tr
        self.tw = tw
        self.gy = gy
        self.k = k
        self.L = L = Pt-Pr
        self.h = array([0 for y in range(T)], dtype=float)
        self.rent = array([0 for y in range(T)], dtype=float)
        self.a = array([0 for y in range(T)], dtype=float)
        self.d = array([0 for y in range(T)], dtype=float)
        self.ah = array([0 for y in range(T)], dtype=float)
        self.c = array([0 for y in range(T)], dtype=float)
        self.l = array([0 for y in range(T)], dtype=float)
        self.beq = array([0 for y in range(T)], dtype=float)
        self.H = 0
        self.R = 0
        self.A = 0
        self.AH = 0
        self.D = 0
        self.C = 0
        self.DebtRatio = 0
        
        self.K = K = k*L
        self.Beq = Beq = 0.21
        self.y = y = k**alpha
        self.r = r = (alpha)*k**(alpha-1) - delta
        self.w = w = (1-alpha)*k**alpha
        self.tb = tb = zeta*(1-tw)*Pr/(L+zeta*Pr)
        self.b = b #zeta*(1-tw-tb)*w
        
        self.Tax = Tax = 11.0
        self.G = G = 6.72
        self.Tr = Tr
        self.qh = qh
        self.qr = qr
        # container for r, w, b, tr, tw, tb, Tr
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, 
                            self.tb, self.Tr, self.qh, self.qr])


    def sum(self, gs):
        T = self.T
        gn = float(len(gs))
        self.h = array([sum([g.hh[g.hpath[y]] for g in gs]) for y in range(T)], dtype=float)/gn
        self.rent = array([sum([g.rpath[y] for g in gs]) for y in range(T)], dtype=float)/gn
        self.a = array([sum([g.apath[y] for g in gs]) for y in range(T)], dtype=float)/gn
        self.d = array([sum([-g.apath[y]*(g.apath[y]<0) for g in gs]) for y in range(T)], dtype=float)/gn
        self.c = array([sum([g.cpath[y] for g in gs]) for y in range(T)], dtype=float)/gn
        self.l = array([sum([g.epath[y] for g in gs]) for y in range(T)], dtype=float)/gn
        self.ah = self.qh*self.h + self.a

        self.Beq = sum(array([(self.a[y] + self.qh*self.h[y])*self.pop[y]*(1-self.sp[y]) for y in range(T)], dtype=float))
        self.H = sum([self.h[y]*self.pop[y] for y in range(T)])
        self.R = sum([self.rent[y]*self.pop[y] for y in range(T)])
        self.A = sum([self.a[y]*self.pop[y] for y in range(T)])
        self.D = sum([self.d[y]*self.pop[y] for y in range(T)])
        self.C = sum([self.c[y]*self.pop[y] for y in range(T)])
        self.L = sum([self.l[y]*self.pop[y] for y in range(T)])
        self.AH = sum([self.ah[y]*self.pop[y] for y in range(T)])
        self.DebtRatio = self.D/self.C


    def currentstate(self):
        """ print market prices and others like tax """
        print "r=%2.2f," %(self.r*100),"w=%2.2f" %(self.w),'\n',\
                "Bequest=%2.2f" %(self.Beq), "Tr=%2.2f" %(self.Tr),"tb=%2.2f" %(self.tb),\
                "b=%2.2f" %(self.b),'\n',"qh=%2.5f" %(self.qh),\
                "H=%3.2f," %(self.H),'\n',"qr=%2.5f" %(self.qr),\
                "R=%2.2f" %(self.R),'\n',"Debt=%2.2f" %(self.D), "Cons=%2.2f" %(self.C),\
                "Debt Ratio=%2.2f" %(self.DebtRatio), "Liquid Asset=%2.2f" %(self.A)


    def update(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,TS from last iteration """
        self.Tax = self.tw*self.w*self.L + self.tr*self.r*(self.A+self.D)
        self.G = self.gy*self.C
        self.Tr = (self.Tax+self.Beq-self.G)/self.Pt
        self.tb = self.zeta*(1-self.tw)*self.Pr/(self.L+self.zeta*self.Pr)
        self.b = self.zeta*(1-self.tw-self.tb)*self.w
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, self.tb, 
                        self.Tr, self.qh, self.qr])


class cohort:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, beta=0.96, sigma=2.0, gamma=1, aH=5.0, aL=0.0, y=-1, dti = 0.5,
        aN=101, psi=0.03, tol=0.01, neg=-1e5, W=45, R=30, a0 = 0, tcost = 0.0, ltv=0.7):
        self.beta, self.sigma, self.gamma, self.psi = beta, sigma, gamma, psi
        self.R, self.W, self.y = R, W, y
        self.tcost, self.ltv, self.dti = tcost, ltv, dti
        self.T = T = (y+1 if (y >= 0) and (y <= W+R-2) else W+R)
        self.aH, self.aL, self.aN = aH, aL, aN 
        self.aa = (aL+aH)/2.0+(aH-aL)/2.0*linspace(-1,1,aN)
        self.tol, self.neg = tol, neg
        """ house sizes and number of feasible feasible house sizes """
        # self.hh = array([0.0, 0.4, 1.0])
        self.hh = array([0.0, 0.6, 1.0])
        # self.hh = loadtxt('hh.txt', delimiter='\n')
        self.sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.hN = hN = len(self.hh)
        """ age-specific productivity """
        self.ef = loadtxt('ef.txt', delimiter='\n')
        """ employment status: if unemployed, 40% of market wage is paid """
        self.xx = [1, 0.4]
        self.xN = xN = len(self.xx)
        self.uu = 0.2
        self.ee = (23+self.uu)/24.0  # invariant distribution is (0.96 0.04)
        self.trx = array([[self.ee, 1-self.ee], [1-self.uu, self.uu]], dtype=float)
        """ value function and its interpolation """
        self.v = array([[[[0 for i in range(aN)] for h in range(hN)] for x in range(xN)] for y in range(T)], dtype=float)
        self.vtilde = [[[[] for h in range(hN)] for x in range(xN)] for y in range(T)]
        """ policy functions and interpolation for optimal next period asset """
        self.ho = array([[[[0 for i in range(aN)] for h in range(hN)] for x in range(xN)] for y in range(T)], dtype=int)
        self.ao = array([[[[[0 for i in range(aN)] for h1 in range(hN)] for h0 in range(hN)] for x in range(xN)] for y in range(T)], dtype=float)
        self.atilde = [[[[[] for h1 in range(hN)] for h0 in range(hN)] for x in range(xN)] for y in range(T)]
        self.co = array([[[[[0 for i in range(aN)] for h1 in range(hN)] for h0 in range(hN)] for x in range(xN)] for y in range(T)], dtype=float)
        self.ro = array([[[[[0 for i in range(aN)] for h1 in range(hN)] for h0 in range(hN)] for x in range(xN)] for y in range(T)], dtype=float)
        """ the following paths for a, c, n and u are used in direct and value function methods
        In direct method, those paths are directly calculated, while in the value function
        method the paths are calculated from value and policy functions """
        self.apath = array([0 for y in range(T)], dtype=float)
        self.hpath = array([0 for y in range(T)], dtype=int)
        self.xpath = array([0 for y in range(T)], dtype=int)
        self.cpath = array([0 for y in range(T)], dtype=float)
        self.rpath = array([0 for y in range(T)], dtype=float)
        self.epath = array([0 for y in range(T)], dtype=float) # labor supply in efficiency unit
        self.upath = array([0 for y in range(T)], dtype=float)


    def valueNpolicy(self, p):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle, 
        value and decision functions are calculated ***BACKWARD*** """
        [r, w, b, tr, tw, tb, Tr, qh, qr] = p
        T, aa, hh, aN, hN = self.T, self.aa, self.hh, self.aN, self.hN
        # y = -1 : the oldest generation
        for h0 in range(hN):
            for i in range(aN):
                di, self.co[-1,0,h0,0,i], self.ro[-1,0,h0,0,i] = self.findcr(-1, 0, h0, 0, aa[i], 0, p)
                self.v[-1,0,h0,i] = self.util(self.co[-1,0,h0,0,i], self.ro[-1,0,h0,0,i]+hh[h0])
            self.vtilde[-1][0][h0] = interp1d(aa, self.v[-1,0,h0], kind='cubic')
            self.atilde[-1][0][h0][0] = interp1d(aa, linspace(0,0,aN), kind='cubic')
        # y = -2, -3,..., -75
        for y in range(-2, -(T+1), -1):
            """ adjust productivity grid and transition matrix according to age """
            if y >= -(self.R):
                xx, nxx, trx = array([0]), array([0]), array([[1]])
            elif y == -(self.R+1):
                xx, nxx, trx = self.xx, array([0]), array([[1] for x0 in range(self.xN)])
            else:
                xx, nxx, trx = self.xx, self.xx, self.trx
            for x0 in range(len(xx)):
                income = Tr + b if y >= -self.R else Tr + (1-tw-tb)*w*self.ef[y]*xx[x0]
                mdti = (aa + income*self.dti >= 0).argmax() # DTI constraint
                for h0 in range(hN):
                    at = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                    ct = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                    rt = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                    vt = array([[self.neg for i in range(aN)] for h in range(hN)], dtype=float)
                    # print self.apath
                    for h1 in range(hN):
                        mltv = (aa + self.ltv*qh*hh[h1] >= 0).argmax() # LTV constraint
                        m0 = max(0,mltv,mdti)
                        for i in range(aN):    # l = 0, 1, ..., 50
                            # Find a bracket within which optimal a' lies
                            m = max(0, m0)  # Rch91v.g uses m = max(0, m0-1)
                            # print m
                            m0, a0, b0, c0 = self.GetBracket(y, x0, h0, h1, i, m, p)
                            if a0 == b0:
                                at[h1,i] = aa[m0]
                            elif b0 == c0:
                                at[h1,i] = self.aH
                            else:
                                def objfn(a1): # Define objective function for optimal a'
                                    return -self.findv(y, x0, h0, h1, aa[i], a1, p)
                                result = minimize_scalar(objfn, bracket=(a0,b0,c0), method='Golden')
                                at[h1,i] = result.x
                            # Compute consumption, rent and house
                            di, ct[h1,i], rt[h1,i] = self.findcr(y, x0, h0, h1, aa[i], at[h1,i], p)
                            ev = sum([self.vtilde[y+1][x1][h1](at[h1,i])*trx[x0,x1] for x1 in range(len(nxx))])
                            vt[h1,i] = self.util(ct[h1,i],rt[h1,i]+hh[h0]) + self.beta*self.sp[y]*ev
                            self.ao[y,x0,h0,h1,i] = at[h1,i]
                            self.co[y,x0,h0,h1,i] = ct[h1,i]
                            self.ro[y,x0,h0,h1,i] = rt[h1,i]
                        self.atilde[y][x0][h0][h1] = interp1d(aa, self.ao[y,x0,h0,h1], kind='cubic')
                    for i in range(self.aN):
                        h1 = vt[:,i].argmax()
                        self.ho[y,x0,h0,i] = h1
                        self.v[y,x0,h0,i] = vt[h1,i]
                    # print nonzero(self.ho[y,x0,h0]==0)[0][0],y,x0,h0
                    if len(nonzero(self.ho[y,x0,h0]==0)[0]):
                        for i in range(nonzero(self.ho[y,x0,h0]==0)[0][0]):
                            self.ho[y,x0,h0,i] = 0
                    self.vtilde[y][x0][h0] = interp1d(aa, self.v[y,x0,h0], kind='cubic')
                    # ai = [self.aN/4, self.aN*2/4, self.aN-1]
                    # if y % 15 == 0 and h0 == 0 and x0 == 0:
                    #     print '------------'
                    #     print 'y=',y,'inc=%2.2f'%(income),'h0=',h0
                    #     for i in ai:
                    #         print 'ltv=%2.2f'%(hh[self.ho[y,x0,h0,i]]*qh*self.ltv),'dti=%2.2f'%(income*self.dti),\
                    #                 'a0=%2.2f'%(aa[i]),'h1=%2.0f'%(self.ho[y,x0,h0,i]),\
                    #                 'a1=%2.2f'%(self.ao[y,x0,h0,self.ho[y,x0,h0,i],i]),\
                    #                 'c0=%2.2f'%(self.co[y,x0,h0,self.ho[y,x0,h0,i],i]),\
                    #                 'r0=%2.2f'%(self.ro[y,x0,h0,self.ho[y,x0,h0,i],i])


    def nextx(self, y, x0):
        """ return next period nx given current period x """
        if y >= self.T-(self.R+1):
            return 0
        else:
            return random.binomial(1,self.trx[x0,1])
            # return 1*(random.random()>self.uu) if x == 0 else 1*(random.random()>(1-self.ee))


    def GetBracket(self, y, x0, h0, h1, l, m, p):
        """ Find a bracket (a,b,c) such that policy function for next period asset level, 
        a[x;asset[l],y] lies in the interval (a,b) """
        aa = self.aa
        a, b, c = aa[0], 2*aa[0]-aa[1], 2*aa[0]-aa[2]
        minit = m
        m0 = m
        v0 = self.neg
        """ The slow part of if slope != float("inf") is no doubt converting 
        the string to a float. """
        while (a > b) or (b > c):
            v1 = self.findv(y, x0, h0, h1, aa[l], aa[m], p)
            if v1 > v0:
                a, b, = ([aa[m], aa[m]] if m == minit else [aa[m-1], aa[m]])
                v0, m0 = v1, m
            else:
                c = aa[m]
            if m == self.aN - 1:
                a, b, c = aa[m-1], aa[m], aa[m]
            m = m + 1
        return m0, a, b, c


    def findv(self, y, x0, h0, h1, a0, a1, p):
        """ Return the value at the given generation and asset a0 and 
        corresponding consumption and labor supply when the agent chooses his 
        next period asset a1, current period consumption c and labor n
        a1 is always within aL and aH """
        if y >= -(self.R):
            nxx, trx = array([0]), array([[1]])
        elif y == -(self.R+1):
            nxx, trx = array([0]), array([[1] for x0 in range(self.xN)])
        else:
            nxx, trx = self.xx, self.trx

        di, co, ro = self.findcr(y, x0, h0, h1, a0, a1, p)

        if co > 0:
            ev = sum([self.vtilde[y+1][x1][h1](a1)*trx[x0,x1] for x1 in range(len(nxx))])
            value = self.util(co, ro+self.hh[h0]) + self.beta*self.sp[y]*ev
        else:
            value = self.neg
        return value


    def findcr(self, y, x0, h0, h1, a0, a1, p):
        """ FIND budget, consumption and rent given next period house and asset """
        [r, w, b, tr, tw, tb, Tr, qh, qr] = p

        income = b + Tr if y >= -self.R else (1-tw-tb)*w*self.ef[y]*self.xx[x0] + Tr

        di = a0*(1+(1-tr*(a0>0))*r) + (self.hh[h0]-self.hh[h1])*qh \
                - self.hh[h0]*qh*(h0!=h1)*self.tcost - a1 + income
        co = (di + qr*(self.hh[h0]+self.gamma))/(1+self.psi)
        ro = (di*self.psi-qr*(self.hh[h0]+self.gamma))/((1+self.psi)*qr)
        return di, co, ro


    def clip(self, a):
        return self.aL if a <= self.aL else self.aH if a >= self.aH else a


    def util(self, c, s):
        # calculate utility value with given consumption and housing service
        return log(c) + self.psi*log(s+self.gamma) if c > 0 else self.neg


class agent:
    """ This class is just a "struct" to hold an agent's lifecycle behaviour """
    def __init__(self, g):
        self.hh = g.hh
        self.T = T = g.T
        """ the following paths for a, c, n and u are used in direct and value function methods
        In direct method, those paths are directly calculated, while in the value function
        method the paths are calculated from value and policy functions """
        self.apath = array([0 for y in range(T)], dtype=float)
        self.hpath = array([0 for y in range(T)], dtype=int)
        self.xpath = array([0 for y in range(T)], dtype=int)
        self.cpath = array([0 for y in range(T)], dtype=float)
        self.rpath = array([0 for y in range(T)], dtype=float)
        self.epath = array([0 for y in range(T)], dtype=float) # labor supply in efficiency unit
        self.upath = array([0 for y in range(T)], dtype=float)
        self.ppath = array([0 for y in range(T)], dtype=float)


    def simulatelife(self, e, g, ainit=0, hinit=0, xinit=0):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle, 
        value and decision functions are calculated ***BACKWARD*** """
        T, hh, aa = g.T, g.hh, g.aa
        """ find asset and labor supply profiles over life-cycle from value function"""
        # generate each generation's asset, consumption and labor supply forward
        self.apath[0], self.hpath[0], self.xpath[0] = ainit, hinit, xinit
        for y in range(T-1):    # y = 0, 1,..., 58
            if y >= T-(g.R):
                nxx, trx = array([0]), array([[1]])
            elif y == T-(g.R+1):
                nxx, trx = array([0]), array([[1] for x0 in range(g.xN)])
            else:
                nxx, trx = g.xx, g.trx

            for h1 in reversed(range(g.hN)):
                if len(nonzero(g.ho[y,self.xpath[y],self.hpath[y]]==h1)[0]):
                    if self.apath[y] >= aa[nonzero(g.ho[y,self.xpath[y],self.hpath[y]]==h1)[0][0]]:
                        self.hpath[y+1] = h1
                        break

            self.apath[y+1] = g.clip(g.atilde[y][self.xpath[y]][self.hpath[y]][self.hpath[y+1]](self.apath[y]))
            di, c0, r0 = g.findcr(y-T, self.xpath[y], self.hpath[y], self.hpath[y+1], self.apath[y], self.apath[y+1], e.p)
            ev = sum([g.vtilde[y+1][x1][self.hpath[y+1]](self.apath[y+1])*trx[self.xpath[y],x1] for x1 in range(len(nxx))])
            v1 = g.util(c0, r0+hh[self.hpath[y]]) + g.beta*g.sp[y]*ev
            self.cpath[y], self.rpath[y] = c0, r0
            self.upath[y] = g.util(self.cpath[y], self.rpath[y]+hh[self.hpath[y]])
            self.xpath[y+1] = g.nextx(y,self.xpath[y])
        # the oldest generation's consumption and labor supply
        di, self.cpath[T-1], self.rpath[T-1] = g.findcr(-1, 0, self.hpath[-1], 0, self.apath[-1], 0, e.p)
        self.upath[T-1] = g.util(self.cpath[T-1], self.rpath[T-1]+hh[self.hpath[T-1]])
        self.epath = [g.ef[y]*g.xx[self.xpath[y]] for y in range(T)]
        self.ppath = [self.apath[y] + e.qh*g.hh[self.hpath[[y]]] for y in range(T)]


def agentpath(e, g, a):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, top=None, bottom=None)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.plot(a.apath)
    ax4.plot(g.hh[a.hpath])
    ax2.plot(a.cpath)
    ax5.plot(a.rpath)
    ax6.plot(a.xpath)
    ax3.plot(a.ppath)
    ax.set_xlabel('Age')
    ax1.set_title('Liquid Asset')
    ax4.set_title('House')
    ax2.set_title('Consumption')
    ax5.set_title('Rent')
    ax6.set_title('Productivity')
    ax3.set_title('Total Asset')
    ax4.axis([0, 80, 0, 1.1])
    plt.show()
    # time.sleep(1)
    # plt.close() # plt.close("all")    

def agepath(e, g):
    title = 'i=%2.2f'%(e.r*100) + 'ltv=' + str(g.ltv) + 'qh=' + str(e.qh) + 'qr=' + str(e.qr) \
                + 'H=%2.2f'%(e.H) + 'R=%2.2f'%(e.R) +  'Debt=%2.2f'%(e.DebtRatio) + '.png'
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, top=None, bottom=None)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.plot(e.a)
    ax4.plot(e.h)
    ax2.plot(e.c)
    ax5.plot(e.rent)
    ax6.plot(e.l)
    ax3.plot(e.ah)
    ax.set_xlabel('Age')
    ax.set_title(title, y=1.08)
    ax1.set_title('Liquid Asset')
    ax4.set_title('House')
    ax2.set_title('Consumption')
    ax5.set_title('Rent')
    ax6.set_title('Productivity')
    ax3.set_title('Total Asset')
    ax4.axis([0, 80, 0, 1.1])
    path = 'D:\LTV\olg2'
    fullpath = os.path.join(path, title)
    fig.savefig(fullpath)
    # plt.show()
    # time.sleep(1)
    # plt.close() # plt.close("all")    

# def main1(psi=0.1, qh=1.050, qr=0.04145, ltv=0.4, dti=1.2, tcost=0.05):
#     e = state(TG=1, k=4.2, ng=1, dng=0, W=45, R=30, Hs=60, qh=qh, qr=qr)
#     e.printprices()
#     g = cohort(W=45, R=30, psi=psi, beta=0.96, tcost=tcost, gamma=0.99, aL=-1.0, aH=2.0, aN=251, ltv=ltv, dti=dti)
#     g.valueNpolicy(e.p)
#     g.simulatelife(e.p)
#     e.aggregate([g])
#     # e.update()
#     # e.printprices()
#     spath(e, g)
#     return e.Hd_Hs[-1], e.Rd[-1], e.r[-1], e.Dratio[-1]

if __name__ == '__main__':
    qh = 1.0000
    qr = 0.0400
    for i in range(20):
        e = state(TG=1, k=4.2, W=45, R=30, Hs=60, qh=qh, qr=qr, Tr=0.06, b=0.26)
        # e.currentstate()
        g = cohort(W=45, R=30, psi=0.1, beta=0.96, tcost=0.05, gamma=0.99, aL=-1.0, aH=1.0, aN=301, ltv=0.8, dti=2.0)
        g.valueNpolicy(e.p)
        print 'simulation starts...'
        agents = [agent(g) for i in range(10000)]
        for a in agents:
            a.simulatelife(e, g, xinit=random.binomial(1,0.04))
        e.sum(agents)
        e.update()
        e.currentstate()
        agepath(e, g)
        if absolute(e.H-35)<1.0 and absolute(e.R)<1.0:
            break
        else:
            qh = qh*(1+0.01*(e.H-35))
            qr = qr*(1+0.001*(e.R))
    

# def gridsearch():
#     qhN = qrN = 5
#     qh = linspace(0.98,1.20,qhN)
#     qr = linspace(0.039,0.042,qrN)
#     House = array([[0 for i in range(qrN)] for j in range(qhN)])
#     Rent = array([[0 for i in range(qrN)] for j in range(qhN)])
#     Rate = array([[0 for i in range(qrN)] for j in range(qhN)])
#     Dr = array([[0 for i in range(qrN)] for j in range(qhN)])
#     for i in range(qhN):
#         for j in range(qrN):
#             House[i,j], Rent[i,j], Rate[i,j], Dr[i,j] = main1(psi=0.1, qh=qh[i], qr=qr[j], ltv=1.0, dti=1.2, tcost=0.05)
#     with open('hrr10.pickle','wb') as f:
#         pickle.dump([House, Rent, Rate, Dr], f)