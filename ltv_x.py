# -*- coding: utf-8 -*-
"""
Jan. 12, 2015, Hyun Chang Yi
Computes the model of Section 9.3. in Heer/Maussner using 
direct method from Secion 9.1.

HOUSEHOLD'S UTILITY FUNCTION IS DIFFERENT FROM THAT OF SECTION 9.1. AND 9.2.
"""

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from numpy import linspace, array, absolute, loadtxt, prod, log, random
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
        k=3.5, l=0.3, TG=4, W=45, R=30, ng = 1.01, dng = 0.0):
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
        ng0, ng1 = ng, ng + dng
        sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        m0 = array([prod(sp[:t+1])/ng0**t for t in range(T)], dtype=float)
        m1 = array([prod(sp[:t+1])/ng1**t for t in range(T)], dtype=float)
        self.sp = array([sp for t in range(TS)], dtype=float)
        self.pop = array([m1*ng1**(t+1) for t in range(TS)], dtype=float)
        for t in range(T-1):
            self.pop[t,t+1:] = m0[t+1:]*ng0**(t+1)
        """Construct containers for market prices, tax rates, transfers, other aggregate variables"""
        self.Pt = Pt = array([sum(self.pop[t]) for t in range(TS)], dtype=float) 
        self.Pr = Pr = array([sum([self.pop[t,y] for y in range(W,T)]) for t in range(TS)], dtype=float) 
        self.tr = array([tr for t in range(TS)], dtype=float)
        self.tw = array([tw for t in range(TS)], dtype=float)
        self.gy = array([gy for t in range(TS)], dtype=float)
        self.k = array([k for t in range(TS)], dtype=float)
        self.L = L = array([(Pt[t]-Pr[t]) for t in range(TS)], dtype=float)
        self.h = array([0 for t in range(TS)], dtype=float)
        self.r = array([0 for t in range(TS)], dtype=float)
        self.d = array([0 for t in range(TS)], dtype=float)
        self.c = array([0 for t in range(TS)], dtype=float)
        #self.L = L = array([l*(Pt[t]-Pr[t]) for t in range(TS)], dtype=float)
        self.K = K = array([k*L[t] for t in range(TS)], dtype=float)
        self.C = C = array([0 for t in range(TS)], dtype=float)
        self.Beq = Beq = array([K[t]/Pt[t]*sum((1-self.sp[t])*self.pop[t]) for t in range(TS)], dtype=float)
        self.y = y = array([k**alpha for t in range(TS)], dtype=float)
        self.r = r = array([(alpha)*k**(alpha-1) - delta for t in range(TS)], dtype=float)
        self.w = w = array([(1-alpha)*k**alpha for t in range(TS)], dtype=float)
        self.tb = tb = array([zeta*(1-tw)*Pr[t]/(L[t]+zeta*Pr[t]) for t in range(TS)], dtype=float)
        self.b = array([zeta*(1-tw-tb[t])*w[t] for t in range(TS)], dtype=float)
        self.Tax = Tax = array([tw*w[t]*L[t]+tr*r[t]*K[t] for t in range(TS)], dtype=float)
        self.G = G = array([gy*y[t]*L[t] for t in range(TS)], dtype=float)
        self.Tr = array([(Tax[t]+Beq[t]-G[t])/Pt[t] for t in range(TS)], dtype=float)
        self.qh = array([qh for t in range(TS)], dtype=float)
        self.qr = array([qr for t in range(TS)], dtype=float)
        # container for r, w, b, tr, tw, tb, Tr
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, self.tb, self.Tr, self.qh, self.qr])
        # whether the capital stock has converged
        self.Converged = False


    def sum(self, gs):
        T = self.T
        self.h = array([sum([g.hpath[t] for g in gs]) for t in range(T)], dtype=float)
        self.r = array([sum([g.rpath[t] for g in gs]) for t in range(T)], dtype=float)
        self.d = array([sum([g.apath[t]*(g.apath[t]<0) for g in gs]) for t in range(T)], dtype=float)
        self.c = array([sum([g.cpath[t] for g in gs]) for t in range(T)], dtype=float)

        H = sum([self.h[y]*self.pop[-1,y] for y in range(T)])
        R = sum([self.r[y]*self.pop[-1,y] for y in range(T)])
        D = sum([self.d[y]*self.pop[-1,y] for y in range(T)])
        C = sum([self.c[y]*self.pop[-1,y] for y in range(T)])
        self.Dratio = -D/C
        return H, R, D, C


    def aggregate(self, gs):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        W, T, TS = self.W, self.T, self.TS
        """Aggregate all cohorts' capital and labor supply at each year"""
        K1, L1, H1, R1 = array([[0 for t in range(TS)] for i in range(4)], dtype=float)
        for t in range(TS):
            if t <= TS-T-1:
                K1[t] = sum([gs[t+y].apath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                L1[t] = sum([gs[t+y].epath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                H1[t] = sum([gs[t+y].hpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                R1[t] = sum([gs[t+y].rpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                self.D[t] = sum([gs[t+y].apath[-(y+1)]*(gs[t+y].apath[-(y+1)]<0)*self.pop[t,-(y+1)] for y in range(T)])
                self.Beq[t] = sum([gs[t+y].apath[-(y+1)]*self.pop[t,-(y+1)]
                                    /self.sp[t,-(y+1)]*(1-self.sp[t,-(y+1)]) for y in range(T)])
                self.C[t] = sum([gs[t+y].cpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
            else:
                K1[t] = sum([gs[-1].apath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                L1[t] = sum([gs[-1].epath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                H1[t] = sum([gs[-1].hpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                R1[t] = sum([gs[-1].rpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                self.D[t] = sum([gs[-1].apath[-(y+1)]*(gs[-1].apath[-(y+1)]<0)*self.pop[t,-(y+1)] for y in range(T)])
                self.Beq[t] = sum([gs[-1].apath[-(y+1)]*self.pop[t,-(y+1)]
                                    /self.sp[t,-(y+1)]*(1-self.sp[t,-(y+1)]) for y in range(T)])
                self.C[t] = sum([gs[-1].cpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
        self.Converged = (max(absolute(K1-self.K)) < self.tol*max(absolute(self.K)))
        """ Update the economy's aggregate K and N with weight phi on the old """
        self.K = self.phi*self.K + (1-self.phi)*K1
        self.L = self.phi*self.L + (1-self.phi)*L1
        self.k = self.K/self.L
        self.Dratio = -self.D/self.C
        self.Hd = H1
        self.Rd = R1
        self.Hd_Hs = self.Hd - self.Hs
        # print "K=%2.2f," %(self.K[0]),"L=%2.2f," %(self.L[0]),"K/L=%2.2f" %(self.k[0])


    def printprices(self):
        """ print market prices and others like tax """
        for i in range(self.TS/self.T):
            print "r=%2.2f," %(self.r[i*self.T]*100),"w=%2.2f," %(self.w[i*self.T]),\
                    "Tr=%2.2f" %(self.Tr[i*self.T]), "b=%2.2f," %(self.b[i*self.T]),\
                    "qh=%2.2f" %(self.qh[i*self.T]), "qr=%2.2f," %(self.qr[i*self.T])
            # print "K=%2.2f," %(self.K[i*self.T]),"L=%2.2f," %(self.L[i*self.T]),\
            #         "K/L=%2.2f" %(self.k[i*self.T]), "HD=%2.2f" %(self.Hd[i*self.T]),\
            #         "RD=%2.2f" %(self.Rd[i*self.T])


    def update(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,TS from last iteration """
        for t in range(self.TS):
            self.Pt[t] = sum(self.pop[t])
            self.Pr[t] = sum([self.pop[t,y] for y in range(self.W,self.T)])
            self.y[t] = self.k[t]**self.alpha
            self.r[t] = max(0.01,self.alpha*self.k[t]**(self.alpha-1)-self.delta)
            self.w[t] = (1-self.alpha)*self.k[t]**self.alpha
            self.Tax[t] = self.tw[t]*self.w[t]*self.L[t] + self.tr[t]*self.r[t]*self.k[t]*self.L[t]
            self.G[t] = self.gy[t]*self.y[t]*self.L[t]
            self.Tr[t] = (self.Tax[t]+self.Beq[t]-self.G[t])/self.Pt[t]
            self.tb[t] = self.zeta*(1-self.tw[t])*self.Pr[t]/(self.L[t]+self.zeta*self.Pr[t])
            self.b[t] = self.zeta*(1-self.tw[t]-self.tb[t])*self.w[t]
            #self.qh[t] = self.qh[t]*(1+0.1*(self.Hd[t]-self.Hs))
            #self.qr[t] = self.qr[t]*(1+0.1*self.Rd[t])
        # print "for r=%2.2f," %(self.r[0]*100), "w=%2.2f," %(self.w[0]), \
        #         "Tr=%2.2f," %(self.Tr[0]), "b=%2.2f," %(self.b[0]), "Beq.=%2.2f," %(self.Beq[0])
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, self.tb, self.Tr, self.qh, self.qr])


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
        self.hh = array([0.0, 1.0])
        # self.hh = loadtxt('hh.txt', delimiter='\n')
        self.sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.hN = hN = len(self.hh)
        """ age-specific productivity """
        self.ef = loadtxt('ef.txt', delimiter='\n')
        """ employment status """
        self.xx = [1, 0.4] #array([[0.4, 1] for y in range(T)], dtype=float)  # if unemployed, 40% of market wage is paid
        self.xN = xN = len(self.xx) #array([len(self.xx[y]) for y in range(T)], dtype=int)
        self.uu = 0.1
        self.ee = (23+self.uu)/24.0  # invariant distribution is (0.04 0.96)
        self.trx = array([[self.ee, 1-self.ee], [1-self.uu, self.uu]], dtype=float)
        """ value function and its interpolation """
        self.v = array([[[[0 for i in range(aN)] for h in range(hN)] for x in range(xN)] for y in range(T)], dtype=float)
        self.vtilde = [[[[] for h in range(hN)] for x in range(xN)] for y in range(T)]
        """ policy functions and interpolation for optimal next period asset """
        self.ao = array([[[[0 for i in range(aN)] for h in range(hN)] for x in range(xN)] for y in range(T)], dtype=float)
        self.atilde = [[[[] for h in range(hN)] for x in range(xN)] for y in range(T)]
        self.ho = array([[[[0 for i in range(aN)] for h in range(hN)] for x in range(xN)] for y in range(T)], dtype=int)
        self.co = array([[[[0 for i in range(aN)] for h in range(hN)] for x in range(xN)] for y in range(T)], dtype=float)
        self.ro = array([[[[0 for i in range(aN)] for h in range(hN)] for x in range(xN)] for y in range(T)], dtype=float)
        """ the following paths for a, c, n and u are used in direct and value function methods
        In direct method, those paths are directly calculated, while in the value function
        method the paths are calculated from value and policy functions """
        self.apath = array([a0 for y in range(T)], dtype=float)
        self.hpath = array([0 for y in range(T)], dtype=int)
        self.xpath = array([1 for y in range(T)], dtype=int)
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
                di, self.co[-1,0,h0,i], self.ro[-1,0,h0,i] = self.findcr(-1, 0, h0, 0, aa[i], 0, p)
                self.v[-1,0,h0,i] = self.util(self.co[-1,0,h0,i], self.ro[-1,0,h0,i]+hh[h0])
            self.vtilde[-1][0][h0] = interp1d(aa, self.v[-1,0,h0], kind='cubic')
            self.atilde[-1][0][h0] = interp1d(aa, linspace(0,0,aN), kind='cubic')
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
                income = Tr[y] + b[y] if y >= -self.R else Tr[y] + (1-tw[y]-tb[y])*w[y]*self.ef[y]*xx[x0]
                mdti = (aa + income*self.dti >= 0).argmax() # DTI constraint
                for h0 in range(hN):
                    at = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                    ct = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                    rt = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                    vt = array([[self.neg for i in range(aN)] for h in range(hN)], dtype=float)
                    # print self.apath
                    for h1 in range(hN):
                        mltv = (aa + self.ltv*qh[y]*hh[h1] >= 0).argmax() # LTV constraint
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
                    for i in range(self.aN):
                        h1 = vt[:,i].argmax()
                        self.ho[y,x0,h0,i] = h1
                        self.v[y,x0,h0,i] = vt[h1,i]
                        self.co[y,x0,h0,i] = ct[h1,i]
                        self.ro[y,x0,h0,i] = rt[h1,i]
                        self.ao[y,x0,h0,i] = at[h1,i]
                    self.vtilde[y][x0][h0] = interp1d(aa, self.v[y,x0,h0], kind='cubic')
                    self.atilde[y][x0][h0] = interp1d(aa, self.ao[y,x0,h0], kind='cubic')

                    ai = [self.aN/4, self.aN*2/4, self.aN*3/4, self.aN-1]
                    if y % 15 == 0 and h0 == 0 and x0 == 0:
                        print '------------'
                        print 'y=',y,'inc=%2.2f'%(income),'h0=',h0
                        for i in ai:
                            print 'ltv=%2.2f'%(hh[self.ho[y,x0,h0,i]]*qh[y]*self.ltv),'dti=%2.2f'%(income*self.dti),\
                                    'a0=%2.2f'%(aa[i]),'h1=%2.0f'%(self.ho[y,x0,h0,i]),\
                                    'a1=%2.2f'%(self.ao[y,x0,h0,i]),\
                                    'c0=%2.2f'%(self.co[y,x0,h0,i]),'r0=%2.2f'%(self.ro[y,x0,h0,i])


    def simulatelife(self, p, ainit=0, hinit=0, xinit=0):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle, 
        value and decision functions are calculated ***BACKWARD*** """
        T, hh = self.T, self.hh
        """ find asset and labor supply profiles over life-cycle from value function"""
        # generate each generation's asset, consumption and labor supply forward
        self.apath[0], self.hpath[0], self.xpath[0] = ainit, hinit, xinit
        for y in range(T-1):    # y = 0, 1,..., 58
            if y >= T-(self.R):
                nxx, trx = array([0]), array([[1]])
            elif y == T-(self.R+1):
                nxx, trx = array([0]), array([[1] for x0 in range(self.xN)])
            else:
                nxx, trx = self.xx, self.trx
            self.apath[y+1] = self.clip(self.atilde[y][self.xpath[y]][self.hpath[y]](self.apath[y]))
            
            v0 = self.neg
            for h1 in range(self.hN):
                di, c1, r1 = self.findcr(y-T, self.xpath[y], self.hpath[y], h1, self.apath[y], self.apath[y+1], p)
                if c1 > 0:
                    print nxx, trx, self.xpath[y], y, h1
                    ev = sum([self.vtilde[y+1][x1][h1](self.apath[y+1])*trx[self.xpath[y],x1] for x1 in range(len(nxx))])
                    v1 = self.util(c1, r1+hh[self.hpath[y]]) + self.beta*self.sp[y]*ev
                    if v1 > v0:
                        v0, self.cpath[y], self.rpath[y], self.hpath[y+1] = v1, c1, r1, h1
            self.upath[y] = self.util(self.cpath[y], self.rpath[y]+hh[self.hpath[y]])
            self.xpath[y+1] = self.nextx(y,self.xpath[y])
        # the oldest generation's consumption and labor supply
        di, self.cpath[T-1], self.rpath[T-1] = self.findcr(-1, 0, self.hpath[-1], 0, self.apath[-1], 0, p)
        self.upath[T-1] = self.util(self.cpath[T-1], self.rpath[T-1]+hh[self.hpath[T-1]])
        self.epath = self.ef[-self.T:]


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

        income = b[y] + Tr[y] if y >= -self.R else (1-tw[y]-tb[y])*w[y]*self.ef[y]*self.xx[x0] + Tr[y]

        di = a0*(1+(1-tr[y])*r[y]) + (self.hh[h0]-self.hh[h1])*qh[y] \
                - self.hh[h0]*qh[y]*(h0!=h1)*self.tcost - a1 + income
        co = (di + qr[y]*(self.hh[h0]+self.gamma))/(1+self.psi)
        ro = (di*self.psi-qr[y]*(self.hh[h0]+self.gamma))/((1+self.psi)*qr[y])
        return di, co, ro


    def clip(self, a):
        return self.aL if a <= self.aL else self.aH if a >= self.aH else a


    def util(self, c, s):
        # calculate utility value with given consumption and housing service
        return log(c) + self.psi*log(s+self.gamma) if c > 0 else self.neg


class agent:
    """ This class is just a "struct" to hold an agent's lifecycle behaviour """
    def __init__(self, g):
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


    def simulatelife(self, g, p, ainit=0, hinit=0, xinit=0):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle, 
        value and decision functions are calculated ***BACKWARD*** """
        T, hh = g.T, g.hh
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
            self.apath[y+1] = g.clip(g.atilde[y][self.xpath[y]][self.hpath[y]](self.apath[y]))
            
            v0 = g.neg
            for h1 in range(g.hN):
                di, c1, r1 = g.findcr(y-T, self.xpath[y], self.hpath[y], h1, self.apath[y], self.apath[y+1], p)
                if c1 > 0:
                    # print nxx, trx, self.xpath[y], y, h1
                    ev = sum([g.vtilde[y+1][x1][h1](self.apath[y+1])*trx[self.xpath[y],x1] for x1 in range(len(nxx))])
                    v1 = g.util(c1, r1+hh[self.hpath[y]]) + g.beta*g.sp[y]*ev
                    if v1 > v0:
                        v0, self.cpath[y], self.rpath[y], self.hpath[y+1] = v1, c1, r1, h1
            self.upath[y] = g.util(self.cpath[y], self.rpath[y]+hh[self.hpath[y]])
            self.xpath[y+1] = g.nextx(y,self.xpath[y])
        # the oldest generation's consumption and labor supply
        di, self.cpath[T-1], self.rpath[T-1] = g.findcr(-1, 0, self.hpath[-1], 0, self.apath[-1], 0, p)
        self.upath[T-1] = g.util(self.cpath[T-1], self.rpath[T-1]+hh[self.hpath[T-1]])
        self.epath = g.ef[-self.T:]



def spath(e, g):
    title = 'qh=' + str(e.qh[-1]) + 'qr=' + str(e.qr[-1]) \
                    + 'Hd=%2.2f'%(e.Hd[-1]) + 'Rd=%2.2f'%(e.Rd[-1]) + 'ltv=' \
                    + str(g.ltv) + 'dti=' + str(g.dti) + 'Debt=%2.2f'%(e.Dratio[-1]) + '.png'
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
    ax2.plot(g.hh[g.hpath])
    ax3.plot(g.cpath)
    ax4.plot(g.rpath)
    ax.set_xlabel('Age')
    ax.set_title(title, y=1.08)
    ax1.set_title('Liquid Asset')
    ax2.set_title('House')
    ax3.set_title('Consumption')
    ax4.set_title('Rent')
    path = 'D:\LTV\olg2'
    fullpath = os.path.join(path, title)
    fig.savefig(fullpath)
    plt.show()
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
    e = state(TG=1, k=4.2, ng=1, dng=0, W=45, R=30, Hs=60, qh=1.15, qr=0.0450)
    e.printprices()
    g = cohort(W=45, R=30, psi=0.1, beta=0.96, tcost=0.05, gamma=0.99, aL=-1.0, aH=2.0, aN=301, ltv=0.6, dti=2.0)
    g.valueNpolicy(e.p)
    print 'simulation starts...'
    aa = [agent(g) for i in range(100)]
    for a in aa:
        a.simulatelife(g, e.p, xinit=random.binomial(1,0.04))
    print e.sum(aa)
    # e.update()
    # e.printprices()
    # spath(e, g)

def gridsearch():
    qhN = qrN = 5
    qh = linspace(0.98,1.20,qhN)
    qr = linspace(0.039,0.042,qrN)
    House = array([[0 for i in range(qrN)] for j in range(qhN)])
    Rent = array([[0 for i in range(qrN)] for j in range(qhN)])
    Rate = array([[0 for i in range(qrN)] for j in range(qhN)])
    Dr = array([[0 for i in range(qrN)] for j in range(qhN)])
    for i in range(qhN):
        for j in range(qrN):
            House[i,j], Rent[i,j], Rate[i,j], Dr[i,j] = main1(psi=0.1, qh=qh[i], qr=qr[j], ltv=1.0, dti=1.2, tcost=0.05)
    with open('hrr10.pickle','wb') as f:
        pickle.dump([House, Rent, Rate, Dr], f)