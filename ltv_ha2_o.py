# -*- coding: utf-8 -*-
"""
Jan. 12, 2015, Hyun Chang Yi
Computes the model of Section 9.3. in Heer/Maussner using 
direct method from Secion 9.1.

HOUSEHOLD'S UTILITY FUNCTION IS DIFFERENT FROM THAT OF SECTION 9.1. AND 9.2.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod, log, arange, set_printoptions, isnan
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle

from multiprocessing import Process, Lock, Manager


class state:
    """ This class is just a "struct" to hold  the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, alpha=0.3, delta=0.08, phi=0.8, tol=0.01, Hs = 10,
        tr = 0.01, tw = 0.21, zeta=0.35, gy = 0.195, qh = 10, qr = 0.3,
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
        self.Hd = array([self.Hs for t in range(TS)], dtype=float)
        self.Hd_Hs = array([0 for t in range(TS)], dtype=float)
        self.Rd = array([0 for t in range(TS)], dtype=float)
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
                self.Beq[t] = sum([gs[t+y].apath[-(y+1)]*self.pop[t,-(y+1)]
                                    /self.sp[t,-(y+1)]*(1-self.sp[t,-(y+1)]) for y in range(T)])
                self.C[t] = sum([gs[t+y].cpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
            else:
                K1[t] = sum([gs[-1].apath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                L1[t] = sum([gs[-1].epath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                H1[t] = sum([gs[-1].hpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                R1[t] = sum([gs[-1].rpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                self.Beq[t] = sum([gs[-1].apath[-(y+1)]*self.pop[t,-(y+1)]
                                    /self.sp[t,-(y+1)]*(1-self.sp[t,-(y+1)]) for y in range(T)])
                self.C[t] = sum([gs[-1].cpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
        self.Converged = (max(absolute(K1-self.K)) < self.tol*max(absolute(self.K)))
        """ Update the economy's aggregate K and N with weight phi on the old """
        self.K = self.phi*self.K + (1-self.phi)*K1
        self.L = self.phi*self.L + (1-self.phi)*L1
        self.k = self.K/self.L
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
            print "K=%2.2f," %(self.K[i*self.T]),"L=%2.2f," %(self.L[i*self.T]),\
                    "K/L=%2.2f" %(self.k[i*self.T]), "HD-HS =%2.2f" %(self.Hd_Hs[i*self.T]),\
                    "RD=%2.2f" %(self.Rd[i*self.T])


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
    def __init__(self, beta=0.99, sigma=2.0, gamma=1, aH=5.0, aL=0.0, y=-1, dti = 0.5,
        aN=101, psi=0.03, tol=0.01, neg=-1e10, W=45, R=30, a0 = 0, tcost = 0.0, ltv=0.7):
        self.beta, self.sigma, self.gamma, self.psi = beta, sigma, gamma, psi
        self.R, self.W, self.y = R, W, y
        self.tcost, self.ltv, self.dti = tcost, ltv, dti
        self.T = T = (y+1 if (y >= 0) and (y <= W+R-2) else W+R)
        self.aH, self.aL, self.aN, self.aa = aH, aL, aN, aL+(aH-aL)*linspace(0,1,aN)
        self.tol, self.neg = tol, neg
        """ house sizes and number of feasible feasible house sizes """
        self.hh = [0.0, 0.2, 0.5, 0.7, 1.0]
        # self.hh = loadtxt('hh.txt', delimiter='\n')
        self.sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.hN = len(self.hh)
        """ age-specific productivity """
        self.ef = loadtxt('ef.txt', delimiter='\n')
        """ value function and its interpolation """
        self.v = array([[[0 for i in range(aN)] for h in range(self.hN)] for y in range(T)], dtype=float)
        self.vtilde = [[[] for h in range(self.hN)] for y in range(T)]
        """ policy functions used in value function method """
        self.na = [[[0 for i in range(2)] for h in range(self.hN)] for y in range(T)]
        self.ao = array([[[0 for i in range(aN)] for h in range(self.hN)] for y in range(T)], dtype=float)
        self.ho = array([[[0 for i in range(aN)] for h in range(self.hN)] for y in range(T)], dtype=float)
        self.co = array([[[0 for i in range(aN)] for h in range(self.hN)] for y in range(T)], dtype=float)
        self.ro = array([[[0 for i in range(aN)] for h in range(self.hN)] for y in range(T)], dtype=float)
        """ the following paths for a, c, n and u are used in direct and value function methods
        In direct method, those paths are directly calculated, while in the value function
        method the paths are calculated from value and policy functions """
        self.apath = array([a0 for y in range(T)], dtype=float)
        self.hpath = array([0 for y in range(T)], dtype=float)
        self.cpath = array([0 for y in range(T)], dtype=float)
        self.rpath = array([0 for y in range(T)], dtype=float)
        self.spath = array([0 for y in range(T)], dtype=float)
        self.epath = array([0 for y in range(T)], dtype=float) # labor supply in efficiency unit
        self.upath = array([0 for y in range(T)], dtype=float)


    def findvpath(self, p):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle, 
        value and decision functions are calculated ***BACKWARD*** """
        [r, w, b, tr, tw, tb, Tr, qh, qr] = p
        T, aa, hh, aN, hN = self.T, self.aa, self.hh, self.aN, self.hN
        psi, tcost, beta, gamma, sp = self.psi, self.tcost, self.beta, self.gamma, self.sp
        # y = -1 : the oldest generation
        for h in range(self.hN):
            for i in range(self.aN):
                budget = aa[i]*(1+(1-tr[-1])*r[-1]) + hh[h]*qh[-1]*(1-tcost) + b[-1] + Tr[-1]
                self.co[-1,h,i] = (budget+qr[-1]*(hh[h]+gamma))/(1+psi)
                self.ro[-1,h,i] = (budget*psi-qr[-1]*(hh[h]+gamma))/((1+psi)*qr[-1])
                self.v[-1,h,i] = self.util(self.co[-1,h,i], self.ro[-1,h,i]+hh[h])
                # print 'h=',h,'a=',aa[i],'budget=%2.2f'%(budget), 'c=%2.2f'%(self.co[-1,h,i]), \
                # 'r=%2.2f'%(self.ro[-1,h,i]),'v=%2.2f'%(self.v[-1,h,i])
            self.vtilde[-1][h] = interp1d(aa, self.v[-1,h], kind='cubic')

        # y = -2, -3,..., -60
        for y in range(-2, -(T+1), -1):
            income = Tr[y] + b[y] if y >= -self.R else Tr[y] + (1-tw[y]-tb[y])*w[y]*self.ef[y]
            mdti = (aa + income*self.dti >= 0).argmax() # DTI constraint
            # print 'dit',income*0.3,mdti,aa[mdti]
            for h0 in range(self.hN):
                at = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                ct = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                rt = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                vt = array([[float('-inf') for i in range(aN)] for h in range(hN)], dtype=float)
                casht = array([[0 for i in range(aN)] for h in range(hN)], dtype=float)
                # print self.apath
                for h1 in range(self.hN):
                    mltv = (aa + self.ltv*qh[y]*hh[h1] >= 0).argmax() # LTV constraint
                    # print (aa + self.ltv*qh[y]*hh[h1] > 0)
                    # print m0
                    m0 = max(0,mltv,mdti)
                    # m0 = 0
                    for i in range(self.aN):    # l = 0, 1, ..., 50
                        # Find a bracket within which optimal a' lies
                        m = max(0, m0)  # Rch91v.g uses m = max(0, m0-1)
                        # print m
                        m0, a0, b0, c0 = self.GetBracket(y, h0, h1, i, m, p)
                        # a0, b0, c0 = self.GetBracket2(y, h0, h1, i, p)
                        # print y, h0, h1, i, m0, a0, b0, c0
                        # Find optimal a' using Golden Section Search
                        # print 'a=',self.aa[i],'bracket=','(',a0,',',c0,')'
                        if a0 == b0:
                            at[h1,i] = aa[m0]
                        elif b0 == c0:
                            at[h1,i] = self.aH
                        else:
                            # print a0, b0, c0
                            def objfn(a1): # Define objective function for optimal a'
                                return -self.findv(y, h0, h1, aa[i], a1, p)
                            result = minimize_scalar(objfn, bracket=(a0,b0,c0), method='Golden')
                            at[h1,i] = result.x
                        # print y, h0, h1, i, m0, a0, b0, c0, 'a1=',at[h1,i]
                        # Compute consumption, rent and house
                        casht[h1,i] = self.budget(y,h0,h1,aa[i],at[h1,i],p)
                        ct[h1,i] = (casht[h1,i]+qr[y]*(hh[h0]+gamma))/(1+psi)
                        rt[h1,i] = (casht[h1,i]*psi-qr[y]*(hh[h0]+gamma))/((1+psi)*qr[y])
                        vt[h1,i] = self.util(ct[h1,i],rt[h1,i]+hh[h0]) + beta*sp[y]*self.vtilde[y+1][h1](at[h1,i])
                for i in range(self.aN):
                    h1 = vt[:,i].argmax()
                    # print 'y,h0,h1',y, h0,i, 'h1',h1,'v=', vt[0,i], vt[1,i], vt[2,i], vt[3,i]
                    self.v[y,h0,i] = vt[h1,i]
                    self.co[y,h0,i] = ct[h1,i]
                    self.ro[y,h0,i] = rt[h1,i]
                    self.ho[y,h0,i] = h1
                    self.ao[y,h0,i] = at[h1,i]
                    # print at[h1,i]
                    cash = casht[h1,i]
                ai = [self.aN/4, self.aN*2/4, self.aN*3/4, self.aN-1]
                # set_printoptions(precision=2)
                if y % 15 == 0:
                    print '------------'
                    print 'y=',y,'inc=%2.2f'%(income),'h0=',h0
                    for i in ai:
                        print 'ltv=%2.2f'%(hh[int(self.ho[y,h0,i])]*qh[y]*self.ltv),'dti=%2.2f'%(income*self.dti),\
                                'a0=%2.2f'%(aa[i]),'h1=%2.0f'%(self.ho[y,h0,i]),\
                                'a1=%2.2f'%(self.ao[y,h0,i]),\
                                'c0=%2.2f'%(self.co[y,h0,i]),'r0=%2.2f'%(self.ro[y,h0,i])
                              # 'budget=%2.2f' %(cash),'c=%2.2f' %(self.co[y,h0,i]),'r=%2.2f' %(self.ro[y,h0,i])
                        # print '------','v1(0,a1)=%2.2f' %(beta*self.vtilde[y+1][0](self.ao[y,h,i])),\
                        #       'v1(1,a1)=%2.2f' %(beta*self.vtilde[y+1][1](self.ao[y,h,i])),\
                        #       'v1(2,a1)=%2.2f' %(beta*self.vtilde[y+1][2](self.ao[y,h,i]))

                self.vtilde[y][h0] = interp1d(aa, self.v[y,h0], kind='cubic')
            # if (y == -50):
            #     break
        """ find asset and labor supply profiles over life-cycle from value function"""
        # generate each generation's asset, consumption and labor supply forward
        for y in range(T-1):    # y = 0, 1,..., 58
            self.apath[y+1] = self.clip(interp1d(aa, self.ao[y,self.hpath[y]], kind='cubic')(self.apath[y]))
            v0 = self.neg
            for h1 in range(self.hN):
                if y >= T-self.R:    # y = -2, -3, ..., -60
                    budget = self.apath[y]*(1+(1-tr[y])*r[y]) + (self.hpath[y]-hh[h1])*qh[y] \
                                - self.hpath[y]*qh[y]*(self.hpath[y]!=hh[h1])*tcost \
                                + b[y] + Tr[y] - self.apath[y+1]
                else:
                    budget = self.apath[y]*(1+(1-tr[y])*r[y]) + (self.hpath[y]-hh[h1])*qh[y] \
                                - self.hpath[y]*qh[y]*(self.hpath[y]!=hh[h1])*tcost \
                                + Tr[y] - self.apath[y+1] + (1-tw[y]-tb[y])*w[y]*self.ef[y]
                c1 = (budget+qr[y]*(self.hpath[y]+gamma))/(1+psi)
                r1 = (budget*psi-qr[y]*(self.hpath[y]+gamma))/((1+psi)*qr[y])
                if c1 <= 0:
                    pass
                else:
                    v1 = self.util(c1, r1+self.hpath[y]) + beta*sp[y]*self.vtilde[y+1][h1](self.apath[y+1])
                    if v1 >= v0:
                        v0, self.cpath[y], self.rpath[y], self.hpath[y+1], self.spath[y] = v1, c1, r1, hh[h1], r1+self.hpath[y]
            self.upath[y] = self.util(self.cpath[y], self.spath[y])
        # the oldest generation's consumption and labor supply
        budget = (1+(1-tr[T-1])*r[T-1])*self.apath[T-1] + b[T-1] + Tr[T-1] + self.hpath[T-1]*qh[T-1]*(1-self.tcost)
        self.cpath[T-1] = (budget+qr[T-1]*(self.hpath[T-1]+gamma))/(1+psi)
        self.rpath[T-1] = (budget*psi-qr[T-1]*(self.hpath[T-1]+gamma))/((1+psi)*qh[T-1])
        self.spath[T-1] = self.rpath[T-1]
        self.upath[T-1] = self.util(self.cpath[T-1], self.rpath[T-1]+self.hpath[T-1])
        self.epath = self.ef[-self.T:]


    def GetBracket(self, y, h0, h1, l, m, p):
        """ Find a bracket (a,b,c) such that policy function for next period asset level, 
        a[x;asset[l],y] lies in the interval (a,b) """
        aa = self.aa
        # print aa
        a, b, c = aa[0], 2*aa[0]-aa[1], 2*aa[0]-aa[2]
        # print 'abc=',a,b,c
        minit = m
        m0 = m
        v0 = float('-inf')
        while (a > b) or (b > c):
            v1 = self.findv(y, h0, h1, aa[l], aa[m], p)
            # print y,l,m,'v0=%2.2f'%(v0),'v1=%2.2f'%(v1),a,b,c
            # print v1
            if v1 > v0:
                a, b, = ([aa[m], aa[m]] if m == minit else [aa[m-1], aa[m]])
                v0, m0 = v1, m
            else:
                c = aa[m]
            if m == self.aN - 1:
                a, b, c = aa[m-1], aa[m], aa[m]
            m = m + 1
        return m0, a, b, c


    def findv(self, y, h0, h1, a0, a1, p):
        """ Return the value at the given generation and asset a0 and 
        corresponding consumption and labor supply when the agent chooses his 
        next period asset a1, current period consumption c and labor n
        a1 is always within aL and aH """
        [r, w, b, tr, tw, tb, Tr, qh, qr] = p
        T, aa, hh, aN, hN = self.T, self.aa, self.hh, self.aN, self.hN
        psi, tcost, beta, gamma, sp = self.psi, self.tcost, self.beta, self.gamma, self.sp

        cash = self.budget(y, h0, h1, a0, a1, p)
        if cash + qr[y]*(hh[h0]+gamma) > 0:
            co = (cash+qr[y]*(hh[h0]+gamma))/(1+psi)
            ro = (cash*psi-qr[y]*(hh[h0]+gamma))/((1+psi)*qr[y])
            v = self.util(co, ro+hh[h0]) + beta*sp[y]*self.vtilde[y+1][h1](a1)
        else:
            v = self.neg #float('-inf')
        return v


    def budget(self, y, h0, h1, a0, a1, p):
        """ budget for consumption and rent given next period house and asset """
        [r, w, b, tr, tw, tb, Tr, qh, qr] = p
        hh = self.hh
        if y >= -self.R:    # y = -2, -3, ..., -60
            # print y, h0, h1, tr[y], r[y], hh[h0], hh[h1], qh[y], b[y], Tr[y]
            b = a0*(1+(1-tr[y])*r[y]) + (hh[h0]-hh[h1])*qh[y] - hh[h0]*qh[y]*(h0!=h1)*self.tcost + b[y] + Tr[y] - a1
        else:
            b = a0*(1+(1-tr[y])*r[y]) + (hh[h0]-hh[h1])*qh[y] - hh[h0]*qh[y]*(h0!=h1)*self.tcost \
                    + Tr[y] + (1-tw[y]-tb[y])*w[y]*self.ef[y] - a1
        return b


    def clip(self, a):
        return self.aL if a <= self.aL else self.aH if a >= self.aH else a


    def util(self, c, s):
        # calculate utility value with given consumption and housing service
        # print 'c=',c,'s',s
        return log(c) + self.psi*log(s+self.gamma) if c > 0 else self.neg #float('-inf')
        # (((c+self.psi)**self.gamma*(1-l)**(1-self.gamma))**(1-self.sigma))/(1-self.sigma)


def spath(e, g):
    title = 'psi=' + str(g.psi) + 'qh=' + str(e.qh[-1]) + 'qr=' + str(e.qr[-1]) \
                    + 'Hd-Hs=%2.2f'%(e.Hd_Hs[-1]) + 'Rd=%2.2f'%(e.Rd[-1]) + 'ltv=' \
                    + str(g.ltv) + 'dti=' + str(g.dti) + '.png'
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
    ax2.plot(g.hpath)
    ax3.plot(g.cpath)
    ax4.plot(g.rpath)
    ax.set_xlabel('Age')
    ax.set_title(title, y=1.08)
    ax1.set_title('Liquid Asset')
    ax2.set_title('House')
    ax3.set_title('Consumption')
    ax4.set_title('Rent')
    plt.show()
    fig.savefig(title)
    # time.sleep(1)
    # plt.close() # plt.close("all")
    

def main1(psi=0.1, qh=1.3, qr=0.05, ltv=0.6, dti=1.2, tcost=0.04):
    e = state(TG=1, ng=1, dng=0, W=45, R=30, Hs=10, qh=qh, qr=qr)
    e.printprices()
    g = cohort(W=45, R=30, psi=psi, beta=0.96, tcost=tcost, gamma=1, aL=-1.0, aH=1.0, aN=501, ltv=ltv, dti=dti)
    g.findvpath(e.p)
    e.aggregate([g])
    e.update()
    e.printprices()
    spath(e, g)
    return e.Hd_Hs[-1], e.Rd[-1], e.r[-1]

# if __name__ == '__main__':

def gridsearch():    
    qhN = qrN = 5
    qh = linspace(1,2,qhN)
    qr = linspace(0.02,0.08,qrN)
    House = array([[0 for i in range(qrN)] for j in range(qhN)])
    Rent = array([[0 for i in range(qrN)] for j in range(qhN)])
    Rate = array([[0 for i in range(qrN)] for j in range(qhN)])
    for i in range(qhN):
        for j in range(qrN):
            House[i,j], Rent[i,j], Rate[i,j] = main1(psi=0.1, qh=qh[i], qr=qr[j], ltv=0.8, dti=2.0, tcost=0.05)
    with open('hrr.pickle','wb') as f:
        pickle.dump([House, Rent, Rate], f)