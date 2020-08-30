#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


# In[4]:


# Fixed parameters
file = open("T0n05Vn15sid.csv","w+")
K = 0.0
U = -6.0
V = -1.5
Temp = 0.001  # Temperature needs to be set up here
nval = 0.5
# Fixed functions
def eps(kx,ky):
    f = -2.0*(np.cos(kx) + np.cos(ky))
    return f

def sk(kx, ky):
    f = 1/2*(np.cos(kx) + np.cos(ky))
    return f

def dk(kx, ky):
    f = 1/2*(np.cos(kx) - np.cos(ky))
    return f

def deltaSing(kx,ky,d0,ds,dd):
    sing = (d0 + ds*sk(kx,ky) + (1.0j)*dd*dk(kx,ky))
    return sing

def energy(mu,kx,ky,d0,ds,dd):
    f = np.sqrt((eps(kx, ky) - mu)**2 + np.abs(deltaSing(kx,ky,d0,ds,dd))**2)
    return f

def fermi(T,mu,kx,ky,d0,ds,dd):
    f = 1.0/(np.exp(energy(mu,kx,ky,d0,ds,dd)/T) + 1.0)
    return f

def g(T,mu,kx,ky,d0,ds,dd):
    f = (1.0 - 2.0*fermi(T,mu,kx,ky,d0,ds,dd))/(2.0*energy(mu,kx,ky,d0,ds,dd)+0.0000001)
    return f


# In[ ]:


while U<6.0:
    print('nval =', nval,' Temp =', Temp, ' U = ',U)
    L = 12
    NSites = L**2
    delK = 2.0*np.pi/np.sqrt(NSites)
    kvals = [-np.pi + delK + i*delK for i in range(L)]
    
    def n(mu,T,d0,ds,dd):
        sump = 2.0/(NSites)*sum((eps(kvals[i],kvals[j])-mu)*g(T,mu,kvals[i],kvals[j],d0,ds,dd) for i in range(len(kvals)) for j in range(len(kvals)))
        f = 1.0-sump
        return f

    sktab = np.zeros((len(kvals),len(kvals)))
    dktab = np.zeros((len(kvals),len(kvals)))
    for i in range(len(kvals)):
        for j in range(int(len(kvals))):
            sktab[i,j] = sk(kvals[i], kvals[j])
            dktab[i,j] = dk(kvals[i], kvals[j])
            
    def numeqn(mu,d0,ds,dd):
        return nval - n(mu,Temp,d0,ds,dd)
    
    def FreeE(p):
        Del0, Dels, Deld = p 
    
        mu = optimize.root(numeqn, -1.5, args=(Del0,Dels,Deld), method='broyden1', options={'ftol':0.03}).x
    
        gp = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        lnp = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        epsEp = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        delSingp = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        del2p = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        i00 = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        iss = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        idd = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        for i in range(len(kvals)):
            for j in range(len(kvals)):
                gp[i,j] = g(Temp, mu, kvals[i], kvals[j], Del0, Dels, Deld)
                lnp[i,j] = np.log(1.0-fermi(Temp, mu, kvals[i], kvals[j], Del0, Dels, Deld))
                epsEp[i,j] = eps(kvals[i],kvals[j])-mu-energy(mu,kvals[i],kvals[j],Del0,Dels,Deld)
                delSingp[i,j] = deltaSing(kvals[i],kvals[j],Del0,Dels,Deld)
                del2p[i,j] = (np.abs(delSingp[i,j])**2)*gp[i,j]
                i00[i,j] = delSingp[i,j]*gp[i,j]
                iss[i,j] = sktab[i,j]*i00[i,j]
                idd[i,j] = dktab[i,j]*i00[i,j]
            
        I00 = np.abs(1.0/(NSites)*np.sum(i00))**2
        Iss = np.abs(1.0/(NSites)*np.sum(iss))**2
        Idd = np.abs(1.0/(NSites)*np.sum(idd))**2

        logterm = 2.0*Temp/NSites*(np.sum(lnp))
        enerterm = 1.0/(NSites)*(np.sum(epsEp))+2.0/(NSites)*(np.sum(del2p))
        Vterm = U*I00 + 4.0*V*Iss +4.0*V*Idd
    
        return np.real(enerterm + logterm + mu*nval + Vterm)
                
    def FreeEs(p):
        Del0, Dels = p 
        Deld = 0.0
        return FreeE((Del0,Dels,Deld))
    
    def FreeEd(p):
        Deld = p 
        Del0 = 0.0
        Dels = 0.0
        return FreeE((Del0,Dels,Deld))
    
    freeEval = 10.0;
    for i in range(24):
        r1 = 0.0
        r2 = 0.0
        r3 = np.random.uniform(0.11,3.0)
        f = FreeE((r1,r2,r3))
        if f<freeEval:
            freeEval = f
            globmin = np.array([r3])
            print(globmin,f)
            
    #globmind=optimize.minimize(FreeEd, globmin, callback=None, options={'gtol': 2e-3, 'disp': True}).x 
    globmind1=np.array([0.04,0.04,globmin[0]])
    globmind1=optimize.minimize(FreeE, globmind1, callback=None, options={'gtol': 2e-3, 'disp': True}).x
    freeEdval1=FreeE(globmind1)
    
    freeEval = 10.0;
    for i in range(78):
        r1 = np.random.uniform(0.05,2.0)
        r2 = np.random.uniform(-2.0,2.0)
        r3 = 0.0
        f = FreeE((r1,r2,r3))
        if f<freeEval:
            freeEval = f
            globmin = np.array([r1,r2])
            print(globmin,f)
    
    #globmins=optimize.minimize(FreeEs, globmin, callback=None, options={'gtol': 2e-3, 'disp': True}).x
    globmins1=np.array([globmin[0],globmin[1],0.04])
    globmins1=optimize.minimize(FreeE, globmins1, callback=None, options={'gtol': 2e-3, 'disp': True}).x
    freeEsval1=FreeE(globmins1)
    
    
    freelist=np.array([freeEdval1,freeEsval1])
    globminlist=np.array(np.array([globmind1,globmins1]))
    result = np.where(freelist == np.amin(freelist))
    globmin = globminlist[result[0]][0]
    print(globmin, " ", np.amin(freelist))
    
    # Fixed parameters for larger lattice
    L = 34
    NSites = L**2
    delK = 2.0*np.pi/np.sqrt(NSites)
    kvals = [-np.pi + delK + i*delK for i in range(L)]
    
    def n(mu,T,d0,ds,dd):
        sump = 2.0/(NSites)*sum((eps(kvals[i],kvals[j])-mu)*g(T,mu,kvals[i],kvals[j],d0,ds,dd) for i in range(len(kvals)) for j in range(len(kvals)))
        f = 1.0-sump
        return f

    sktab = np.zeros((len(kvals),len(kvals)))
    for i in range(len(kvals)):
        for j in range(int(len(kvals))):
            sktab[i,j] = sk(kvals[i], kvals[j])
        
    dktab = np.zeros((len(kvals),len(kvals)))
    for i in range(len(kvals)):
        for j in range(int(len(kvals))):
            dktab[i,j] = dk(kvals[i], kvals[j])
        
    def FreeE(p):
        Del0, Dels, Deld = p 
    
        mu = optimize.root(numeqn, -1.5, args=(Del0,Dels,Deld), method='broyden1', options={'ftol':0.008}).x
    
        gp = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        lnp = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        epsEp = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        delSingp = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        del2p = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        i00 = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        iss = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        idd = np.zeros((len(kvals),len(kvals)),dtype=np.complex_)
        for i in range(len(kvals)):
            for j in range(len(kvals)):
                gp[i,j] = g(Temp, mu, kvals[i], kvals[j], Del0, Dels, Deld)
                lnp[i,j] = np.log(1.0-fermi(Temp, mu, kvals[i], kvals[j], Del0, Dels, Deld))
                epsEp[i,j] = eps(kvals[i],kvals[j])-mu-energy(mu,kvals[i],kvals[j],Del0,Dels,Deld)
                delSingp[i,j] = deltaSing(kvals[i],kvals[j],Del0,Dels,Deld)
                del2p[i,j] = (np.abs(delSingp[i,j])**2)*gp[i,j]
                i00[i,j] = delSingp[i,j]*gp[i,j]
                iss[i,j] = sktab[i,j]*i00[i,j]
                idd[i,j] = dktab[i,j]*i00[i,j]
            
        I00 = np.abs(1.0/(NSites)*np.sum(i00))**2
        Iss = np.abs(1.0/(NSites)*np.sum(iss))**2
        Idd = np.abs(1.0/(NSites)*np.sum(idd))**2

        logterm = 2.0*Temp/NSites*(np.sum(lnp))
        enerterm = 1.0/(NSites)*(np.sum(epsEp))+2.0/(NSites)*(np.sum(del2p))
        Vterm = U*I00 + 4.0*V*Iss +4.0*V*Idd
    
        return np.real(enerterm + logterm + mu*nval + Vterm)
    
    globmin=optimize.minimize(FreeE, globmin, callback=None, options={'gtol': 5e-6, 'disp': True}).x
    F=FreeE(globmin)
    
    data = (U, V, globmin[0],globmin[1],globmin[2], F)
    print(data)
    file.write("%s,%s,%s,%s,%s,%s\n" % data)
    U = U + 0.05
    
file.close()


# In[ ]:




