# args[1] - number of nodes ('512','1024','2048')
# args[2] - network index
# args[3] - random seed
import sys, os
sys.path.insert(0, './PySources/')
sys.path.insert(0, './NetworkStates/')

#args = sys.argv

import numpy as np
import numpy.linalg as la
import numpy.random as rand
import pickle

import network_generation as ngen
import network_util as nutil
import network_plot as nplot

import numba
from numpy import zeros, ones, diag, array, where, dot, c_, r_, arange, sum, linspace
from numpy.random import randn
from scipy.linalg import solve, svd, norm
#from scipy.linalg import svd, norm
#from numpy.linalg import solve
from scipy.integrate import odeint, solve_ivp
#import scipy.sparse as sr

def LoadNetwork(NN, Index, RandomSeed):
    """
    NN - Number of nodes
    Index - Net number
    """
    # make network objects. net3 is the one usable for cpp lin-solver
    #NNodes = int(args[1])
    XFlag = False
    if NN == 64:
        NS = '00064'
        XFlag = True
    if NN == 128:
        NS = '00128'
        XFlag = True
    if NN == 256:
        NS = '00256'
        XFlag = True
    if NN == 512:
        NS = '00512'
        XFlag = True
    if NN == 1024:
        NS = '01024'
        XFlag = True
    if NN == 2048:
        NS = '02048'
        XFlag = True
    if NN == 4096:
        NS = '04096'
        XFlag = True
    assert(XFlag)
    NNet = Index

    if NN in [512, 1024, 2048, 4096]:
        fn = './NetworkStates/statedb_N' + NS + '_Lp-4.0000'
        #fn = '../../NetworkStates/statedb_N' + NS + '_Lp-4.0000'
        net = ngen.convert_jammed_state_to_network(fn,NNet) # choose number of nodes
        LL = net['box_mat'].diagonal()
        net['node_pos'] = net['node_pos']/LL[0] - 0.5
        net2 = ngen.convert_to_network_object(net)
    else:
        if NN == 128:
            NNet = NNet + 50
        if NN == 256:
            NNet = NNet + 100
        fn = './NetworkStates/data_N' + NS + '_Lp0.0100_r' + str(NNet) + '.txt'
        #fn = '../../NetworkStates/statedb_N' + NS + '_Lp-4.0000'
        f = open(fn)
        Data = f.readlines()
        Data = [d.rsplit() for d in Data]
        f.close()
        NN = int(Data[0][0])
        NE = int(Data[0][1])
        L = float(Data[0][2])
        pos = Data[1:NN+1]
        pos = np.asarray(pos, float).flatten()
        edges = np.asarray(Data[NN+1:],int)
        EI = edges.T[0]
        EJ = edges.T[1]

        net = {}
        net['DIM'] = 2
        net['box_L'] = np.array([L, L])
        net['NN'] = NN
        net['node_pos'] = pos
        net['NE'] = NE
        net['edgei'] = EI
        net['edgej'] = EJ
        net['box_mat'] = np.array([[L, 0.], [0., L]])
        LL = net['box_mat'].diagonal()
        net['node_pos'] = net['node_pos']/LL[0] - 0.5
        net2 = ngen.convert_to_network_object(net)
        
    NRandSeed = RandomSeed
    rand.seed(NRandSeed)
    return net, net2

#setup useful network objects
# DIM = 1
# NN = net2.NN
# NE = net2.NE
# EI = array(net2.edgei)
# EJ = array(net2.edgej)

# Conv = zeros([NE, NN])
# for i in range(NE):
#     Conv[i,EI[i]] = +1.
#     Conv[i,EJ[i]] = -1.

# # setup matrices
# D1 = array([where(EI==i,1,0) for i in range(NN)])
# D2 = array([where(EJ==j,1,0) for j in range(NN)])
# DD = D1 + D2

# ids = c_[EI,EJ]
# UT = zeros([NN,NN])
# G = zeros(NN); G[-1] = 1.



# flow solver

# Modified get Ps with ground constraint on final node
@numba.jit()
def GetPs(Data, Nodes, K, NN, NE, EI, EJ, DD, UT, G):
    D = diag(dot(DD, K))
    UT[EI,EJ] = -K
    LD = D + UT + UT.T

    cs = len(Nodes)
    id2 = arange(cs)

    S = zeros([NN, cs])
    S[Nodes, id2] = +1.

    LDB = zeros([NN+1+cs, NN+1+cs])
    LDB[:NN,:NN] = LD
    LDB[NN,:NN] = G
    LDB[:NN,NN] = G
    LDB[:NN,NN+1:] = S
    LDB[NN+1:,:NN] = S.T

    f = zeros(NN+1+cs)
    f[NN+1:] = Data

    P = solve(LDB, f, assume_a='sym', check_finite=False)[:NN]
    #P = solve(LDB, f)[:NN]
    return P

@numba.jit()
def EvalP(Data, Nodes, K, NN, NE, EI, EJ, DD, UT, G):
    PS = array([GetPs(d, Nodes, K, NN, NE, EI, EJ, DD, UT, G) for d in Data])
    return PS

@numba.jit()
def Eval(Data, Nodes, K, NN, NE, EI, EJ, DD, UT, G):
    PS = array([GetPs(d, Nodes, K, NN, NE, EI, EJ, DD, UT, G) for d in Data])
    return PS[:,EI] - PS[:,EJ]


# discrete time steps

def dIdt(P, K, DD, UT, G):
    D = diag(dot(DD, K))
    UT[EI,EJ] = -K
    LD = D + UT + UT.T
    DIDT = dot(LD, P)
    return DIDT

def step(P, Nodes, K, dt=1.e-3):
    DPDT = dIdt(P, K)
    dP = DPDT * dt
    dP[Nodes] = 0
    NNodes = list(set(arange(NN)) - set(Nodes))
    dP[NNodes] = dP[NNodes] - dP[NNodes].mean()
    #print(mean(dP))
    return dP

def EquilibrateP(P, Nodes, K, dt=2.e-1, steps=500):
    Pn = P.copy()
    for n in range(steps):
        dP = step(Pn, Nodes, K, dt)
        Pn = Pn - dP
    return Pn

# time evolution
from scipy.linalg import expm
def Advance(P, Nodes, K, t=0):
    D = diag(dot(DD, K))
    UT[EI,EJ] = -K
    LD = D + UT + UT.T
    LD[Nodes] = 0.
    NNodes = list(set(arange(NN)) - set(Nodes))
    LD[NNodes] = LD[NNodes] - mean(LD[NNodes],0)
    E = expm(- LD * t)
    PA = E.dot(P)
    return PA

# Cost functions

@numba.jit()
def CostSingleK(Data, SourceNodes, TargetNodes, K, Comp = []):
    P = EvalP(Data, SourceNodes, K)
    Y = P[:,TargetNodes]
    YL = Comp
    Cost = 0.5 * (Y - YL)**2
    return sum(Cost)

@numba.jit()
def CostSingleDerivative(Data, SourceNodes, TargetNodes, K, Comp = [], component=0, dk=1.e-3):
    DK1 = zeros(NE)
    DK1[component] = dk
    KN = K + DK1
    C1 = CostSingleK(Data, SourceNodes, TargetNodes, KN, Comp)
    if component%200==0:
        print(component)
    return C1

from scipy.integrate import odeint

@numba.jit()
def dydt(y,t,params):
    """
    y - vector containing NN node pressures then NE edge conductance values
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - constrained nodes indices
        params[3] - other nodes indices
        params[4] - DM - Delta matrix
        params[5] - eta - nudge amplitude
        params[6] - alpha - learning rate
        params[7] - State (0 = free, 1= clamped)
    """
    NN = params[0]
    NE = params[1]
    ConsNodes = params[2]
    OtherNodes = params[3]
    DM = params[4]
    eta = params[5]
    alpha = params[6]
    State = params[7]
    
    # pressure values at all nodes
    p = y[:NN]
    
    # k values for all edges
    k = y[NN:]
    
    # compute energy gradients
    DpE0 = dot(dot(DM.T * k, DM), p)
    
    # compute time derivatives for pressures
    DpDt = zeros(NN)
    DpDt[ConsNodes] = 0.
    DpDt[OtherNodes] = -DpE0[OtherNodes]
    DpDt[-1] = 0.
    #DpDt[OtherNodes] = DpDt[OtherNodes] - DpDt[OtherNodes].mean()
    
    
    # compute time derivatives for conductances
    #DkDt = (alpha / eta**2) * (1. - 2.*State) * DkE0
    #DkDt = DkDt * (k > 1.e-2)
    DkDt = zeros(NE)
    
    # merge for final time derivative vector
    dydt = r_[DpDt, DkDt]
    return dydt

@numba.jit()
def dPKdt(y,t,params):
    """
    Compute the change of both physical and learning degrees of freedom
    y - vector containing NN node pressures then NE edge conductance values
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - Desired values
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comp = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comp
    
    # k values for all edges
    k = y[2*NN:]
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes]
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    DpcDt[STNodes] = 0.
    DpcDt[NSTNodes] = -DpcE0[NSTNodes]
    DpcDt[-1] = 0.
    
    # compute time derivatives for conductances
    ppf = (pf[EI] - pf[EJ])**2.
    ppc = (pc[EI] - pc[EJ])**2.
    DkDt = (alpha / eta) * (ppf - ppc) 
    DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, DkDt]
    return dpkdt

@numba.jit()
def dPdtSwitch(t,y,params):
    """
    Compute the change in only the free state physical degrees of freedom at fixed learning degrees of freedom.
    y - vector containing NN node pressures then NE edge conductance values
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - Desired values
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
        params[13] - tau - physics rate
        params[14] - C0 - initial cost value (for event function)
        params[15] - Thresh - Learning success threshold (for event function)
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comp = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    tau = params[13]
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comp
    
    # k values for all edges
    k = y[2*NN:]
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes] * tau
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    DpcDt[STNodes] = 0.
    DpcDt[NSTNodes] = -DpcE0[NSTNodes] * tau
    DpcDt[-1] = 0.
    
    ## compute time derivatives for conductances
    #ppf = (pf[EI] - pf[EJ])**2.
    #ppc = (pc[EI] - pc[EJ])**2.
    #DkDt = (alpha / eta) * (ppf - ppc) 
    #DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, zeros(NE)]
    return dpkdt

@numba.jit()
def dPKdtSwitch(t,y,params):
    """
    Compute the change of both physical and learning degrees of freedom
    y - vector containing NN node pressures then NE edge conductance values
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - Desired values
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
        params[13] - tau - physics rate
        params[14] - C0 - initial cost value (for event function)
        params[15] - Thresh - Learning success threshold (for event function)
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comp = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    tau = params[13]
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comp
    
    # k values for all edges
    k = y[2*NN:]
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes] * tau
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    DpcDt[STNodes] = 0.
    DpcDt[NSTNodes] = -DpcE0[NSTNodes] * tau
    DpcDt[-1] = 0.
    
    # compute time derivatives for conductances
    ppf = (pf[EI] - pf[EJ])**2.
    ppc = (pc[EI] - pc[EJ])**2.
    DkDt = (alpha / eta) * (ppf - ppc) 
    DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, DkDt]
    return dpkdt

@numba.jit()
def dPKdtSwitch2(t,y,params):
    """
    Compute the change of both physical and learning degrees of freedom
    This time try to change the clamped targets during the simulation.
    y - vector containing NN node pressures then NE edge conductance values
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - Desired values
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
        params[13] - tau - physics rate
        params[14] - C0 - initial cost value (for event function)
        params[15] - Thresh - Learning success threshold (for event function)
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comp = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    tau = params[13]
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pct0 = pc[TargetNodes]
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comp
    
    # k values for all edges
    k = y[2*NN:]
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes] * tau
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    #DpcDt[STNodes] = 0.
    DpcDt[SourceNodes] = 0.
    DpcDt[TargetNodes] = pc[TargetNodes] - pct0
    DpcDt[NSTNodes] = -DpcE0[NSTNodes] * tau
    DpcDt[-1] = 0.
    
    # compute time derivatives for conductances
    ppf = (pf[EI] - pf[EJ])**2.
    ppc = (pc[EI] - pc[EJ])**2.
    DkDt = (alpha / eta) * (ppf - ppc) 
    DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, DkDt]
    return dpkdt


@numba.jit()
def dPKdtSwitchRegression(t,y,params):
    """
    Compute the change of both physical and learning degrees of freedom
    y - vector containing NN node pressures then NE edge conductance values
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - All output values (all training examples)
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
        params[13] - tau - physics rate
        params[14] - C0 - initial cost value (for event function)
        params[15] - Thresh - Learning success threshold (for event function)
        params[16] - All input values (all training examples)
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comps = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    tau = params[13]
    Datas = params[16]
    
    # first find out which example index (ei) to show the network
    # every alpha seconds the example changes to the next one
    lD = len(Datas)
    tda = t/(lD*alpha)
    ei = int(tda) % lD # index of current example
    en = (ei + 1) % lD # index of next example
    frac = tda - int(tda)      # how close we are to the next example
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    pf[SourceNodes] = Datas[ei]    # 100% current example
    #pf[SourceNodes] = (1 - frac) * Datas[ei] + frac * Datas[en]      # interplolate inputs between training examples
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pc[SourceNodes] = Datas[ei]    # 100% current example
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comps[ei]    # 100% current example
    #pc[SourceNodes] = (1 - frac) * Datas[ei] + frac * Datas[en]      # interplolate inputs between training examples
    #pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * ((1 - frac) * Comps[ei] + frac * Comps[en])     # interplolate outputs between training examples as well

    # k values for all edges
    k = y[2*NN:]
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes] * tau
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    DpcDt[STNodes] = 0.
    DpcDt[NSTNodes] = -DpcE0[NSTNodes] * tau
    DpcDt[-1] = 0.
    
    # compute time derivatives for conductances
    ppf = (pf[EI] - pf[EJ])**2.
    ppc = (pc[EI] - pc[EJ])**2.
    DkDt = (alpha / eta) * (ppf - ppc) 
    DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, DkDt]
    return dpkdt


@numba.jit()
def dPKdtSwitchRamp(t,y,params):
    """
    Same as before but with variable tau/alpha as a function of time
    Compute the change of both physical and learning degrees of freedom
    y - vector containing NN node pressures then NE edge conductance values
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - Desired values
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
        params[13] - tau - physics rate
        params[14] - C0 - initial cost value (for event function)
        params[15] - Thresh - Learning success threshold (for event function)
        params[16] - Parameters of power ramp [tauI, tauF, tI, tF, taupower]
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comp = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    #tau = params[13]
    tauI = params[16][0]
    tauF = params[16][1]
    tI = params[16][2]
    tF = params[16][3]
    taupower = params[16][4]
    tau = alpha * ((tauF - tauI) * (t/tF)**taupower + tauI)
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comp
    
    # k values for all edges
    k = y[2*NN:]
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes] * tau
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    DpcDt[STNodes] = 0.
    DpcDt[NSTNodes] = -DpcE0[NSTNodes] * tau
    DpcDt[-1] = 0.
    
    # compute time derivatives for conductances
    ppf = (pf[EI] - pf[EJ])**2.
    ppc = (pc[EI] - pc[EJ])**2.
    DkDt = (alpha / eta) * (ppf - ppc) 
    DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, DkDt]
    return dpkdt


@numba.jit()
def dPKdtSwitchAdaptive(t,y,params):
    """
    Same as before but adaptive gamma/alpha ratio set by whether the non-eq error grows or not.
    Compute the change of both physical and learning degrees of freedom
    y - vector containing NN node pressures then NE edge conductance values + 1 value for NEq cost + 1 for alpha
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - Desired values
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
        params[13] - tau - physics rate
        params[14] - C0 - initial cost value (for event function)
        params[15] - Thresh - Learning success threshold (for event function)
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comp = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    tau = params[13]
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comp
    
    # k values for all edges
    k = y[2*NN:-2]
    
    # Non-Eq cost value
    CNEq = y[-2]
    
    # Learning rate alpha
    Alpha = y[-1]
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes] * tau
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    DpcDt[STNodes] = 0.
    DpcDt[NSTNodes] = -DpcE0[NSTNodes] * tau
    DpcDt[-1] = 0.
    
    # compute time derivatives for conductances
    ppf = (pf[EI] - pf[EJ])**2.
    ppc = (pc[EI] - pc[EJ])**2.
    DkDt = (Alpha / eta) * (ppf - ppc) 
    DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # compute derivative for NEq cost function
    DNCDt = sum((pf[TargetNodes] - Comp) * DpfDt[TargetNodes])
    #if DNCDt > 0.: # reduce learning rate alpha=param[12]
    #    params[12] = params[12] * 0.99
    
    # adapt learning rate
    DAlpha = 0.
    if DNCDt > 0.:
        #DAlpha = -DNCDt
        #DAlpha = -DNCDt*100.
        DAlpha = - 0.001 * Alpha
    if Alpha + DAlpha < 1.e-9:
        DAlpha = -Alpha + 1.e-9
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, DkDt, DNCDt, DAlpha]
    return dpkdt


@numba.jit()
def dPKdtSwitchAdaGrad(t,y,params):
    """
    Same as before but adaptive gamma/alpha ratio set by AdaGrad.
    Compute the change of both physical and learning degrees of freedom
    y - vector containing NN node pressures then NE edge conductance values + NE values for 'alpha_j'
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - Desired values
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
        params[13] - tau - physics rate
        params[14] - C0 - initial cost value (for event function)
        params[15] - Thresh - Learning success threshold (for event function)
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comp = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    tau = params[13]
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comp
    
    # k values for all edges
    k = y[2*NN:2*NN+NE]
    
    # AdaGrad sums
    rj = y[2*NN+NE:]
    
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes] * tau
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    DpcDt[STNodes] = 0.
    DpcDt[NSTNodes] = -DpcE0[NSTNodes] * tau
    DpcDt[-1] = 0.
    
    # compute time derivatives for conductances
    ppf = (pf[EI] - pf[EJ])**2.
    ppc = (pc[EI] - pc[EJ])**2.
    LR = (ppf - ppc) / eta
    DkDt = alpha * LR / (rj + 1.e-8)**0.5
    DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # Update the AdaGrad sums
    DrjDt = LR**2.
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, DkDt, DrjDt]
    return dpkdt


@numba.jit()
def dPKdtSwitchAdaGrad1(t,y,params):
    """
    Same as before but adaptive gamma/alpha ratio set by AdaGrad that averages all edge changes.
    Compute the change of both physical and learning degrees of freedom
    y - vector containing NN node pressures then NE edge conductance values + 1 values for mean alpha change
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - EI - left edges
        params[3] - EJ - right edges
        params[4] - SourceNodes
        params[5] - TargetNodes
        params[6] - SourceNodes and TargetNodes
        params[7] - All but SourceNodes
        params[8] - All but SourceNodes & TargetNodes
        params[9] - Desired values
        params[10] - DM - Delta matrix
        params[11] - eta - nudge amplitude
        params[12] - alpha - learning rate
        params[13] - tau - physics rate
        params[14] - C0 - initial cost value (for event function)
        params[15] - Thresh - Learning success threshold (for event function)
    """
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    STNodes = params[6]
    NSNodes = params[7]
    NSTNodes = params[8]
    Comp = params[9]
    DM = params[10]
    eta = params[11]
    alpha = params[12]
    tau = params[13]
    
    # "free" pressure values at all nodes
    pf = y[:NN]
    
    # "clamped" pressure values at all nodes
    pc = y[NN:2*NN]
    pc[TargetNodes] = (1. - eta) * pf[TargetNodes] + eta * Comp
    
    # k values for all edges
    k = y[2*NN:2*NN+NE]
    
    # AdaGrad sums
    rj = y[-1]
    
    
    # compute energy gradients
    DpfE0 = dot(dot(DM.T * k, DM), pf)
    DpcE0 = dot(dot(DM.T * k, DM), pc)
    
    # compute time derivatives for free pressures
    DpfDt = zeros(NN)
    DpfDt[SourceNodes] = 0.
    DpfDt[NSNodes] = -DpfE0[NSNodes] * tau
    DpfDt[-1] = 0.
    
    # compute time derivatives for clamped pressures
    DpcDt = zeros(NN)
    DpcDt[STNodes] = 0.
    DpcDt[NSTNodes] = -DpcE0[NSTNodes] * tau
    DpcDt[-1] = 0.
    
    # compute time derivatives for conductances
    ppf = (pf[EI] - pf[EJ])**2.
    ppc = (pc[EI] - pc[EJ])**2.
    LR = (ppf - ppc) / eta
    DkDt = alpha * LR / (rj + 1.e-8)**0.5
    DkDt = where(k + DkDt > 1.e-6, DkDt, -k + 1.e-6)
    
    # Update the AdaGrad sum
    DrjDt = sum(LR**2.)/NE
    
    # merge for final time derivative vector
    dpkdt = r_[DpfDt, DpcDt, DkDt, DrjDt]
    return dpkdt


@numba.jit()
def LearningComplete(t, y, params):
    NN = params[0]
    TargetNodes = params[5]
    Comp = params[9]
    C0 = params[14]
    Thresh = params[15]
    
    pf = y[:NN]
    CC = sum((pf[TargetNodes] - Comp)**2.)
    return CC/C0 - Thresh
LearningComplete.terminal = True 


@numba.jit()
def LearningCompleteEq(t, y, params):
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    Comp = params[9]
    C0 = params[14]
    Thresh = params[15]
    DD = params[16]
    UT = params[17]
    G = params[18]    
    
    Data = y[SourceNodes]
    K = y[2*NN:]
    pf = GetPs(Data, SourceNodes, K, NN, NE, EI, EJ, DD, UT, G)
    CC = sum((pf[TargetNodes] - Comp)**2.)
    return CC/C0 - Thresh
LearningCompleteEq.terminal = True

@numba.jit()
def LearningCompleteEq2(t, y, params):
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    Comp = params[9]
    eta = params[11]
    C0 = params[14]
    Thresh = params[15]
    DD = params[16]
    UT = params[17]
    G = params[18]    
    
    Data = y[SourceNodes]
    K = y[2*NN:]
    pf = GetPs(Data, SourceNodes, K, NN, NE, EI, EJ, DD, UT, G)
    Nudge = pf[TargetNodes] * (1 - eta) + Comp * eta
    pc = GetPs(r_[Data, Nudge], r_[SourceNodes, TargetNodes], K, NN, NE, EI, EJ, DD, UT, G)
    
    dpf = pf[EI] - pf[EJ]
    dpc = pc[EI] - pc[EJ]
    ppf = K * dpf**2
    ppc = K * dpc**2
    
    CC = sum(ppc - ppf)/eta
    return CC/C0 - Thresh
LearningCompleteEq2.terminal = True 


@numba.jit()
def LearningCompleteEqReg(t, y, params):
    NN = params[0]
    NE = params[1]
    EI = params[2]
    EJ = params[3]
    SourceNodes = params[4]
    TargetNodes = params[5]
    Comps = params[9]
    C0 = params[14]
    Thresh = params[15]
    Datas = params[16]
    DD = params[17]
    UT = params[18]
    G = params[19]    
    
    K = y[2*NN:]
    pfs = EvalP(Datas, SourceNodes, K, NN, NE, EI, EJ, DD, UT, G)
    CC = sum((pfs[:,TargetNodes] - Comps)**2.) / len(Datas)
    return CC/C0 - Thresh
LearningCompleteEqReg.terminal = True 



@numba.jit()
def OneIteration(y0, SourceNodes, TargetNodes, Comp, T2, params): # one cycle of clamped then free state
    # y0 is the result of the free state in the previous iteration - starting point for this iteration
    
    NN = params[0][0]
    NE = params[0][1]
    ConsNodes = params[0][2]
    OtherNodes = params[0][3]
    DM = params[0][4]
    eta = params[0][5]
    alpha = params[0][6]
    State = params[0][7]
    
    EI = params[0][8] 
    EJ = params[0][9]
    DD = params[0][10]
    UT = params[0][11]
    G = params[0][12]
    
    PF0 = y0[:NN]
    K0 = y0[NN:]
        
    # Clamped State
    ConsNodes = r_[SourceNodes, TargetNodes]
    OtherNodes = list(set(arange(NN)) - set(ConsNodes))
    Params = ([NN, NE, ConsNodes, OtherNodes, DM, eta, alpha, 0],)
    
    PC0 = PF0.copy()
    Nudge = eta * (Comp - PF0[TargetNodes])
    PC0[TargetNodes] = PF0[TargetNodes] + Nudge
    #PC0[OtherNodes] = PF0[OtherNodes] - sum(Nudge)/len(OtherNodes)
    ys = r_[PC0, K0]
    t = linspace(0,T2,100)
    sol = odeint(dydt, ys, t, args=Params)
    YC = sol[-1]
    
    EPC = GetPs(r_[PC0[SourceNodes], PC0[TargetNodes]], ConsNodes, K0, NN, NE, EI, EJ, DD, UT, G)
    #EquibDistance = norm(YF[:NN]-PF0)/norm(YC[:NN]-PF0)
    #norm((sol[:,:NN]-PC),axis=1)/norm(sol[0,:NN]-PC)
    EquibDistance = 1. - norm(YC[:NN]-EPC)/norm(PC0-EPC)
    
    # Clamped learning rule
    PC = YC[:NN]
    DPC = PC[EI] - PC[EJ]
    PPC = DPC**2.
    #PPC = (EPC[EI] - EPC[EJ])**2
    #YC[NN:] = YC[NN:] #- alpha / eta**2. * PPC
    
    # Free State
    ConsNodes = SourceNodes
    OtherNodes = list(set(arange(NN)) - set(ConsNodes))
    Params = ([NN, NE, ConsNodes, OtherNodes, DM, eta, alpha, 0],)
    PF1 = YC[:NN].copy()
    K1 = YC[NN:].copy()
    ys = r_[PF1, K1]
    sol = odeint(dydt, ys, t, args=Params)
    YF = sol[-1]
    
    EPF = GetPs(PC0[SourceNodes], SourceNodes, K0, NN, NE, EI, EJ, DD, UT, G)
    #print(YF[:NN]-EPF)
    EquibDistance2 = 1. - norm(YF[:NN]-EPF)/norm(YC[:NN]-EPF)
        
    # Free learning rule
    PF = YF[:NN]
    DPF = PF[EI] - PF[EJ]
    PPF = DPF**2.
    #PPF = (EPF[EI] - EPF[EJ])**2
    YF[NN:] = YF[NN:] + alpha / eta * (PPF - PPC)
    YF[NN:] = YF[NN:].clip(1.e-4,1.e6)
    
    DK1 = (PPF - PPC)/norm(PPF - PPC)
    EDPF = EPF[EI] - EPF[EJ]
    EPPF = EDPF**2.
    EDPC = EPC[EI] - EPC[EJ]
    EPPC = EDPC**2.
    DK2 = (EPPF - EPPC)/norm(EPPF - EPPC)
    Dot = dot(DK1,DK2)
    
    ConsNodes = SourceNodes
    OtherNodes = list(set(arange(NN)) - set(ConsNodes))
    Params = ([NN, NE, ConsNodes, OtherNodes, DM, eta, alpha, 0],)
    ys = YF.copy()
    sol = odeint(dydt, ys, t, args=Params)
    #EPFN = GetPs(PF0[SourceNodes], SourceNodes, YF[NN:])
    YFN = sol[-1]
    
    return YF, YC, EquibDistance, EquibDistance2, YFN, Dot # return the end state vector after each state (free, clamped)


@numba.jit()
def dydt2(y,t,params):
    """
    y - vector containing NN node pressures then NE edge conductance values
    t - time
    params:
        params[0] - NN - number of nodes
        params[1] - NE - number of edges
        params[2] - constrained nodes indices
        params[3] - other nodes indices
        params[4] - DM - Delta matrix
        params[5] - DM.T
        params[6] - k - Conductances
    """
    NN = params[0]
    #NE = params[1]
    #ConsNodes = params[2]
    OtherNodes = params[3]
    DM = params[4]
    DMT = params[5]
    k = params[6]
    
    # pressure values at all nodes
    p = y

    # compute energy gradients
    #DpE0 = dot(dot(dot(DMT, diag(k)), DM), p)
    DpE0 = dot(dot(DMT * k, DM), p)
    #DkE0 = 0.5 * dot(DM, p)**2.
    
    # compute time derivatives for pressures
    DpDt = zeros(NN)
    #DpDt[ConsNodes] = 0.
    DpDt[OtherNodes] = -DpE0[OtherNodes]
    DpDt[-1] = 0.
    return DpDt


def TRT(t, RampParams):
    ### Real time during training
    tauI = RampParams[0]
    tauF = RampParams[1]
    tI = RampParams[2]
    tF = RampParams[3]
    taupower = RampParams[4]
    
    tRT1 = tauI * t[t < tI]
    tRT2 = tauI * tI + (tauF - tauI) * (t[(t >= tI) * (t < tF)]/tF)**(taupower+1.) / (taupower+1.) * tF + tauI * (t[(t >= tI) * (t < tF)] - tI)
    tRT3 = tauI * tI + (tauF - tauI) / (taupower+1.) * tF + tauI * (tF - tI) + tauF * (t[t >= tF] - tF)
    tRT = r_[tRT1, tRT2, tRT3]
    return tRT