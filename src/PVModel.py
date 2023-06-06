import numpy as np
from special import *
from scipy.constants import Boltzmann as k, eV as q, zero_Celsius as T0

class PVModelClass: 
  def __init__(self, Params):
    self.EgV = 1.121
    self.Tr=25+T0
    self.Sr=1000
    self.T0= T0
    self.q = q
    self.k = k
    self.Params = Params
    self.Eg = lambda T: self.EgV*(1-0.0002677*(T-self.Tr))  

  def __call__(self, x, PVModule, model):
    if model==5:
      return self.DeSoto(x, self.Params[PVModule][model])
    elif model==6:
      return self.Dobos(x, self.Params[PVModule][model])
    elif model==7:
      return self.Boyd(x, self.Params[PVModule][model])
    elif model=='mod':
      return self.BoydMod(x, self.Params[PVModule][7])

  def DeSoto(self, x, Params):
    """DeSoto Model: 5-parameter model {Rs, Gp, IL, I0, b}, alphaIsc"""
    Rs, Gp, IL, I0, b, alphaIsc = Params["Rs"], Params["Gp"], Params["IL"], Params["I0"], Params["b"], Params["alphaIsc"]    
    bF  = lambda x: b*(self.Tr/(x[:,1]+self.T0))
    ILF = lambda x: (x[:,0]/self.Sr)*(IL+alphaIsc*(x[:,1]+self.T0-self.Tr))
    I0F = lambda x: I0*((x[:,1]+self.T0)/self.Tr)**3*np.exp(self.q/self.k*(self.Eg(self.Tr)/self.Tr-self.Eg(x[:,1]+self.T0)/(x[:,1]+self.T0)).astype(np.longdouble))
    RsF = lambda x: Rs*np.ones(x[:,0].shape)
    GpF = lambda x: Gp*(x[:,0]/self.Sr)
    return [np.array(k).astype(np.float32).reshape(-1,1) for k in [RsF(x), GpF(x), ILF(x), I0F(x), bF(x)]]

  def Dobos(self, x, Params):
    """Dobos Model: 6-parameter model {Rs, Gp, IL, I0, b, Adjust}, alphaIsc"""
    Rs, Gp, IL, I0, b, alphaIsc, Adjust = Params["Rs"], Params["Gp"], Params["IL"], Params["I0"], Params["b"], Params["alphaIsc"], Params["Adjust"]
    bF  = lambda x: b*(self.Tr/(x[:,1]+self.T0))
    ILF = lambda x: (x[:,0]/self.Sr)*(IL+(alphaIsc*(1-Adjust/100))*(x[:,1]+self.T0-self.Tr))
    I0F = lambda x: I0*((x[:,1]+self.T0)/self.Tr)**3*np.exp(self.q/self.k*(self.Eg(self.Tr)/self.Tr-self.Eg(x[:,1]+self.T0)/(x[:,1]+self.T0)).astype(np.longdouble))
    RsF = lambda x: Rs*np.ones(x[:,0].shape)
    GpF = lambda x: Gp*(x[:,0]/self.Sr)
    return [np.array(k).astype(np.float32).reshape(-1,1) for k in [RsF(x), GpF(x), ILF(x), I0F(x), bF(x)]]

  def Boyd(self, x, Params):
    """Boyd Model: 7-parameter model {Rs, Gp, IL, I0, b, mIL, deltaRs} alphaIsc"""
    Rs, Gp, IL, I0, b, alphaIsc = Params["Rs"], Params["Gp"], Params["IL"], Params["I0"], Params["b"], Params["alphaIsc"]
    deltaRs, mI0 = Params["deltaRs"], Params["mI0"]
    # Original
    bF  = lambda x: b*(self.Tr/x[:,1])
    ILF = lambda x: (x[:,0]/self.Sr)*(IL+alphaIsc*(x[:,1]-self.Tr))
    I0F = lambda x: I0*(self.Sr/x[:,0])**mI0*(x[:,1]/self.Tr)**3*np.exp(self.q/self.k*(self.Eg(self.Tr)/self.Tr-self.Eg(x[:,1]+self.T0)/(x[:,1]+self.T0)).astype(np.longdouble))
    RsF = lambda x: Rs*(x[:,0]/self.Sr)*np.exp(deltaRs*(x[:,1]-self.Tr).astype(np.longdouble))
    GpF = lambda x: Gp*(x[:,0]/self.Sr)
    return [np.array(k).astype(np.float32).reshape(-1,1) for k in [RsF(x), GpF(x), ILF(x), I0F(x), bF(x)]]

  def BoydMod(self, x, Params):
    Rs, Gp, IL, I0, b, alphaIsc = Params["Rs"], Params["Gp"], Params["IL"], Params["I0"], Params["b"], Params["alphaIsc"]
    deltaRs, mI0 = Params["deltaRs"], Params["mI0"]
    # Original
    # bF  = lambda x: b*(self.Tr/x[:,1])
    # ILF = lambda x: (x[:,0]/self.Sr)*(IL+alphaIsc*(x[:,1]-self.Tr))
    # I0F = lambda x: I0*(self.Sr/x[:,0])**mI0*(x[:,1]/self.Tr)**3*np.exp(self.q/self.k*(self.Eg(self.Tr)/self.Tr-self.Eg(x[:,1]+self.T0)/(x[:,1]+self.T0)).astype(np.longdouble))
    # RsF = lambda x: Rs*(x[:,0]/self.Sr)*np.exp(deltaRs*(x[:,1]-self.Tr).astype(np.longdouble))
    # GpF = lambda x: Gp*(x[:,0]/self.Sr)
    # Matlab Profe
    mRs, mGp, mIL, lambda1 = Params["mRs"], Params["mGp"], Params["mIL"], Params["lambda"]
    RsF = lambda x: Rs*(x[:,0]/self.Sr)**mRs*np.exp(deltaRs*(x[:,1]+self.T0-self.Tr).astype(np.longdouble))
    GpF = lambda x: Gp*(x[:,0]/self.Sr)**mGp
    ILF = lambda x: (x[:,0]/self.Sr)**mIL*(IL+lambda1*(x[:,1]+self.T0-self.Tr))
    I0F = lambda x: I0*(self.Sr/x[:,0])**mI0*((x[:,1]+self.T0)/self.Tr)**3*np.exp(self.q/self.k*(self.Eg(self.Tr)/self.Tr-self.Eg(x[:,1]+self.T0)/(x[:,1]+self.T0)).astype(np.longdouble))
    bF  = lambda x: b*(self.Tr/(x[:,1]+self.T0))
    return [np.array(k).astype(np.float32).reshape(-1,1) for k in [RsF(x), GpF(x), ILF(x), I0F(x), bF(x)]]


