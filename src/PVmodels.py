from sympy import exp
import numpy as np
import tensorflow as tf
from scipy.constants import Boltzmann as k, eV as q, zero_Celsius as T0
from .special import *
import decimal
decimal.getcontext().prec = 20


class PVModel: 
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref):
    self.b_ref = b_ref
    self.IL_ref = IL_ref
    self.I0_ref = I0_ref
    self.Rs_ref = Rs_ref
    self.Gp_ref = Gp_ref
    self.T_ref = T_ref 
    self.S_ref = S_ref
    self.EgV   = 1.121
    
  def update_params(self, S, T):
    self.b, self.IL, self.I0, self.Rs, self.Gp = self.params(S,T)

  def Eg(self, T):
    return self.EgV*(1-0.0002677*(T-self.T_ref))  
  
  def auxiliar(self, T):
    return q/k *(self.Eg(self.T_ref)/self.T_ref-self.Eg(T)/T)  
        
  def fun_foc(self, Vmp):
    return Vmp/(1+self.Rs*self.Gp)*(self.Gp+self.W0(Vmp)/self.Rs/(1+self.W0(Vmp)))    
  
  def fun_Vpv(self, Ipv):
    return (self.IL+self.I0-(1+self.Gp*self.Rs)*Ipv)/(self.Gp)-lambertw(self.b*self.I0/(self.Gp)*tf.exp(self.b*(self.IL+self.I0-Ipv)/(self.Gp)))/(self.b)

  def fun_Ipv(self, Vpv):
    return (self.IL+self.I0-self.Gp*Vpv)/(self.Rs*self.Gp+1)-self.W0(Vpv)/(self.b*self.Rs)
  
  def W0(self, Vpv):
    return lambertw(self.b*self.Rs*self.I0/(self.Rs*self.Gp+1)*tf.exp(self.b*(self.Rs*(self.IL+self.I0)+Vpv)/(self.Rs*self.Gp+1)))

  def funVoc(self, Ipv, step_tol=1e-8):
    """alternative method for estimating Voc"""
    # Voc = (IL+I0-(1+Gp*Rs)*Ipv)/Gp-lambertw(b*I0/Gp*exp(b*(IL+I0-Ipv)/Gp))/b
    type_ = self.b.numpy().dtype
    b  = self.b.numpy().astype(float)
    IL = self.IL.numpy().astype(float)
    I0 = self.I0.numpy().astype(float)
    Rs = self.Rs.numpy().astype(float)
    Gp = self.Gp.numpy().astype(float)
    
    K1 = (b*I0/(Gp))
    K2 = (b*(IL+I0-Ipv)/(Gp))
    w_list = []
    try:
      for n in range(len(K1)):
        z = decimal.Decimal(K1[n][0])*np.exp(decimal.Decimal(K2[n][0]))
        w = np.asarray([(z+1).ln()]).reshape(-1,1)
        z = np.asarray([z]).reshape(-1,1)
        step = w
        while np.max(np.abs(step)) > step_tol:
          ew = np.exp(w)
          numer = (w*ew - z)
          step = numer/(ew*(w+1) - (w+2)*numer/(2*w + 2))
          w = w - step
        w_list.append(w.astype(np.longdouble))
      w_list = np.asanyarray(w_list).reshape(-1,1)
      return ((IL+I0-(1+Gp*Rs)*Ipv)/(Gp)-w_list/(b)).astype(type_)
    except:
      # resultado de la exponencial, elemento a elemento
      z = decimal.Decimal(K1)*np.exp(decimal.Decimal(K2))
      # valor auxiliar para la función de lambertw
      w = np.asarray([(z+1).ln()]).reshape(-1,1)
      z = np.asarray([z]).reshape(-1,1)
      step = w
      # Estimación del valor asociado a la función de lambertw
      while np.max(np.abs(step)) > step_tol:
        ew = np.exp(w)
        numer = (w*ew - z)
        step = numer/(ew*(w+1) - (w+2)*numer/(2*w + 2))
        w = w - step

      w_list = np.asarray(w.astype(np.longdouble)).reshape([-1,1])
      # Estimación de Voc
      return ((IL+I0-(1+Gp*Rs)*Ipv)/(Gp)-w_list/(b)).astype(type_)

  def predict(self, S, T, MaxIterations=10000, tol=1e-9, alpha=0.05, beta=0.95, VmpIni=None): 
    self.update_params(S, T)
    Isc = self.fun_Ipv(0)
    Vsc = self.fun_Vpv(Isc)
    Voc = self.fun_Vpv(0)
    if tf.reduce_any(tf.math.is_nan(Voc)):
      Voc = self.funVoc(0)
    Ioc = self.fun_Ipv(Voc)   
    fun_Vmp = lambda Vmp: self.fun_Ipv(Vmp)-self.fun_foc(Vmp)
    IniIterations = 0
    if VmpIni==None: 
      Vmp0, Vmp1 = Voc*alpha, Voc*beta
    else: 
      Vmp0, Vmp1 = VmpIni
    error = tf.math.abs(Vmp1-Vmp0)
    while tf.reduce_all(tf.math.less_equal(np.float64(tol), error)):
      IniIterations+=1
      Vmp11 = Vmp0-fun_Vmp(Vmp0)*(Vmp1-Vmp0)/(fun_Vmp(Vmp1)-fun_Vmp(Vmp0))
      Vmp0, Vmp1 = Vmp1, Vmp11
      error = tf.math.abs(Vmp1-Vmp0)
      if IniIterations==MaxIterations:
        break
    Vmp = Vmp1
    Imp = self.fun_Ipv(Vmp)
    Pmp = Vmp*Imp
    return [Isc, Vsc, Imp, Vmp, Pmp, Ioc, Voc]
  



  
class Model5PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, alphaT, T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref,Gp_ref, T_ref, S_ref)
    self.alphaT = alphaT
      
  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(T),
      self.Rsfun(S),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    return self.b_ref*(self.T_ref/T)
  
  def ILfun(self, S, T):
    return (S/self.S_ref)*(self.IL_ref+self.alphaT*(T-self.T_ref))
  
  def I0fun(self, T):
    return self.I0_ref*(T/self.T_ref)**3*np.exp(self.auxiliar(T))

  def Rsfun(self, S):
    return self.Rs_ref*(S/S) # to vector

  def Gpfun(self, S):
    return self.Gp_ref*(S/self.S_ref)
  
class ModelA5PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, alphaT, Adjust, T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref)
    self.alphaT = alphaT
    self.Adjust = Adjust

  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(T),
      self.Rsfun(S),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    return self.b_ref*(self.T_ref/T)
  
  def ILfun(self, S, T):
    return (S/self.S_ref)*(self.IL_ref+self.alphaT*(1-self.Adjust/100)*(T-self.T_ref))
  
  def I0fun(self, T):
    return self.I0_ref*(T/self.T_ref)**3*np.exp(self.auxiliar(T))

  def Rsfun(self, S):
    return self.Rs_ref*(S/S) # to vector

  def Gpfun(self, S):
    return self.Gp_ref*(S/self.S_ref)
  
class Model6PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, alphaT, mIL, T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref)
    self.alphaT = alphaT
    self.mIL = mIL
      
  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(T),
      self.Rsfun(S),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    return self.b_ref*(self.T_ref/T)
  
  def ILfun(self, S, T):
    return (S/self.S_ref)**self.mIL*(self.IL_ref+self.alphaT*(T-self.T_ref))
  
  def I0fun(self, T):
    return self.I0_ref*(T/self.T_ref)**3*np.exp(self.auxiliar(T))

  def Rsfun(self, S):
    return self.Rs_ref*(S/S) # to vector

  def Gpfun(self, S):
    return self.Gp_ref*(S/self.S_ref)
  
class Model7PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, alphaT, mI0, deltaRs, T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref)
    self.alphaT = alphaT
    self.mI0 = mI0
    self.deltaRs = deltaRs
      
  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(S, T),
      self.Rsfun(T),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    return self.b_ref*(self.T_ref/T)
  
  def ILfun(self, S, T):
    return (S/self.S_ref)*(self.IL_ref+self.alphaT*(T-self.T_ref))
  
  def I0fun(self, S, T):
    return self.I0_ref*(S/self.S_ref)**self.mI0*(T/self.T_ref)**3*np.exp(self.auxiliar(T)).astype(np.longdouble)

  def Rsfun(self, T):
    return self.Rs_ref*np.exp(self.deltaRs*(T-self.T_ref)).astype(np.longdouble)

  def Gpfun(self, S):
    return self.Gp_ref*(S/self.S_ref)
  
class Model11PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref,
                     mI0, mRs, mGp,
                     alphaT, deltaI0, deltaRs,
                     T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref)
    self.mI0 = mI0
    self.mRs = mRs
    self.mGp = mGp
    self.alphaT = alphaT
    self.deltaI0 = deltaI0
    self.deltaRs = deltaRs
    
  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(S, T),
      self.Rsfun(S, T),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    return self.b_ref*(self.T_ref/T)
  
  def ILfun(self, S, T):
    return (S/self.S_ref)*(self.IL_ref+self.alphaT*(T-self.T_ref))
  
  def I0fun(self, S, T):
    try: 
      if self.I0_ref.is_Symbol:
        return self.I0_ref*(S/self.S_ref)**self.mI0*(T/self.T_ref)**3*exp(self.deltaI0*self.auxiliar(T))
      else:
        return self.I0_ref*(S/self.S_ref)**self.mI0*(T/self.T_ref)**3*np.exp(self.deltaI0*self.auxiliar(T))
    except:
      return self.I0_ref*(S/self.S_ref)**self.mI0*(T/self.T_ref)**3*np.exp(self.deltaI0*self.auxiliar(T))

  def Rsfun(self, S, T):
    try: 
      if self.Rs_ref.is_Symbol:
        return self.Rs_ref*(S/self.S_ref)**self.mRs*exp(self.deltaRs*(T-self.T_ref))
      else:
        return self.Rs_ref*(S/self.S_ref)**self.mRs*np.exp(self.deltaRs*(T-self.T_ref))
    except:
      return self.Rs_ref*(S/self.S_ref)**self.mRs*np.exp(self.deltaRs*(T-self.T_ref))

  def Gpfun(self, S):
    return self.Gp_ref*(S/self.S_ref)**self.mGp