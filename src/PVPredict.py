import numpy as np
import tensorflow as tf
from special import *
import decimal
decimal.getcontext().prec = 20

class PVPredictClass:
  def __init__(self):  
    eps = 1e-8 # Estabilidad en la división por Gp 
    self.W0 = lambda Rs, Gp, IL, I0, b, Vpv: \
        lambertw( b*Rs*I0/(Rs*Gp+1)*tf.exp(b*(Rs*(IL+I0)+Vpv)/(Rs*Gp+1)))
    self.fun_Ipv = lambda Rs, Gp, IL, I0, b, Vpv: \
        (IL+I0-Gp*Vpv)/(Rs*Gp+1)-self.W0(Rs, Gp, IL, I0, b, Vpv)/(b*Rs)
    self.fun_Vpv = lambda Rs, Gp, IL, I0, b, Ipv: \
        (IL+I0-(1+Gp*Rs)*Ipv)/(Gp+eps)-lambertw(b*I0/(Gp+eps)*tf.exp(b*(IL+I0-Ipv)/(Gp+eps)))/(b+eps)
    self.fun_foc = lambda Rs, Gp, IL, I0, b, Vmp: \
        Vmp/(1+Rs*Gp)*(Gp+self.W0(Rs, Gp, IL, I0, b, Vmp)/Rs/(1+self.W0(Rs, Gp, IL, I0, b, Vmp)))    

  def funVoc(self, Rs, Gp, IL, I0, b, Ipv, step_tol=1e-8):
    eps = 1e-8
    # Voc = (IL+I0-(1+Gp*Rs)*Ipv)/Gp-lambertw(b*I0/Gp*exp(b*(IL+I0-Ipv)/Gp))/b
    # K1 y K2 son los elementos dentro de la exponencial
    K1 = (b*I0/(Gp+eps)).astype(float)
    K2 = (b*(IL+I0-Ipv)/(Gp+eps)).astype(float)
    # se amplia la cantidad de decimales a emplear
    data = []
    for n in range(len(K1)):
      # resultado de la exponencial, elemento a elemento
      z = decimal.Decimal(K1[n][0])*np.exp(decimal.Decimal(K2[n][0]))
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
      data.append(w.astype(np.longdouble))
    # Estimación de Voc
    return ((IL+I0-(1+Gp*Rs)*Ipv)/(Gp+eps)-np.asarray(data).reshape(-1,1)/(b+eps)).astype(np.longdouble)
  

  def predict(self, Rs, Gp, IL, I0, b, MaxIterations=200, tol=1e-8, alpha=0.1, beta=0.8, VmpIni=None): # alpha y beta definen los valores extremos
    Isc = self.fun_Ipv(Rs, Gp, IL, I0, b, 0)
    Vsc = self.fun_Vpv(Rs, Gp, IL, I0, b, Isc)
    Voc = self.fun_Vpv(Rs, Gp, IL, I0, b, 0)
    if tf.reduce_any(tf.math.is_nan(Voc)): # Corrige el desbordamiento de la exponenecial
      try: Rs, Gp, IL, I0, b = [k.numpy().astype(np.longdouble) for k in [Rs, Gp, IL, I0, b]]
      except: Rs, Gp, IL, I0, b = [k.astype(np.longdouble) for k in [Rs, Gp, IL, I0, b]]
      Voc = self.funVoc(Rs, Gp, IL, I0, b, 0)
      Rs, Gp, IL, I0, b, Voc = [k.astype(np.float64) for k in [Rs, Gp, IL, I0, b, Voc]]
    Ioc = self.fun_Ipv(Rs, Gp, IL, I0, b, Voc)
    fun_Vmp = lambda Vmp: self.fun_Ipv(Rs, Gp, IL, I0, b, Vmp)-self.fun_foc(Rs, Gp, IL, I0, b, Vmp)
    IniIterations = 0
    if VmpIni==None: Vmp0, Vmp1 = Voc*alpha, Voc*beta
    else: Vmp0, Vmp1 = VmpIni
    error = tf.math.abs(Vmp1-Vmp0)
    while tf.reduce_all(tf.math.less_equal(np.float64(tol), error)):
      IniIterations+=1
      Vmp11 = Vmp0-fun_Vmp(Vmp0)*(Vmp1-Vmp0)/(fun_Vmp(Vmp1)-fun_Vmp(Vmp0))
      Vmp0, Vmp1 = Vmp1, Vmp11
      error = tf.math.abs(Vmp1-Vmp0)
      if IniIterations==MaxIterations:
        break        
    Vmp = Vmp1
    Imp = self.fun_Ipv(Rs, Gp, IL, I0, b, Vmp)
    return tf.concat([tf.cast(k, tf.float64) for k in [Isc, Vsc, Imp, Vmp, Ioc, Voc]], axis=1) 