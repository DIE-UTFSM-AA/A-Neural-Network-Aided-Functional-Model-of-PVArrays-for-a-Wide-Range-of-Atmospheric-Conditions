from scipy import optimize
import sympy as sym
import numpy as np
import pandas as pd
import tensorflow as tf
from sympy.functions import exp
from sympy import LambertW
from sympy.solvers import solve
from scipy.constants import Boltzmann as k, eV as q, zero_Celsius as T0
from sympy.utilities.lambdify import lambdify


class overfittingClass:
  def __init__(self, PVModule, DataSets, PVfactorsDataSheet, IEC61853, PVPredictClass, mode:int=1):
    """
    Mode
      1: Valores de la norma IEC61853 corregidos
      2: Valores de la norma IEC61853 no corregidos
    """
    print(PVModule)
    self.PVModule=PVModule
    self.PVPredictClass = PVPredictClass
    [X, X_tr, X_val, X_tst, Y, Y_tr, Y_val, Y_tst, DataPlot, idxTest] = DataSets.DataSet[PVModule]
    Norma = IEC61853[IEC61853.PVModule==PVModule]
    _, self.alphaIsc, alphaImp, self.betaVoc, betaVmp, deltaPmp = PVfactorsDataSheet[PVfactorsDataSheet.Module==PVModule].to_numpy()[0]
    self.betaVoc/=100
    self.alphaIsc/=100

    if mode==1:
      data  = Norma[Norma.columns[1:8]]  
    elif mode==2: 
      data = Norma[Norma.columns[8:]]  
      
    self.x = data[data.columns[[1,0]]].to_numpy()
    self.Isc1, self.Voc1, self.Imp1, self.Vmp1, self.Pmp1 = [k.numpy() for k in tf.split(data[data.columns[2:]].to_numpy(), axis=1, num_or_size_splits=5)]

    self.S, self.T, self.Rs, self.Gp, self.IL, self.I0, self.b = sym.symbols('S, T, Rs, Gp, IL, I0, b') # T: en K
    Tr = 25+T0
    Sr = 1000
    # DeSoto en simbolico
    Eg  = lambda T:    (1.121*(1-0.0002677*(T-Tr)))
    ILF = lambda S, T: (S/Sr)*(self.IL+self.alphaIsc*(T-Tr))
    I0F = lambda S, T: self.I0*(T/Tr)**3*exp(q/k*(Eg(Tr)/Tr-Eg(T)/T))
    RsF = lambda S, T: self.Rs
    GpF = lambda S, T: self.Gp*(S/Sr)
    funVoc = lambda S, T, Rs, Gp, IL, I0, b:(ILF(S,T)+I0F(S,T))/GpF(S,T)-LambertW(b*I0F(S,T)/GpF(S,T)*exp(b*(ILF(S,T)+I0F(S,T))/GpF(S,T)))/b
    self.diffVoc_T = sym.diff(funVoc(self.S, self.T, self.Rs, self.Gp, self.IL, self.I0, self.b), self.T)

  def Findb(self, x, bF0):
    bArray = np.linspace(1e-4, 2, 100000)
    Rsx, Gpx, ILx, I0x = x
    bF1 = bF0.subs(self.Rs, Rsx)
    bF2 = bF1.subs(self.Gp, Gpx)
    bF3 = bF2.subs(self.IL, ILx)
    bF4 = bF3.subs(self.I0, I0x)
    bF5 = lambdify(self.b, bF4) # Lambda function 
    func = lambda x: bF5(x).real-self.betaVoc*100
    bx = bArray[np.nanargmin(np.abs(func(bArray)))]
    return bx

  def DeSotoFindZero(self, x, Isc, Pmp, Imp, Vmp, Voc, bF0):
    Isc, Imp = [k[0] for k in [Isc, Imp]]
    IFun1 = lambda Rsx, Gpx, ILx, I0x, bx, V: self.PVPredictClass().fun_Ipv(Rsx, Gpx, ILx, I0x, bx, V)
    IFun2 = lambda Rsx, Gpx, ILx, I0x, bx, V: self.PVPredictClass().fun_foc(Rsx, Gpx, ILx, I0x, bx, V)
    VocFu = lambda Rsx, Gpx, ILx, I0x, bx:    self.PVPredictClass().fun_Vpv(Rsx, Gpx, ILx, I0x, bx, 0)
    bx = self.Findb(x, bF0)
    Rsx, Gpx, ILx, I0x = x
    Isc1 = IFun1(Rsx, Gpx, ILx, I0x, bx, 0).numpy()
    Ioc1 = IFun1(Rsx, Gpx, ILx, I0x, bx, Voc).numpy()[0]
    Imp1 = IFun1(Rsx, Gpx, ILx, I0x, bx, Vmp).numpy()[0]
    Imp2 = IFun2(Rsx, Gpx, ILx, I0x, bx, Vmp).numpy()[0]
    # print('   ',Rsx, Gpx, ILx, I0x, bx, sum([(Isc1-Isc)**2, (Ioc1)**2, (Imp1-Imp)**2, (Imp2-Imp)**2]))
    return ((Isc1-Isc)**1, (Ioc1)**1, (Imp1-Imp)**1, (Imp2-Imp)**1)
  
  def Optimization(self, n, Sx, Tx, x0):
    Isc, Voc, Imp, Vmp, Pmp = [k[n] for k in [self.Isc1, self.Voc1, self.Imp1, self.Vmp1, self.Pmp1]]
    bF  = self.diffVoc_T.subs(self.S, Sx) 
    bF0 = bF.subs(self.T, Tx+T0)
    root = optimize.least_squares(lambda x: self.DeSotoFindZero(x, Isc, Pmp, Imp, Vmp, Voc, bF0), x0, jac='3-point',
              bounds=((1e-3, 1e-7, 1e-3, 1e-15), (1e3, 1, 5, 1e-5)))
    Rsx, Gpx, ILx, I0x = root.x    
    bx = self.Findb(root.x, bF0)
    ## reshape
    Rsx, Gpx, ILx, I0x, bx = [np.asarray([k]).reshape(-1,1) for k in [Rsx, Gpx, ILx, I0x, bx]]
    ## Predicción
    Isc1, _, Imp1, Vmp1, _, Voc1 = tf.split(self.PVPredictClass().predict(Rsx, Gpx, ILx, I0x, bx), axis=1, num_or_size_splits=6)  
    ## list2value
    Rsx, Gpx, ILx, I0x, bx = [k[0][0] for k in [Rsx, Gpx, ILx, I0x, bx]]
    Isc, Imp, Vmp, Voc, Pmp = [k[0] for k in [Isc, Imp, Vmp, Voc, Pmp]]
    Isc1, Imp1, Vmp1, Voc1, Pmp1 = [k.numpy()[0][0] for k in [Isc1, Imp1, Vmp1, Voc1, Imp1*Vmp1]]  
    error = sum([(k1-k2)**2/k1 for k1,k2 in [[Isc, Isc1], [Imp, Imp1], [Vmp, Vmp1], [Pmp, Pmp1], [Voc, Voc1]]])
    return bx, Rsx, Gpx, ILx, I0x, Isc1, Imp1, Vmp1, Voc1, Pmp1, Isc, Imp, Vmp, Voc, Pmp, error, root.cost
  

  def Search(self, path, save=True):
    np.seterr(all='ignore') 
    AA = []
    print("{:>4s}  {:>4s}  {:>6s}  {:>6s}  {:>6s}  {:>6s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
      'S','T','b','Rs','Gp','IL','I0','Isc_p','Imp_p','Vmp_p','Pmp_p','Voc_p','Isc_r','Imp_r','Vmp_r','Pmp_r','Voc_r','error','cost'))
    for n, [Sx, Tx] in enumerate(self.x):
      try:
        bx, Rsx, Gpx, ILx, I0x, Isc1, Imp1, Vmp1, Voc1, Pmp1, Isc, Imp, Vmp, Voc, Pmp, error, cost = self.Optimization(n, Sx, Tx, [Rsx, Gpx, ILx, I0x])
      except:
        bx, Rsx, Gpx, ILx, I0x, Isc1, Imp1, Vmp1, Voc1, Pmp1, Isc, Imp, Vmp, Voc, Pmp, error, cost = self.Optimization(n, Sx, Tx, x0=[1, 1/10, 1/10, 1e-9])
      print("{:>4d}  {:>4d}  {:>6.2f}  {:>6.2f}  {:>6.2f}  {:>6.2f}  {:>8.2e}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}".format(
            Sx, Tx, bx, Rsx, Gpx, ILx, I0x, Isc1, Imp1, Vmp1, Voc1, Pmp1, Isc, Imp, Vmp, Voc, Pmp, error, cost))
      AA.append([Sx, Tx, bx, Rsx, Gpx, ILx, np.log(I0x), Isc1, Imp1, Vmp1, Voc1, Pmp1, Isc, Imp, Vmp, Voc, Pmp, error, cost])
    df=pd.DataFrame(AA, columns=['S','T','b','Rs','Gp','IL','log(I0)','Isc_p','Imp_p','Vmp_p','Pmp_p','Voc_p','Isc_r','Imp_r','Vmp_r','Pmp_r','Voc_r','error','cost'])    
    if save:
      df.to_csv(path+'\\'+self.PVModule+'_overfitting.csv', index=False)
    return df