import os
import numpy as np
import pandas as pd
import tensorflow as tf
from .special import *
from sklearn.model_selection import train_test_split

class DataSetClass:
  def __init__(self, Modules, ModelParams, path):
    self.Path = path
    self.Data = {}
    self.PVModuleParams = {}
    self.DataSet = {}
    # for PVModule in Modules:
    #   DataPath = normpath(self.Path,PVModule)
    #   try: 
    #     df = pd.concat([pd.read_csv(normpath(DataPath,'Cocoa_'+PVModule+'.zip'), compression='zip'),
    #                     pd.read_csv(normpath(DataPath,'Eugene_'+PVModule+'.zip'), compression='zip')
    #                     ], ignore_index=True)
    #   except: 
    #     df  = pd.read_csv(normpath(DataPath,'Golden_'+PVModule+'.csv'))
    #   TimeSeries = df.drop(df.columns[np.arange(0, 41)], axis=1)
    #   df.drop(df.columns[np.append([2, 4, 6,  8, 10, 12], np.arange(14, df.shape[1]))], axis=1, inplace=True)
    #   df.rename(columns={k:['Time', 'Irradiance (W/m2)', 'Temperature (°C)'][n] for n,k in enumerate(df.columns[:3])}, inplace=True)
    #   self.Data[PVModule] = {'df':df, 'TimeSeries':TimeSeries}
    #   try:
    #     self.DataSet[PVModule] = self.CreateDataSet(PVModule)
    #   except:pass
    #   try:
    #     params = {}
    #     models = ModelParams[ModelParams.Module==PVModule]
    #     for k in models[ModelParams.columns[1]]:
    #       PVmodel = models[(models[ModelParams.columns[1]]==k)&(models[ModelParams.columns[3:8]]>=0).apply(np.prod, axis=1).astype(bool)].to_dict('records')[0]
    #       PVmodel = {k.replace(' ', ''):PVmodel[k]  for k in PVmodel}
    #       if   k == 5:  PVmodel['name'], PVmodel['label'] = 'De Soto', 'DS'
    #       elif k == 6:  PVmodel['name'], PVmodel['label'] = 'Dobos', 'D'
    #       elif k == 7:  PVmodel['name'], PVmodel['label'] = 'Boyd', 'B'
    #       elif k == 11: PVmodel['name'], PVmodel['label'] = 'Proposed', 'P'
    #       params[k] = PVmodel
    #     self.PVModuleParams[PVModule] = params
    #   except:pass
      
  def CreateDataSet(self, PVModule, seed=123, DataSet=[70, 20, 10], Nday=5):
    np.random.seed(seed), tf.random.set_seed(seed)
    df = self.Data[PVModule]['df']
    days = df[df.columns[0]].str.split('T').str[0]
    daysT = days.unique()[np.random.randint(days.unique().size, size=Nday)]
    for k in range(Nday):
      if not(k): dayP, idxPlot, DataPlot = (days==daysT[int(0)]).to_numpy(), [], []
      else: dayP = np.logical_or(dayP,(days==daysT[int(k)]).to_numpy())
      DataPlot.append([
          df[days==daysT[int(k)]][df.columns[[1,2]]].to_numpy(dtype='float32'),
          df[days==daysT[int(k)]][df.columns[np.arange(3, df.shape[1])]].to_numpy(dtype='float32'),
          df[days==daysT[int(k)]][df.columns[0]].str.split('T').str[1].str[:-3].to_numpy()])
      idxPlot += df[days==daysT[int(k)]].index.values.tolist()
      if not(k): xx, yy = DataPlot[-1][0], DataPlot[-1][1]
      else: xx, yy = np.vstack([xx, DataPlot[-1][0]]), np.vstack([yy, DataPlot[-1][1]])
    df = df.iloc[np.logical_not(dayP)]
    X, Y = [df[df.columns[k]] for k in [[1,2], np.arange(3, df.shape[1])]]
    ridxs = tf.random.shuffle(tf.range(X.shape[0]))
    X, Y = [pd.DataFrame(tf.gather(k, ridxs).numpy().astype(np.float32)) for k in [X, Y]]
    X_tr,  X_tst, Y_tr,  Y_tst = train_test_split(X, Y, test_size=sum(DataSet[1:3])/sum(DataSet), random_state=seed)
    X_tst, X_val, Y_tst, Y_val = train_test_split(X_tst, Y_tst, test_size=DataSet[1]/sum(DataSet[1:3]), random_state=seed)
    idxTest = idxPlot + Y_tst.index.values.tolist()
    X, X_tr, X_val, X_tst = [k.to_numpy(dtype='float32') for k in [X, X_tr, X_val, X_tst]]
    Y, Y_tr, Y_val, Y_tst = [k.to_numpy(dtype='float32') for k in [Y, Y_tr, Y_val, Y_tst]]
    X_tst, Y_tst = [np.vstack(k).astype(np.float32) for k in [[xx, X_tst], [yy, Y_tst]]]
    return [X, X_tr, X_val, X_tst, Y, Y_tr, Y_val, Y_tst, DataPlot, idxTest]