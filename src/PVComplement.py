import os
import pandas as pd
normpath = lambda x, y: os.path.normpath(x+os.sep+y)

# Rutas de acceso
path1 = normpath(os.getcwd(),'Inputs')
path2 = normpath(os.getcwd(),'Results')
path3 = normpath(path2,'overfitting')
path4 = normpath(path2,'weigths')
path5 = normpath(path2,'training')
path6 = normpath(path2,'3D_DNN')

# Archivos auxiliares
IEC61853 = pd.read_csv(normpath(path1,'IEC61853.csv'))
del IEC61853['FF Corrected (%)']
ModelParamsMATLAB  = pd.read_csv(normpath(path1,'ModelParamsv1.csv'), index_col=0)
PVfactorsDataSheet = pd.read_csv(normpath(path1,'PVfactors.csv'))
PVfactorsDataSheet.rename(columns={k:['Module','alphaIsc (%/°C)','alphaImp (%/°C)','betaVoc (%/°C)','betaVmp (%/°C)','deltaPmp (%/°C)'][n] for n,k in enumerate(PVfactorsDataSheet.columns)}, inplace=True)

# PVModules
PVModules = ModelParamsMATLAB.Module.unique()
