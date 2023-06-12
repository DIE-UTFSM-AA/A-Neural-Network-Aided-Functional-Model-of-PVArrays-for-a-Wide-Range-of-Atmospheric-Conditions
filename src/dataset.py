import os, itertools, copy, pickle
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class dataset:
  url_bbdd = "https://datahub.duramat.org/dataset/e024ad97-f724-476c-8712-797a5b213cdb/resource/87d2133e-b2d5-4282-a899-9b6f5aa23d38/download/data-for-validating-models.zip"
  columns = {'POA irradiance CMP22 pyranometer (W/m2)': 'S',
           'PV module back surface temperature (degC)':'T',
           'Isc (A)':'Isc',
           'Pmp (W)':'Pmp', 
           'Imp (A)':'Imp',
           'Vmp (V)':'Vmp', 
           'Voc (V)':'Voc'
           }

  def __init__(self, project_path:str):
    # check files
    source_data = os.path.join(project_path, 'Data For Validating Models')
    if not all([os.path.exists(source_data), os.path.isdir(source_data)]):
      print('download data')
      self.download(project_path)

    self.data_path = os.path.join(project_path, 'data')
    if not all([os.path.exists(self.data_path), os.path.isdir(self.data_path)]):
      os.makedirs(self.data_path)

    # obtaining auxiliary paths
    self.folders = {}
    self.list_of_modules = {}
    for dir_name in os.listdir(source_data):
      dir_ = os.path.join(source_data, dir_name)
      if os.path.isdir(dir_):
        self.folders[dir_name] = dir_
        self.list_of_modules[dir_name] = [module.split('_')[-1].replace('.csv','') for module in os.listdir(dir_)]
    self.modules = np.unique(list(itertools.chain(*self.list_of_modules.values()))).tolist()
    
  def download(self, folder:str):    
    with urlopen(self.url_bbdd) as zipresp:
      with ZipFile(BytesIO(zipresp.read())) as zfile:
        zfile.extractall(folder)
  
  def read(self, module:str):
    df_list = []
    for folder_name, modules in self.list_of_modules.items():
      if module in modules:
        folder = self.folders[folder_name]
        csv_file = os.path.join(folder, '{0}_{1}.csv'.format(folder_name, module))

        # read csv
        df = pd.read_csv(csv_file, skiprows=2, names=range(720), low_memory=False)
        df.columns = df.iloc[0]
        df.drop(df.index[0], inplace=True)
        df_list.append(df[self.columns.keys()].astype(np.float32))
        del df
    return pd.concat(df_list, ignore_index=True).rename(columns=self.columns)
    
  def create_datasets(self, df0:pd.DataFrame, ratio:list=[7, 2, 1], seed:int=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # create X and Y sets
    df  = copy.copy(df0)
    dfx = df[['S', 'T']]
    dfy = df[['Isc', 'Pmp', 'Imp', 'Vmp', 'Voc']]

    # dataframe shuffle
    ridxs = tf.random.shuffle(tf.range(dfx.shape[0]))
    X, Y = [pd.DataFrame(tf.gather(dfw, ridxs)).values for dfw in [dfx, dfy]]

    # create sub-sets
    X_tr,  X_tst, Y_tr,  Y_tst = train_test_split(X, Y,         test_size=sum(ratio[1:3])/sum(ratio), random_state=seed)
    X_tst, X_val, Y_tst, Y_val = train_test_split(X_tst, Y_tst, test_size=ratio[1]/sum(ratio[1:3]),   random_state=seed)

    return [X, X_tr, X_val, X_tst, Y, Y_tr, Y_val, Y_tst]


  def __call__(self, module:str, ratio:list=[7, 2, 1], seed:int=0):
    
    processed_data = os.path.join(self.data_path, module+".pkl")


    if all([os.path.exists(processed_data), os.path.isfile(processed_data)]):
      with open(processed_data, 'rb') as handle:
        print('Read {0}'.format(module))
        data = pickle.load(handle)
    else:
      with open(processed_data, 'wb') as handle:
        print('Processed {0}'.format(module))
        # read csv
        df = self.read(module)

        # create datasets
        [X, X_tr, X_val, X_tst, Y, Y_tr, Y_val, Y_tst] = self.create_datasets(df, ratio=ratio, seed=seed)

        # save datasets
        data = {
          'df': df, 
          'x_data': [X, X_tr, X_val, X_tst],
          'y_data': [Y, Y_tr, Y_val, Y_tst],
        }
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


