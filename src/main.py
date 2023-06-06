from os.path import join, isdir, exists
from os import system, makedirs, environ
import subprocess, sys, time, logging

def pip_list():
  args = [sys.executable, "-m", "pip", "freeze"]
  p = subprocess.run(args, check=True, capture_output=True)
  pip_list = p.stdout.decode().split('\r\n')[:-1]
  return [pp.split('==')[0] for pp in pip_list]

verbose = sys.argv[3]
assert isinstance(verbose, str)
assert verbose in ['T', 'F', 't', 'f', 'true', 'false', 'True', 'False']
verbose = verbose.upper()=='T' or verbose.upper()=='TRUE'

logging.basicConfig(filename="logfile.log", filemode='w', format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if verbose:
  formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s","%Y-%m-%d %H:%M:%S")
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.DEBUG)
  handler.setFormatter(formatter)
  logger.addHandler(handler)


logger.info("===========================================================================")
logger.info("                            Project folder paths                           ".upper())
logger.info("===========================================================================")

global_path = sys.argv[1]
assert isinstance(global_path, str)
assert exists(global_path)
assert isdir(global_path)
logger.info(global_path)

scr_path = join(global_path, 'scr')
assert exists(scr_path)
assert isdir(scr_path)
logger.info(scr_path)

BBDD_path = join(global_path, 'inputs')
assert exists(BBDD_path)
assert isdir(BBDD_path)
logger.info(BBDD_path)

results_path = join(global_path, 'results')
if not exists(results_path):
  makedirs(results_path)
else:
  assert isdir(results_path)
logger.info(results_path)
logger.info("")
logger.info("")

flag = int(sys.argv[2])
assert isinstance(flag, int)
assert flag in [0, 1]

installed_packages = pip_list()
pip_path = join(scr_path, "python38","python -m pip")
for package in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'keras', 'scipy', 'openpyxl', 'tensorflow', 'python-dateutil', 'tensorflow-probability', 'sympy', 'pyDOE']:
  if package not in installed_packages:
    system(pip_path+" install {0}".format(package))

# matplotlib
environ['MPLCONFIGDIR'] = join(scr_path, "python38")

from PVPredict import PVPredictClass
from PVModel import PVModelClass
from kerasModel import CustomModel, DNNmodelClass
from PVDataSet import DataSetClass
from PVoverfitting import overfittingClass
from PV3D import SurFacePlot
from PVComplement import path1, path2, path3, path4, path5, path6, ModelParamsMATLAB, PVfactorsDataSheet, PVModules, IEC61853, normpath

print(path1)
print(path2)
print(path3)
print(path4)
print(path5)
print(path6)

# # Preprocesado de datos
# if 'DataSets' not in globals().keys():
#   DataSets = DataSetClass(PVModules, ModelParamsMATLAB, path1)
  
# # Generación de clases en funcióna los parametros
# PVmodel = PVModelClass(DataSets.PVModuleParams)