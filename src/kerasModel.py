import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l1
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, TerminateOnNaN, CSVLogger   
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# from tensorflow.keras.layers import Dense

class CustomModel(keras.Model):
  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape: 
      x0, x1 = tf.split(x, [5, 2], 1)
      y_pred = self(x0, training=True) # delta x
      # loss = self.compiled_loss(y, tf.concat([x, y_pred], axis=1), regularization_losses=self.losses)
      loss = self.compiled_loss(y, tf.concat([x, y_pred], axis=1)) # 
    grads = tape.gradient(loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self.compiled_metrics.update_state(y, tf.concat([x, y_pred], axis=1))
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    x, y = data
    y_pred = self(x, training=False)
    # self.compiled_loss(y, tf.concat([x, y_pred], axis=1), regularization_losses=self.losses)
    self.compiled_loss(y, tf.concat([x, y_pred], axis=1))
    self.compiled_metrics.update_state(y, tf.concat([x, y_pred], axis=1))
    return {m.name: m.result() for m in self.metrics}


class DNNmodelClass:
  def __init__(self, PVModule, X, Y, Model, Predict, seed, path,
               Hidden=[100, 70, 40, 10], 
               training=False, mode='MSE', 
               actHidden='tanh', actOut='linear',
               activity_regularizer=l1(0.15)):
    Rs, Gp, IL, I0, b = Model(X)
    self.Xscaler   = StandardScaler().fit(np.hstack([Rs, Gp, IL, np.log(I0), b, X]))
    self.Yscaler   = StandardScaler().fit(np.hstack([Rs, Gp, IL, np.log(I0), b]))
    self.InYscaler = lambda x: x*self.Yscaler.var_**.5+ self.Yscaler.mean_
    Zscaler        = StandardScaler().fit(Y)
    self.Zscaler   = lambda x: (x-Zscaler.mean_)/Zscaler.var_**2
    self.PVModule  = PVModule
    self.Hidden    = Hidden
    self.actHid    = actHidden
    self.actOut    = actOut
    self.actReg    = activity_regularizer
    self.Model     = Model
    self.Predict   = Predict
    self.training  = training
    self.seed_x    = seed
    self.path      = path
    
    self.Error     = {}
    if mode=='MSE':  
      self.Error=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    elif mode=='MSLE': 
      self.Error=tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.NONE)
    elif mode=='MAE': 
      self.Error=tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    elif mode=='MAPE': 
      self.Error=tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.NONE)

  def DNNModel(self, seed=1234):
    np.random.seed(seed), tf.random.set_seed(seed)
    input_dense = keras.Input(shape=(2,))
    dense = Dense(self.Hidden[0], activation=self.actHid, activity_regularizer=self.actReg)(input_dense)
    for num, Nneu in enumerate(self.Hidden[1:]):
      dense = Dense(Nneu, activation=self.actHid, activity_regularizer=self.actReg)(dense)
    output_dense=Dense(5, activation=self.actOut)(dense)
    model=CustomModel(input_dense, output_dense)
    return model

  def DNNPreprocessing(self, x):
    Rs, Gp, IL, I0, b = self.Model(x)
    return self.Xscaler.transform(np.hstack([Rs, Gp, IL, np.log(I0), b, x]))

  def DNNParams(self, x, model):
    xFit = self.DNNPreprocessing(x)
    Rs, Gp, IL, LogI0, b, _, _ = tf.split(xFit, axis=1, num_or_size_splits=7)
    DRs, DGp, DIL, DLogI0, Db  = tf.split(model(xFit), axis=1, num_or_size_splits=5)
    Rs, Gp, IL, LogI0, b = tf.split(self.InYscaler(tf.concat([Rs+DRs, Gp+DGp, IL+DIL, LogI0+DLogI0, b+Db], axis=1)), axis=1, num_or_size_splits=5)
    Rs, Gp, IL, b = [np.abs(k) for k in [Rs, Gp, IL, b]]
    return [Rs, Gp, IL, np.exp(LogI0), b]

  def DNNPredict(self, y_pred):
    Rs, Gp, IL, LogI0, b, _, _, DRs, DGp, DIL, DLogI0, Db = tf.split(y_pred, axis=1, num_or_size_splits=12)
    Rs, Gp, IL, LogI0, b = tf.split(self.InYscaler(tf.concat([Rs+DRs, Gp+DGp, IL+DIL, LogI0+DLogI0, b+Db], axis=1)), axis=1, num_or_size_splits=5)
    # x0, _, Dx0 = tf.split(y_pred, [5, 2, 5], 1)
    # Rs, Gp, IL, b = tf.split(self.InYscaler(x0+Dx0), axis=1, num_or_size_splits=5)
    Rs, Gp, IL, b = [tf.math.abs(k) for k in [Rs, Gp, IL, b]]
    Isc, Vsc, Imp, Vmp, Ioc, Voc = tf.split(self.Predict.predict(Rs, Gp, IL, tf.math.exp(LogI0), b), axis=1, num_or_size_splits=6)
    return tf.concat([Isc, Imp*Vmp, Imp, Vmp, Voc], axis=1)

















  def DNNMetric(self, y_true, y_pred, var):
    if self.training:
      return tf.reduce_mean(
           self.Error(tf.split(self.Zscaler(y_true),                  axis=1, num_or_size_splits=5)[var],
                      tf.split(self.Zscaler(self.DNNPredict(y_pred)), axis=1, num_or_size_splits=5)[var]))
    else:
      return tf.reduce_mean(self.Error(tf.split(y_true,                  axis=1, num_or_size_splits=5)[var],
                                       tf.split(self.DNNPredict(y_pred), axis=1, num_or_size_splits=5)[var]))

  def Isc(self, y_true, y_pred):
    return self.DNNMetric(y_true, y_pred, var=0)
    
  def Pmp(self, y_true, y_pred):
    return self.DNNMetric(y_true, y_pred, var=1)
    
  def Imp(self, y_true, y_pred):
    return self.DNNMetric(y_true, y_pred, var=2)
    
  def Vmp(self, y_true, y_pred):
    return self.DNNMetric(y_true, y_pred, var=3)
    
  def Voc(self, y_true, y_pred):
    return self.DNNMetric(y_true, y_pred, var=4)
    
  def CustomLoss(self, y_true, y_pred):
    return tf.reduce_sum([self.Isc(y_true, y_pred),
                          self.Pmp(y_true, y_pred),
                          self.Imp(y_true, y_pred),
                          self.Vmp(y_true, y_pred),
                          self.Voc(y_true, y_pred)])

  def DNNCallbacks(self):
    return [History(), 
            TerminateOnNaN(),
            EarlyStopping(patience=5, 
                          monitor="val_loss", 
                          restore_best_weights=True),
            ModelCheckpoint(filepath=self.path+'\\checkpoint\\'+self.PVModule+'\\'+str(self.seed_x),
                        save_weights_only=False,
                        monitor='val_loss',
                        mode='min',
                        save_best_only=False),
            CSVLogger(self.path+'\\checkpoint\\'+self.PVModule+'\\'+str(self.seed_x)+'\\history.csv', separator=",", append=True),] 