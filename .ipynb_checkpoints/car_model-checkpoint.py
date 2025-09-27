from keras.layers import Conv1D, Activation, MaxPool1D, Flatten, Dense
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout,  Add
from keras.layers import Dropout
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import KFold
from tqdm import tqdm
import keras 

class AdaptiveDropout(Dropout):
    def __init__(self, initial_rate=0.1, max_rate=0.5, **kwargs):
        # 必须显式传递rate给父类
        super().__init__(rate=initial_rate, **kwargs)  
        self.initial_rate = initial_rate
        self.max_rate = max_rate
    
    def call(self, inputs, training=None):
        if training:
            # 计算当前训练进度
            progress = tf.cast(self.model.optimizer.iterations, tf.float32) 
            total_steps = 150 * (len(X_train) // 2000 ) # 假设总epoch=150, batch_size=2000
            current_rate = tf.minimum(
                self.initial_rate + (self.max_rate - self.initial_rate) * (progress / total_steps),
                self.max_rate
            )
            # 动态修改rate（注意：某些TF版本可能需要特殊处理）
            self.rate = current_rate.numpy() if tf.is_tensor(current_rate) else current_rate
        return super().call(inputs, training=training)

# 在模型中替换原Dropout
#model.add(AdaptiveDropout(0.1))  # 初始0.1，最终0.5
from tensorflow.keras import regularizers
'''
def NN_model(input_dim):
    model = tf.keras.Sequential([
        Dense(256, input_dim=input_dim, activation='swish',
              kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.3),  # 提高丢弃率
        Dense(128, activation='swish', 
              kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)),
        Dropout(0.3),
        Dense(64, activation='swish'),
        Dense(32, activation='swish'),
        Dense(1)
    ])
    return model

'''
def NN_model(input_dim):
    init = keras.initializers.glorot_uniform(seed=1)
    model = keras.models.Sequential()
    model.add(Dense(units=300, input_dim=input_dim, kernel_initializer=init, activation='softplus'))
    AdaptiveDropout(),
    #model.add(Dropout(0.1))
    model.add(Dense(units=300, kernel_initializer=init, activation='softplus'))
    AdaptiveDropout(),
    #model.add(Dropout(0.1))
    model.add(Dense(units=64, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=32, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=8, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=1))
    return model

from keras.callbacks import Callback, EarlyStopping
class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        #self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred3 = self.model.predict(X_train)
        y_pred = np.zeros((len(y_pred3), ))
        y_true = np.zeros((len(y_pred3), ))
        for i in range(len(y_pred3)):
            y_pred[i] = y_pred3[i]
        for i in range(len(y_pred3)):
            y_true[i] = y_train[i]
        trn_s = mean_absolute_error(y_true, y_pred)
        logs['trn_score'] = trn_s
        
        X_val, y_val = self.data[1][0], self.data[1][1]
        y_pred3 = self.model.predict(X_val)
        y_pred = np.zeros((len(y_pred3), ))
        y_true = np.zeros((len(y_pred3), ))
        for i in range(len(y_pred3)):
            y_pred[i] = y_pred3[i]
        for i in range(len(y_pred3)):
            y_true[i] = y_val[i]
            val_s = mean_absolute_error(y_true, y_pred)
        logs['val_score'] = val_s
        print('trn_score', trn_s, 'val_score', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)