# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 10:08:23 2022

import numpy as np
import tensorflow as tf 
sess = tf.compat.v1.Session()
import pandas as pd
import os
#os.environ['CUDA_VISIBLE_DEVICES']="0"
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import BatchNormalization,Dense, Dropout, Activation,Concatenate,LeakyReLU
import scipy.stats as si
pd.options.display.float_format = '{:40,.8f}'.format
np.seterr(divide='ignore')
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


from tensorflow.python.distribute.cluster_resolver.tpu.tpu_cluster_resolver import is_running_in_gce  # pylint: disable=unused-import
from tensorflow.python.distribute.cluster_resolver.tpu.tpu_cluster_resolver import TPUClusterResolver
from tensorflow.python.util.tf_export import tf_export

def Data_process(Picklefile):
    
    Datagroup=pd.read_pickle(Picklefile)
    Datagroup_X=[Datagroup['S'],Datagroup['K'],Datagroup['T'],Datagroup['r'],Datagroup['G-vol']]

    Datagroup_X=pd.concat(Datagroup_X,axis=1)
    Norm_Datagroup_X=(Datagroup_X-Datagroup_X.mean())/Datagroup_X.std()
    Norm_Datagroup_X['Trading_Date']=Datagroup['Trading_Date']

    
    Datagroup_Y=np.log(Datagroup['Opt_Price'])
    Datagroup_Y=Datagroup_Y.replace([np.inf, -np.inf], np.nan)
    Datagroup_Y=pd.DataFrame(Datagroup_Y)
    Datagroup_Data=pd.concat([Norm_Datagroup_X,Datagroup_Y],axis=1)
    Datagroup_Data = Datagroup_Data.dropna()
    Datagroup_Data_X=pd.concat([Datagroup_Data['S'],Datagroup_Data['K'],Datagroup_Data['T'],Datagroup_Data['r'],Datagroup_Data['G-vol']],axis=1)
    Datagroup_Data_Y=pd.concat([Datagroup_Data['Opt_Price']],axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(Datagroup_Data_X, Datagroup_Data_Y, test_size=0.2)
    return  X_train, X_test, Y_train, Y_test

Traintest=Data_process("ITM_call_data.pkl")
def model_arch():
    #strategy = tf.distribute.MirroredStrategy()
    #with tf.device('/gpu:0'):
    model = Sequential()
    model.add(Dense(200,kernel_initializer=tf.keras.initializers.GlorotNormal(),activation=LeakyReLU(alpha=0.1),input_shape=(5,)))
    model.add(Dropout(0.2))
    model.add(Dense(100,kernel_initializer=tf.keras.initializers.GlorotNormal(),activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout(0.2))
    model.add(Dense(50,kernel_initializer=tf.keras.initializers.GlorotNormal(),activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    optimizer=tf.keras.optimizers.SGD()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])    
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)
    model.fit(Traintest[0], Traintest[2], epochs=100, validation_split=0.20, verbose=1,
                batch_size=1000,validation_data=(Traintest[1],Traintest[3]),callbacks=[es])
    return model
@tf.function
def p_val_est():
    Gradient_sq_sum=pd.DataFrame(columns=["S_gradient","K_gradient","T_gradient","r_gradient","G-vol_gradient"])
    conv_x=np.array_split(Traintest[0],31770)
    for z in range(5):
        Stack_of_models=[]
        for k in range(5):
            TModel=model_arch()
            Stack_of_models.append(TModel)
        Model_outputs=pd.DataFrame()
        for p in range(len(Stack_of_models)):
            out_batch=[]
            for t in range(len(conv_x)):
                Output_array=Stack_of_models[p](tf.Variable(conv_x[t]))
                out_batch.append(Output_array)
            Model_outputs['Model_'+str(p)]=pd.DataFrame(np.concatenate(out_batch))
        Var_Cov=Model_outputs.cov()
        Multi_norm_sample=np.random.multivariate_normal(np.zeros(len(Model_outputs.columns)), Var_Cov)   
        Model_ind=np.argmax(Multi_norm_sample)
        Gradient_list=[]
        for s in range(len(conv_x)):
            Split=tf.Variable(conv_x[s])
            with tf.GradientTape(persistent=True) as tape:
                pred_y=Stack_of_models[Model_ind](Split)
            Model_gradients=tape.gradient(pred_y,Split)
            Gradient_list.append(Model_gradients)
        Gradient_list=tf.concat(Gradient_list,0)
        Gradient_sq_sum.loc[z]=[np.mean(np.square(Gradient_list[:,0])),np.mean(np.square(Gradient_list[:,1])),np.mean(np.square(Gradient_list[:,2])),np.mean(np.square(Gradient_list[:,3])),np.mean(np.square(Gradient_list[:,4]))]
    for m in range(len(Gradient_sq_sum.columns)):
        print(f'The 90th, 95th and 99th percentiles for {list(Traintest[0].columns.values)[m]} are: ' + repr(np.percentile(Gradient_sq_sum[Gradient_sq_sum.columns[m]], 90)) + ',' + repr(np.percentile(Gradient_sq_sum[Gradient_sq_sum.columns[m]], 95)) + ', and ' + repr(np.percentile(Gradient_sq_sum[Gradient_sq_sum.columns[m]], 99))+ '.')
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
full=strategy.run(p_val_est())
       
    
