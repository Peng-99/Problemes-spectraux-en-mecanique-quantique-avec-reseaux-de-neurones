import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import norm
from scipy import integrate
from keras import layers, models, optimizers
import time
from __future__ import division
from scipy import interpolate
import math
from keras.callbacks import History
from google.colab import files

hbar = 1
omega = 1
m = 1
a = -5.
b = 5
#OPENING FILES
file1 = open("loss_moyen.txt", "w")
file2 = open("ecart-type_loss.txt", "w")
file3 = open("loss_total.txt","w")


############
#SECTION : CREATING THE TARGET
############

pts=1000                            #Number of points on the X and Y axis
linx = np.linspace (a,b,pts)        #X axis
norm = pow(m*omega/(math.pi*hbar),0.25)*np.exp(-m*omega*(linx*linx)/(2*hbar)) #f(cible)=function d'onde pour l'état fondamental


############
#SECTION : LOSS STUDY
############

runs = 30                                 #How many runs we want to compute a mean and a standard dev of the loss
nb_epochs = 100                           #How many epochs in each run
Loss = np.zeros((nb_epochs,runs))         #Loss array
array_epochs = np.arange(nb_epochs) + 1   #Epoch array


for k in range(0,runs):
  print('run n°',k+1)

  def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=0)

  #INITIALIZATION OF NEURAL NETWORK
  model = models.Sequential([
      layers.Dense(200, input_shape=(1,), activation='relu'),
      layers.Dense(200, input_shape=(1,), activation='relu'),      
      layers.Dense(1), # no activation -> linear function of the input
  ])
  opt = optimizers.Adam(learning_rate=0.001)
  model.compile(loss=my_loss_fn,optimizer=opt)
  history = History()
  hist = model.fit(linx, norm,batch_size=50, epochs=nb_epochs,callbacks=[history],verbose=0)
  predictions = model.predict(linx)
  preds = predictions.reshape(-1)


  #Writing the loss at each epoch
  for i in range(0,nb_epochs):
    Loss[i,k] = hist.history['loss'][i]


#Computing the mean
loss_moyen = np.sum(Loss,axis=1)/runs
#Computing the standard deviation
ecarttype = np.std(Loss,1)

#Writing data in files
np.savetxt(file1, loss_moyen, fmt='%.18e', delimiter=' ', newline='\n')
np.savetxt(file2, ecarttype, fmt='%.18e', delimiter=' ', newline='\n')
np.savetxt(file3,Loss, fmt='%.18e', delimiter=' ', newline ='\n')

file1.close()
file2.close()
file3.close()

#opening files
loss_total = np.loadtxt("loss_total.txt", dtype=float, delimiter=' ')
loss_moyen = np.loadtxt("loss_moyen.txt", dtype=float, delimiter=' ')
ecarttype_loss = np.loadtxt("ecart-type_loss.txt", dtype=float, delimiter=' ')

#preparing 3 runs to plot on top of the mean values
loss_1 = np.reshape(loss_total[:,0],-1)
loss_2 = np.reshape(loss_total[:,1],-1)
loss_3 = np.reshape(loss_total[:,2],-1)

plt.yscale('log')
plt.errorbar(array_epochs,loss_moyen, yerr = ecarttype_loss,marker='.',c='deepskyblue',label = 'mean over 30 runs')
plt.plot(array_epochs,loss_1, c = 'k',alpha=0.5,label='run 1')
plt.plot(array_epochs,loss_2,c='darkorange',alpha=0.5,label='run 2')
plt.plot(array_epochs,loss_3,c='r',alpha=0.5,label='run 3')
plt.ylabel('Loss')
plt.xlabel("Epoch")
plt.legend(title = "loss en fonction des epochs")
plt.tick_params(axis='both',labelsize='8')
plt.savefig('loss_epoch.pdf')
files.download('loss_epoch.pdf')