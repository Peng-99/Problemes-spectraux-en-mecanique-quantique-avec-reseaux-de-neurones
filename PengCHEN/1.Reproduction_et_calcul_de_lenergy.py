import numpy as np
import matplotlib.pyplot as plt
import math
import keras
from keras import layers, models, optimizers
import tensorflow as tf
from scipy import integrate
from scipy import interpolate
from scipy.stats import norm

#energy
def energy_compute (a,b,abscissa,function,pot):
  #interpolations
  tck_true = interpolate.splrep(abscissa, function, k=3, s=0)                               #W.F.
  tck_VOH = interpolate.splrep(abscissa, pot, k=3, s=0)                                     #P.F
  tck_true_carre = interpolate.splrep(abscissa, function*function, k=3, s=0)                #W.F.squared
  tck_true_pot_true_carre = interpolate.splrep(abscissa, pot*function*function, k=3, s=0)   #P.F * W.F.squared
  der_true = interpolate.splev(abscissa, tck_true, der=1)                                   #W.F.derivative
  tck_true_der = interpolate.splrep(abscissa,der_true*der_true, k=3,s=0)                    #W.F.derivative.squared
  int_true_carre = interpolate.splint(a,b,tck_true_carre)                                   #integral of W.F.squared
  int_pot_true_carre = interpolate.splint(a,b,tck_true_pot_true_carre)                      #integral of P.F * W.F.squared
  int_true_der = interpolate.splint(a,b,tck_true_der)                                       #integral of W.F.derivative.squared
  Energy = ((-pow(hbar,2)/(2*m))*(function[b]*der_true[b]-function[a]*der_true[a] 
                             - int_true_der) + int_pot_true_carre) / int_true_carre
  return Energy

#Définir les constantes
hbar=1 #Planck constant
omega=1 #la pulsation propre de l'ocillateur
m=1 #la masse d'une particule
#bornes de discrétisation
a=-5
b=5
#discrétisation de l'espace sur 1000 points
pts=1000
#array selon X
linx = np.linspace(a,b,pts) 
VOH = np.zeros_like(linx) 
wave = np.zeros_like(linx)
#oscillateur harmonique
VOH = 0.5*m*omega*omega*linx*linx #potentiel
wave = pow(m*omega/(math.pi*hbar),0.25)*np.exp(-m*omega*(linx*linx)/(2*hbar)) #f(cible)=function d'onde pour l'état fondamental
energy_wave = energy_compute(a,b,linx,wave,VOH)


########
# SECTION : Approximation par machine learning
########

#custom loss
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=0)

#Approximation par machine learning
model = models.Sequential([
                           layers.Dense(200, input_shape=(1,), activation='relu'),
                           layers.Dense(200, input_shape=(1,), activation='relu'),
                           layers.Dense(1), # no activation -> linear function of the input
]) #keras
model.summary() #print
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss=my_loss_fn, optimizer=opt) 

model.fit(linx,wave,batch_size=50,epochs=40)
predictions = model.predict(linx)
preds = predictions.reshape(-1)


energy_preds = energy_compute(a,b,linx,preds,VOH)

print('')
print('Energy_wave = ',energy_wave)
print('Energy_pred = ',energy_preds)

plt.title('Approx. par N.N. avec custom loss')
plt.plot(linx,wave,c='r',label = 'true wave function')
plt.plot(linx[0:pts-1:10],preds[0:pts-1:10],marker='x',c='forestgreen',label = 'custom loss',linestyle='None')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.savefig('custom loss.pdf')

epochs=range(len(history.history['loss']))
plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.title('Traing loss')
plt.legend()
plt.savefig('loss.pdf')