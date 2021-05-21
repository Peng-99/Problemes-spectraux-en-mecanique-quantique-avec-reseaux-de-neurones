import numpy as np
import matplotlib.pyplot as plt
import math
import keras
from keras import layers, models, optimizers
from keras import backend as K
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

#define a loss function to minimize energy
def my_loss_fn(y_true, y_pred):
    y_pred = K.print_tensor(y_pred, message='y_pred = ')
    session = tf.compat.v1.Session()
    y_pred_np = y_pred.eval(session=session)
    energy_pred_np = energy_compute(a,b,linx,y_pred_np,VOH)
    return energy_pred_np

#oscillateur harmonique V(x)=0.5*m*omega*omega*x*x

hbar=1 #Planck constant
omega=1 #w
m=1
a=-5
b=5
pts=1000
linx = np.linspace(a,b,pts) 
#print(linx)
h = (b-a)/float(pts)
VOH = np.zeros_like(linx) 
wave = np.zeros_like(linx)

#oscillateur harmonique: wave function, potential function
VOH = 0.5*m*omega*omega*linx*linx #potential function of oscillateur harmonique
wave_ref = np.zeros_like(linx)
wave_ref = pow(m*omega/(math.pi*hbar),0.25)*np.exp(-m*omega*(linx*linx)/(2*hbar)) #wave function of oscillateur harmonique
energy_wave_ref = energy_compute(a,b,linx,wave_ref,VOH)

#machine learning: predication function
model = models.Sequential([
                           layers.Dense(200, input_shape=(1,), activation='relu'),
                           layers.Dense(200, input_shape=(1,), activation='relu'),
                           layers.Dense(1),
]) #keras
model.summary() #print
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss=my_loss_fn, optimizer=opt) 

model.fit(linx,wave,batch_size=50,epochs=40)
predictions = model.predict(linx)
preds = predictions.reshape(-1) 
energy_preds = energy_compute(a,b,linx,preds,VOH)
    
print('')
print('Energy_wave = ',energy_wave_ref)
print('Energy_pred = ',energy_preds)


plt.title('Approx. par N.N. avec custom loss')
plt.plot(linx,wave_ref,c='r',label = 'true wave function')
plt.plot(linx[0:pts-1:10],preds[0:pts-1:10],marker='x',c='forestgreen',label = 'custom loss',linestyle='None')
plt.ylabel('y')
plt.xlabel('x')
plt.legend() 
plt.savefig('custom_loss.pdf')