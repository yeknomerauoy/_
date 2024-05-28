import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

'''
here the connection between Input (1x1) and dense ( 1st layer 32x1) has a
matrix connecting the two Input (1x1) - M(32x1) - D(32x1) -M_1(32x32) -D_1(32x1)...
here D=M.INPUT
D_1=M_1.D
till
D_3(1x1)=M_3(1x32)*D_2(32x1)
number of parameters = total elements of D_j,M_j
'''
NN = tf.keras.models.Sequential([
tf.keras.layers.Input((1,)),
tf.keras.layers.Dense(units = 32, activation = 'tanh'),
tf.keras.layers.Dense(units = 32, activation = 'tanh'),
tf.keras.layers.Dense(units = 32, activation = 'tanh'),
tf.keras.layers.Dense(units = 1)
])

NN.summary()
optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

def ode_system(t, net):
    t = t.reshape(-1,1) #normalize
    t = tf.constant(t, dtype = tf.float32) # datatype change to f32
    t_0 = tf.zeros((1,1)) 
    one = tf.ones((1,1))
    with tf.GradientTape() as tape:
        tape.watch(t)
        u = net(t)
        u_t = tape.gradient(u, t)
        ode_loss = u_t - tf.math.cos(2*np.pi*t)
    IC_loss = net(t_0) - one
    square_loss = tf.square(ode_loss) + tf.square(IC_loss)
    total_loss = tf.reduce_mean(square_loss)
    return total_loss


'''
================================================================================
================================================================================
'''
train_t = (np.array([0., 0.025, 0.475, 0.5, 0.525, 0.9, 0.95, 1.,
1.05, 1.1, 1.4, 1.45, 1.5, 1.55, 1.6, 1.95, 2.])).reshape(-1, 1)
#train_t = np.linspace(0, 2, 20)
train_loss_record = []
for itr in range(1000):
    with tf.GradientTape() as tape:
        train_loss = ode_system(train_t, NN)
        train_loss_record.append(train_loss)
        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))
    if itr % 1000 == 0:
        print(train_loss.numpy())
plt.figure(figsize = (10,8))
plt.plot(train_loss_record)
plt.show()
'''
================================================================================
================================================================================
'''
'''
================================================================================
================================================================================
'''
test_t = np.linspace(0, 2, 100)
train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
true_u = np.sin(2*np.pi*test_t)/(2*np.pi) + 1
pred_u = NN.predict(test_t).ravel()
plt.figure(figsize = (10,8))
plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(test_t, true_u, '-k',label = 'True')
plt.plot(test_t, pred_u, '--r', label = 'Prediction')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
plt.show()
'''
================================================================================
================================================================================

'''


input()
