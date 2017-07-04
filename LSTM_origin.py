import numpy as np
import tensorflow as tf
import matplotlib as mpl 
from sklearn.externals import joblib
import pickle
mpl.use('Agg')
from matplotlib import pyplot as plt

learn=tf.contrib.learn

HIDDEN_SIZE=30
NUM_LAYERS=2
TIMESTEPS=10
TRAINING_STEPS=10000
BATCH_SIZE=32

TRAINING_EXAMPLES=10000
TESTING_EXAMPLES=1000
SAMPLE_GAP=0.01

def generate_data(seq):
    X=[]
    y=[]
    for i in range(len(seq)-TIMESTEPS-1):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y):
    #tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
    cell=tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]) #tf.nn.rnn_cell.MultiRNNCell
    x_=tf.unstack(X,axis=1)  #unpack upgrade to unstack
    output,_=tf.contrib.rnn.static_rnn(cell, x_, dtype=tf.float32)
    output=output[-1]
    prediction,loss=learn.models.linear_regression(output, y)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), 
                                                optimizer="Adagrad", learning_rate=0.1)
    return prediction, loss, train_op

regressor = learn.Estimator(model_fn=lstm_model)

test_start=TRAINING_EXAMPLES*SAMPLE_GAP
test_end=(TRAINING_EXAMPLES+TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)


    
predicted = [[pred] for pred in regressor.predict(test_X)]

rmse=np.sqrt(((predicted-test_y) ** 2).mean(axis=0))
print ('Mean Square Error is: %f' % rmse[0])

#plot two data sets
fig=plt.figure()
label=['predicted','real_sin']
plt.plot(predicted,'g*')
plt.plot(test_y,'r')
plt.legend(label,loc=0,ncol=2)
#plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])

fig.savefig('sin1.png')