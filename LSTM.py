import numpy as np
import tensorflow as tf
import matplotlib as mpl 
import time
from sklearn.externals import joblib
start = time.clock()
mpl.use('Agg')
from matplotlib import pyplot as plt 
learn=tf.contrib.learn
HIDDEN_SIZE=30 #hidden nodes
NUM_LAYERS=2   #number of LSTM layers
TIMESTEPS=5
TRAINING_STEPS=10000
BATCH_SIZE=32

# TRAINING_EXAMPLES=6293
# TESTING_EXAMPLES=629
#SAMPLE_GAP=0.01

def generate_data(seq):
    X=[]
    y=[]
    for i in range(len(seq)-TIMESTEPS-1):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y):
    cell=tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]) #tf.nn.rnn_cell.MultiRNNCell
    x_=tf.unstack(X,axis=1)  #unpack upgrade to unstack
    output,_=tf.contrib.rnn.static_rnn(cell, x_, dtype=tf.float32)
    output=output[-1]
    prediction,loss=learn.models.linear_regression(output, y)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer="Adagrad", learning_rate=0.0036)
    return prediction, loss, train_op

regressor = learn.Estimator(model_fn=lstm_model)


import csv
width=[]
time2=[]
with open('iceWidth2.csv') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        width.append(float(row['b']))
        time2.append(row['ACQUISITION_HOUR'])
time_fix=[]
# width_fix=[float(w) for w in width]
for t in time2:
     t=t.replace(' ','')
     t=t.replace('-','')
     time_fix.append(float(t))

# train_X, train_y = generate_data(width[6200:6274])

# test_X, test_y = generate_data(width[6275:6292])
# test_X2, test_y2 = generate_data(width[6274:6291])
train_X, train_y = generate_data(width[0:len(width)-19])

test_X, test_y = generate_data(width[len(width)-18:len(width)-1])
test_X2, test_y2 = generate_data(width[len(width)-19:len(width)-2])
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

#save the model 

#joblib.dump(regressor,'lstm.pkl')

start2=time.clock()
predicted = [[pred] for pred in regressor.predict(test_X)]
end2=time.clock()
rmse=np.sqrt(((predicted-test_y2) ** 2).mean(axis=0))
re=(np.abs(predicted-test_y2)).mean(axis=0)
print ('Mean Square Error is: %f' % rmse[0])
print ('relative error is : %f' %re)
print(predicted)
print(test_y2)

end = time.clock()
print('finish cal in %s' % str(end2 - start2))
print('finish all in %s' % str(end - start))
fig=plt.figure()
label=['predicted','real']
plt.plot(predicted,'g')
plt.plot(test_y2,'r')
plt.legend(label,loc=0,ncol=2)  #loc(设置图例显示的位置),
                                #ncol(设置列的数量，使显示扁平化，当要表示的线段特别多的时候会有用)
#plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])

fig.savefig('LSTM_5L.png')
