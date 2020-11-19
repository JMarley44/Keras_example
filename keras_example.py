from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import sys

doFit = True
doPlots = True
doScaling = True

##### make plot
def makePlot(X1,X2, tag, Nb, close, **kwargs):
    plt.clf()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0,8.0))

    xtitle=tag
    title = tag
    for key, value in kwargs.items():
        if key == "xtitle":
            xtitle = value
        elif key=="title":
            title = value


    themin = min( [min(X1), min(X2)])
    themax = max( [max(X1), max(X2)])
    bins = np.linspace(themin, themax, Nb)
    width = np.zeros(len(bins)-1)
    
    #Calculate bin centres
    bincentre = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
            bincentre[i] = bins[i]+((bins[i+1]-bins[i])/2)
            width[i] = bins[i+1]-bins[i]

    #Offset the errorbars for background and signal from the centre so they don't overlap
    err_offset = 0.1*width
    
    #Set axis scale
    #ax.set_yscale('log')
    
    #Implement hatches for errors
    #https://het.as.utexas.edu/HET/Software/Matplotlib/api/patches_api.html#matplotlib.patches.Patch.set_hatch
    
    #Background plot
    plt.hist(X1, bins=bins, label=['Background'],density=True)
    n_back_pos, edge = np.histogram(X1, bins=Nb-1, range=(themin,themax),density=True)
    n_back_count, edge1 = np.histogram(X1, bins=Nb-1, range=(themin,themax))
    back_err = np.sqrt(n_back_count)/(np.sum(n_back_count)*width)
    ax.errorbar(bincentre-err_offset, n_back_pos, xerr=None, yerr=back_err, ls='none', ecolor='k', fmt = 'ko')
    
    #Signal plot
    plt.hist(X2, bins=bins, label=['Signal'], histtype=u'step',density=True)
    n_sig_pos, edge2 = np.histogram(X2, bins=Nb-1, range=(themin,themax),density=True)
    n_sig_count, edge3 = np.histogram(X2, bins=Nb-1, range=(themin,themax))
    sig_err = np.sqrt(n_sig_count)/(np.sum(n_sig_count)*width)
    ax.errorbar(bincentre+err_offset, n_sig_pos, xerr=None, yerr=sig_err, ls='none', ecolor='r', fmt = 'ro')

    ymax = max([(max(n_back_pos)+max(back_err)), (max(n_sig_pos)+max(sig_err))])

    # if title == "Test sample":
    #     print("n back position: ", n_back_pos)
    #     print("n back count: ", n_back_count)
    #     print("X1: ", X1)
    #     print("edge1: ", edge1)
    #     print("edge2: ", edge2)
    #     print("bins: ", bins)

    plt.title(title, fontsize=40)
    plt.xlabel(xtitle, fontsize=25)
    plt.ylabel("# Entries (Norm)", fontsize=25)
    plt.legend(loc='upper right')
    plt.xlim(themin, themax)
    plt.ylim(0, 1.2*ymax)
    ax.set_xticks(bins, minor=True)
    ax.grid(which='minor', axis='x', alpha = 0.5)
    ax.grid(which='major', axis='y', alpha = 0.5)
    plt.savefig(tag+".png")
    if close: plt.close('all')
    
    
# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

attributes = ['# of times pregnant', 'Plasma glucose concentration', 'Diastolic blood pressure',
              'Triceps skin fold thickness', '2-hour serum insulin', 'BMI', 'Diabetes pedigree function',
              'Age', 'Class variable']

if doPlots:
    feat = [str(j) for j in attributes]
    for k,f in enumerate(feat):
        x = np.array(dataset[:,[k,8]])
        x1 = x[ x[:,1] == 0  ]
        x1 = x1[:,0]
        x2 = x[ x[:,1] >0  ]
        x2 = x2[:,0]
        makePlot(x1,x2,f,20, True)


### Normalise data if scaling is enabled
if doScaling:
    print( 'Data normalisation here')
    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    dataset_norm = np.insert(X_norm,8, y, axis=1)
else:
    print( 'Normalisation disabled')
    X_norm = X
    dataset_norm = np.insert(X_norm,8, y, axis=1)

#print(X_norm)
#print(dataset_norm)
#sys.exit()

if doPlots:
    feat = [str(j)+" Normal" for j in attributes]
    for k,f in enumerate(feat):
        x = dataset_norm[:,[k,8]]
        x1 = x[ x[:,1] == 0  ]
        x1 = x1[:,0]
        x2 = x[ x[:,1] >0  ]
        x2 = x2[:,0]
        makePlot(x1,x2,f,20, True)


#sys.exit()
model = None
## use the scaled instead

## shuffle the data
np.random.shuffle(dataset_norm)

shape = dataset_norm.shape
N = shape[0]

X = dataset_norm[:,0:8]
y = dataset_norm[:,8]

N_train = int(2*N/3)

X_train = X[0:N_train,:]
y_train = y[0:N_train]

X_test = X[N_train:,:]
y_test = y[N_train:]

print("will split test and training samples: train ", N_train,"total:", N)

if doFit:
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=150, batch_size=10,
                        validation_data=(X_test,y_test))
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("model_accuracy.png")
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("model_loss.png")
    model.save("my_model")
else:
    model = keras.models.load_model("my_model")

#_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))
#print(' Number of entries: ', len(X))

x_bkg =  dataset_norm[  dataset_norm[:,8] == 0  ][:,0:8] 
x_bkg_true =  dataset_norm[dataset_norm[:,8]==0]
x_sig =  dataset_norm[  dataset_norm[:,8] > 0  ][:,0:8] 

## prediction
res_sig = model.predict(x_sig)
res_bkg = model.predict(x_bkg)

#print(res_sig)
makePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred", 20, False, xtitle="NN output", title="All sample")

data_train = dataset_norm[N_train:,:]
x_bkg = data_train[  data_train[:,8] == 0  ][:,0:8] 
x_sig = data_train[  data_train[:,8] > 0 ][:,0:8]


## prediction
res_sig = model.predict(x_sig)
res_bkg = model.predict(x_bkg)
makePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred_test", 20, False, xtitle="NN output", title="Test sample")

#fpr, tpr, thresholds = roc_curve(true binary (y), res_sig)

_,acc_train = model.evaluate(X_train, y_train)
print('Accuracy train: {:.2f}'.format(acc_train))

score,acc_test = model.evaluate(X_test, y_test)
print('Accuracy test: {:.2f}'.format(acc_test))

#print(score)

#print(y_test)
#print(y_test[ y_test>1  ] )

