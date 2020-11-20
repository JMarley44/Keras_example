from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import sys

doFit = True
doPlots = True
doScaling = True

##### make plot function
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

    #Definition of variables
    themin = min( [min(X1), min(X2)])
    themax = max( [max(X1), max(X2)])
    bins = np.linspace(themin, themax, Nb)
    width = np.zeros(len(bins)-1)
    
    #Calculate bin centres and widths
    bincentre = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
            bincentre[i] = bins[i]+((bins[i+1]-bins[i])/2)
            width[i] = bins[i+1]-bins[i]

    #Offset the errorbars for background and signal from the centre so they don't overlap
    err_offset = 0.1*width
    
    #Set axis scale
    #ax.set_yscale('log')
    
    #Implement hatches for errors?
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

    #Calculate maximum value for y
    ymax = max([(max(n_back_pos)+max(back_err)), (max(n_sig_pos)+max(sig_err))])

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

###Plots of input variables
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

###Normalised plots of input variables (if doScaling is true)
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

'''The keras model'''

if doFit:
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

'''Function for recombining the y data of signal and background'''
def recombine(a):
    length = len(a)
    y_pred = np.zeros(length)
    bkg_count = 0
    res_count = 0

    #Recombining predicted background and signal to match the true binary array
    for i in range(length):
        if a[i]==0:
            y_pred[i]=res_bkg[bkg_count]
            bkg_count = bkg_count+1
        elif a[i]>0:
            y_pred[i]=res_sig[res_count]
            res_count = res_count+1
        else:
            print("ROC Curve Error")
            
    return y_pred

'''Function for predicting binary values'''
def predict(y_pred):
    y_pred_test = np.zeros(len(y_pred))

    for i in range(len(y_pred)):
        if y_pred[i]<0.5:
            y_pred_test[i]=0
        elif y_pred[i]>=0.5:
            y_pred_test[i]=1
            
    return y_pred_test


'''All sample'''

# Take all data, split into background and signal
x_bkg =  dataset_norm[  dataset_norm[:,8] == 0  ][:,0:8] 
x_sig =  dataset_norm[  dataset_norm[:,8] > 0  ][:,0:8] 

# Model prediction
res_sig = model.predict(x_sig)
res_bkg = model.predict(x_bkg)


makePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred", 20, False, xtitle="NN output", title="All sample")

### ROC Curve
# Recombine background and signal for entire data set
y_pred_all = recombine(y)

# Calculate the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_all)

# Plot (FPR against TPR)
plt.figure(figsize=(10.0,8.0))
plt.plot(fpr, tpr, label='RF')
plt.xlabel('False positive rate', fontsize=25)
plt.ylabel('True positive rate', fontsize=25)
plt.title('ROC curve', fontsize=40)
plt.legend(loc='best', fontsize=15)
plt.show()
plt.savefig("ROC_Curve.png")


'''Test sample'''

# Take test data, split into background and signal
data_test = dataset_norm[N_train:,:]
x_bkg = data_test[  data_test[:,8] == 0  ][:,0:8] 
x_sig = data_test[  data_test[:,8] > 0 ][:,0:8]


# Model prediction
res_sig = model.predict(x_sig)
res_bkg = model.predict(x_bkg)
makePlot(res_bkg.flatten(),res_sig.flatten(), "nn_pred_test", 20, False, xtitle="NN output", title="Test sample")

# Recombine background and signal for test data set
y_pred_test = recombine(y_test)

# Predict y binary
y_pred_test_bin = predict(y_pred_test)

# Calculate loss and accuracy
test_loss = metrics.log_loss(y_test, y_pred_test)
test_acc = metrics.accuracy_score(y_test, y_pred_test_bin) 


'''Train sample'''

# Take train data, split into background and signal
data_train = dataset_norm[:N_train,:]
x_bkg = data_train[  data_train[:,8] == 0  ][:,0:8] 
x_sig = data_train[  data_train[:,8] > 0 ][:,0:8]


# Model prediction
res_sig = model.predict(x_sig)
res_bkg = model.predict(x_bkg)

# Recombine background and signal for test data set
y_pred_train = recombine(y_train)

# Predict y binary
y_pred_train_bin = predict(y_pred_train)

# Calculate loss and accuracy
train_loss = metrics.log_loss(y_train, y_pred_train)
train_acc = metrics.accuracy_score(y_train, y_pred_train_bin)


'''Print statements'''

_,acc_train = model.evaluate(X_train, y_train)
print('Accuracy train: {:.2f}'.format(acc_train))

score,acc_test = model.evaluate(X_test, y_test)
print('Accuracy test: {:.2f}'.format(acc_test))

#print(score)

#print(y_test)
#print(y_test[ y_test>1  ] )

print('\nTrain sample loss is: {:.2f}'.format(train_loss), '\nTrain sample accuracy is: {:.2f}'.format(train_acc))
print('\nTest sample loss is: {:.2f}'.format(test_loss), '\nTest sample accuracy is: {:.2f}'.format(test_acc))
