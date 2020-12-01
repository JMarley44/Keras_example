from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import sys

doFit = True
doPlots = True
doScaling = True
doES = True

batch=24
node=9
node1=11
pat=25

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
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("Split_plots/"+tag+".png")
    if close: plt.close('all')

# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

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

model = None
## use the scaled instead

## shuffle the data
np.random.shuffle(dataset_norm)

shape = dataset_norm.shape
N = shape[0]

X = dataset_norm[:,0:8]
y = dataset_norm[:,8]

N_split = int(N/2)

XA = X[0:N_split,:]
yA = y[0:N_split]

XB = X[N_split:,:]
yB = y[N_split:]

N_train = int(0.75*N_split)

XA_train = XA[0:N_train,:]
yA_train = yA[0:N_train]

XA_test = XA[N_train:,:]
yA_test = yA[N_train:]

XB_train = XB[0:N_train,:]
yB_train = yB[0:N_train]

XB_test = XB[N_train:,:]
yB_test = yB[N_train:]

print("will split sets into test and training samples: train ", N_train,"total:", N)

'''The keras model'''
def keras_model(X_train,y_train,X_test,y_test):
    if doFit:
        # Define the model
        model = Sequential()
        model.add(Dense(node, input_dim=8, activation='relu'))
        model.add(Dense(node1, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        if doES:
            # Fit the model with early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat)
            history = model.fit(X_train, y_train, epochs=150, batch_size=batch,
                                        validation_data=(X_test,y_test), callbacks=[es])
        else:
            # Fit the model without early stopping
                history = model.fit(X_train, y_train, epochs=150, batch_size=batch,
                                    validation_data=(X_test,y_test), callbacks=[es])
        
        print(history.history.keys())
        plt.clf()
        plt.figure(figsize=(10.0,8.0))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("Split_plots/model_accuracy.png")
        plt.clf()
        plt.figure(figsize=(10.0,8.0))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig("Split_plots/model_loss.png")
        model.save("my_model")
        return model
    else:
        model = keras.models.load_model("my_model")

#_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))
#print(' Number of entries: ', len(X))

'''Function for recombining the y data of signal and background'''
def recombine(a, res_bkg, res_sig):
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

'''Create models A and B'''
model_a = keras_model(XA_train,yA_train,XA_test,yA_test)
model_b = keras_model(XB_train,yB_train,XB_test,yB_test)

'''A sample'''

# Take A data, split into background and signal
dataset_split_A = dataset_norm[0:N_split,:]
data_test_A = dataset_split_A[N_train:,:]
xA_bkg = data_test_A[  data_test_A[:,8] == 0  ][:,0:8] 
xA_sig = data_test_A[  data_test_A[:,8] > 0 ][:,0:8]

# Model prediction
a_sig = model_a.predict(xA_sig)
a_bkg = model_a.predict(xA_bkg)

### ROC Curve
# Recombine background and signal for data set A
yA_pred_test = recombine(yA_test, a_bkg, a_sig)

# Calculate the ROC curve
fpr_a, tpr_a, thresholds = metrics.roc_curve(yA_test, yA_pred_test)
auc_a = metrics.auc(fpr_a, tpr_a)

'''B sample'''

# Take B data, split into background and signal
dataset_split_B = dataset_norm[N_split:,:]
data_test_B = dataset_split_B[N_train:,:]
xB_bkg = data_test_B[  data_test_B[:,8] == 0  ][:,0:8] 
xB_sig = data_test_B[  data_test_B[:,8] > 0 ][:,0:8]

# Model prediction
b_sig = model_b.predict(xB_sig)
b_bkg = model_b.predict(xB_bkg)

### ROC Curve
# Recombine background and signal for data set B
yB_pred_test = recombine(yB_test, b_bkg, b_sig)

# Calculate the ROC curve
fpr_b, tpr_b, thresholds = metrics.roc_curve(yB_test, yB_pred_test)
auc_b = metrics.auc(fpr_b, tpr_b)

'''A on B data set'''
# Model prediction
ab_sig = model_a.predict(xB_sig)
ab_bkg = model_a.predict(xB_bkg)

### ROC Curve
# Recombine background and signal for data set B
yab_pred_test = recombine(yB_test, ab_bkg, ab_sig)

# Calculate the ROC curve
fpr_ab, tpr_ab, thresholds = metrics.roc_curve(yB_test, yab_pred_test)
auc_ab = metrics.auc(fpr_ab, tpr_ab)

'''B on A data set'''
# Model prediction
ba_sig = model_b.predict(xA_sig)
ba_bkg = model_b.predict(xA_bkg)

### ROC Curve
# Recombine background and signal for data set B
yba_pred_test = recombine(yA_test, ba_bkg, ba_sig)

# Calculate the ROC curve
fpr_ba, tpr_ba, thresholds = metrics.roc_curve(yA_test, yba_pred_test)
auc_ba = metrics.auc(fpr_ba, tpr_ba)

'''Plots'''
#Recombine the test data set for plotting the result
all_bkg_pred = np.concatenate((ab_bkg, ba_bkg))
all_sig_pred = np.concatenate((ab_sig, ba_sig))

makePlot(all_bkg_pred.flatten(),all_sig_pred.flatten(), "nn_pred_2_NN", 20, False, xtitle="NN output", title="Test sample")

#Recombine the data for ROC of AB and BA combined
y_all_test = np.concatenate((yB_test, yA_test))
y_all_test_pred = np.concatenate((yab_pred_test, yba_pred_test))

# Calculate the ROC curve
fpr_all, tpr_all, thresholds = metrics.roc_curve(y_all_test, y_all_test_pred)
auc_all = metrics.auc(fpr_all, tpr_all)

'''ROC Curve plot'''

# Plot (FPR against TPR)
plt.figure(figsize=(10.0,8.0))
plt.plot(fpr_a, tpr_a, label='A on A (area = {:.3f})'.format(auc_a), color='red')
plt.plot(fpr_b, tpr_b, label='B on B (area = {:.3f})'.format(auc_b), color='deepskyblue')

plt.plot(fpr_ab, tpr_ab, label='A on B (area = {:.3f})'.format(auc_ab), color='limegreen')
plt.plot(fpr_ba, tpr_ba, label='B on A (area = {:.3f})'.format(auc_ba), color='darkviolet')

plt.plot(fpr_all, tpr_all, label='AB and BA (area = {:.3f})'.format(auc_all), color='k')

plt.xlabel('False positive rate', fontsize=25)
plt.ylabel('True positive rate', fontsize=25)
plt.title('ROC curve', fontsize=40)
plt.legend(loc='best', fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
plt.savefig("Split_plots/ROC_Curve_2_NN.png")



