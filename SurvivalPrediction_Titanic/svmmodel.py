import csv as csv
import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from KaggleAux import predict as ka
from sklearn import datasets, svm
import sklearn.ensemble as ske

# data science via pandas (data frame like structure as in R)
fdir = '/home/clareh/Documents/coursera/IntroDataScience/datasci_course_materials/SurvivalPrediction_Titanic'
trainfname = '%s/train.csv' % fdir
train_df = pd.read_csv(trainfname,header=0)  # row = 0 is the header row

# random forest machine learning
# fill up all null with the median values
svmtrain_df = train_df

formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'

# fill the null values with the most common place
if sum(svmtrain_df.Embarked.isnull()) > 0:
    common = svmtrain_df.Embarked.dropna().mode().values
    svmtrain_df.Embarked[svmtrain_df.Embarked.isnull()] = common
    print common

if sum(svmtrain_df.Age.isnull()) > 0:
    common = svmtrain_df.Age.dropna().median()
    svmtrain_df.Age[svmtrain_df.Age.isnull()] = common
    print common

# try support vector machine
# select the features for training
y2,x2 = dmatrices(formula_ml, data = svmtrain_df,return_type='matrix')
print svmtrain_df
feat1 = 7
feat2 = 3
print x2
X2 = np.asarray(x2)
X2 = X2[:,[feat1,feat2]]
#print X2


Y2 = np.asarray(y2)
Y2 = y2.flatten()

nsample = len(X2)
np.random.seed(0)
order = np.random.permutation(nsample)

# randomize the order of x and y
X2 = X2[order]
Y2 = Y2[order].astype(np.float)

# select 90% of the sample
indx = int(0.9*nsample)
x_svmtrain = X2[:indx]
y_svmtrain = Y2[:indx]
x_svmtest = X2[indx:]
y_svmtest = Y2[indx:]


kerneltype = ['linear','rbf','poly']
color_map = plt.cm.RdBu_r
for nfig,kernel in enumerate(kerneltype):
    clf = svm.SVC(kernel=kernel,gamma=3)
    clf.fit(x_svmtrain,y_svmtrain)
    #print X2[:,0],X2[:,1]
    
    plt.figure(nfig)
    plt.scatter(X2[:,0],X2[:,1],c=Y2,zorder=10,cmap=color_map)


    # circle out the test data
    plt.scatter(x_svmtest[:,0],x_svmtest[:,1],s=80,facecolors='none',zorder=10)
    
    plt.axis('tight')
    x_min = X2[:,0].min()
    x_max = X2[:,0].max()
    y_min = X2[:,1].min()
    y_max = X2[:,1].max()

    xx,yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
    
    # result in color
    z = z.reshape(xx.shape)
    plt.pcolormesh(xx,yy,z>0,cmap=color_map)
    plt.contour(xx,yy,z,colors=['k','k','k'],linestyles=['--','--','--'],level=[-.5,0,.5])
    plt.title(kernel)
    plt.show()
    
 
# load test data
testfname = '%s/test.csv' % fdir
test_df = pd.read_csv(testfname,header=0)
test_df['Survived'] = 1.0
ids = test_df['PassengerId'].values

# fill the null values with the most common place
if sum(test_df.Embarked.isnull()) > 0:
    common = test_df.Embarked.dropna().mode().values
    test_df.Embarked[test_df.Embarked.isnull()] = common
    #print common

if sum(test_df.Age.isnull()) > 0:
    common = test_df.Age.dropna().median()
    test_df.Age[test_df.Age.isnull()] = common
    #print common


for k in kerneltype:
    clf = svm.SVC(kernel=k,gamma=3).fit(x_svmtrain,y_svmtrain)
    y3,x3 = dmatrices(formula_ml,data=test_df,return_type='dataframe')
    # Change the interger values within x.ix[:,[6,3]].dropna() explore the relationships between other 
    # features. the ints are column postions. ie. [6,3] 6th column and the third column are evaluated. 
    res_svm = clf.predict(x3.ix[:,[6,3]].dropna()) 
    res_svm = DataFrame(res_svm,columns=['Survived'])
    res_svm.to_csv("%s/output_svm_%s.csv" % (fdir,k)) # saves the results for you, change the name as you please. 

    
