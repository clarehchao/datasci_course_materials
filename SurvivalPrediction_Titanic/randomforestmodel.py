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
rftrain_df = train_df

# fill the null values with the most common place
if sum(rftrain_df.Embarked.isnull()) > 0:
    common = rftrain_df.Embarked.dropna().mode().values
    rftrain_df.Embarked[rftrain_df.Embarked.isnull()] = common
    print common

if sum(rftrain_df.Age.isnull()) > 0:
    common = rftrain_df.Age.dropna().median()
    rftrain_df.Age[rftrain_df.Age.isnull()] = common
    print common

# include new features
rftrain_df['FamilySize'] = rftrain_df['SibSp'] + rftrain_df['Parch']
rftrain_df['AgeClass'] = rftrain_df['Age']*rftrain_df['Pclass']

# drop the unused columns
# when using dmatrices, one doesn't need to drop the unnecessary columns since it will pick the needed data out
#rftrain_df = rftrain_df.drop(['Name','Ticket', 'Cabin', 'PassengerId'], axis=1) 

#formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
#formula_ml = 'Survived ~ C(Sex) + FamilySize + AgeClass + C(Embarked)'
formula_ml = 'Survived ~  C(Sex) + Age + SibSp + Parch + C(Embarked)'
y,x = dmatrices(formula_ml, data = rftrain_df,return_type='dataframe')
y = np.asarray(y).ravel()  # random forest takes a 1D flatten array
result_rf = ske.RandomForestClassifier(n_estimators=200).fit(x,y)
score = result_rf.score(x,y)
#oh = 1 - sum(abs(y - result_rf.predict(x)))  #this is the same as score calculated above
print 'score is {0}'.format(score)

# make the prediction with the test data
# prepare the test data
# load test data
testfname = '%s/test.csv' % fdir
test_df = pd.read_csv(testfname,header=0)
test_df['Survived'] = 1.0
ids = test_df['PassengerId'].values

rftest_df = test_df
# fill the null values with the most common place
if sum(rftest_df.Embarked.isnull()) > 0:
    common = rftest_df.Embarked.dropna().mode().values
    rftest_df.Embarked[rftest_df.Embarked.isnull()] = common
    #print common

if sum(rftest_df.Age.isnull()) > 0:
    common = rftest_df.Age.dropna().median()
    rftest_df.Age[rftest_df.Age.isnull()] = common
    #print common

rftest_df['FamilySize'] = rftest_df['SibSp'] + rftest_df['Parch']
rftest_df['AgeClass'] = rftest_df['Age']*rftest_df['Pclass']

junk,x = dmatrices(formula_ml, data = rftest_df,return_type='dataframe')
test_x = np.asarray(x)  
output = result_rf.predict(test_x).astype(int)


#outfname = '%s/randomforestoutput.csv' % fdir
outfname = '%s/randomforestoutput2.csv' % fdir
theoutputfile = open(outfname, "wb")
open_file_object = csv.writer(theoutputfile)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
theoutputfile.close()

