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

# drop all the null data
logittrain_df = train_df.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
logittrain_df  = logittrain_df.dropna()

#logittrain_df['FamilySize'] = logittrain_df['SibSp'] + logittrain_df['Parch']
#logittrain_df['AgeClass'] = logittrain_df['Age']*logittrain_df['Pclass']

# logistic regression
#formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
formula = 'Survived ~ C(Pclass) + Age + SibSp + Parch + C(Embarked)'
#formula = 'Survived ~ FamilySize + C(Sex) + AgeClass + C(Embarked)'
results = {}

female_train_df = logittrain_df[logittrain_df.Sex == 'female']
y1,x1  = dmatrices(formula,data=female_train_df,return_type='dataframe')
model1 = sm.Logit(y1,x1)  # create a model
res1 = model1.fit()  # fit the model to the training data
results['Logit_female'] = [res1,formula]
print res1.summary()

male_train_df = logittrain_df[logittrain_df.Sex == 'male']
y2,x2  = dmatrices(formula,data=male_train_df,return_type='dataframe')
model2 = sm.Logit(y2,x2)  # create a model
res2 = model2.fit()  # fit the model to the training data
results['Logit_male'] = [res2,formula]
print res2.summary()



# load test data
testfname = '%s/test.csv' % fdir
test_df = pd.read_csv(testfname,header=0)
test_df['Survived'] = 1.0
ids = test_df['PassengerId'].values
sex = test_df['Sex'].values

# fill in all the N/A's
# fill the null values with the most common place
if sum(test_df.Embarked.isnull()) > 0:
    common = test_df.Embarked.dropna().mode().values
    test_df.Embarked[test_df.Embarked.isnull()] = common
    print common

if sum(test_df.Age.isnull()) > 0:
    common = test_df.Age.dropna().median()
    test_df.Age[test_df.Age.isnull()] = common
    print common
#print len(test_df)
#test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
#test_df['AgeClass'] = test_df['Age']*test_df['Pclass']

compareresult1 = ka.predict(test_df[test_df.Sex == 'female'],results,'Logit_female')
tmp1 = np.zeros(compareresult1.shape)
tmp1[compareresult1 >= 0.5] = 1
compareresult2 = ka.predict(test_df[test_df.Sex == 'male'],results,'Logit_male')
tmp2 = np.zeros(compareresult2.shape)
tmp2[compareresult2 >= 0.5] = 1


output = np.zeros(ids.shape)
output[sex == 'female'] = tmp1
output[sex == 'male'] = tmp2

output = output.astype(int)

#outfname = '%s/logitoutput.csv' % fdir
outfname = '%s/logitoutput3.csv' % fdir
theoutputfile = open(outfname, "wb")
open_file_object = csv.writer(theoutputfile)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
theoutputfile.close()

