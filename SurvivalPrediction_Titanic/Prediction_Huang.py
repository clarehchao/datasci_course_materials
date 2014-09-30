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
#print df.head(10)
#print df.dtypes  # print out the data type of each features in the data frame
#print df.info()  # tells how many non-null values are in each feature
#print df.shape  # size of the data frame
#print df['Age'][0:10]
#print df.Age[0:10]
#print df[['Sex','Pclass','Age']]
#print df[df['Age'] > 60][['Sex','Pclass','Age']]

# plot the histogram of 'age'
#df['Age'].hist()
#P.show()

# add another column called 'Gender'
#df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
#print df['Gender']

# or convert sex into binary 0 or 1
#df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)
#print df.Gender

#data exploration
# gender vs. survival
#fig = plt.figure()
#plt.scatter(df.Survived,df.Age)

# get some idea of what kind of features are related to survive or not survive
#fig = plt.figure()
#alpha_bar_chart = 0.55
#df[df.Survived==1].Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

#fig = plt.figure()
#df[df.Survived==0].Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
#plt.show()
# no clear distinction between Embarked destination vs. survival rate

"""
fig = plt.figure()
df.Survived[df.Pclass == 3].value_counts().plot(kind='barh',label='class 3',alpha=0.55)
print df.Survived[df.Pclass == 2].value_counts()
df.Survived[df.Pclass == 2].value_counts().plot(kind='barh', color='#FA2379',label='class 2',alpha=0.55)
df.Survived[df.Pclass == 1].value_counts().plot(kind='barh', color='green',label='class 1',alpha=0.55)
plt.ylim(-1,2)
plt.title("Who Survived? with respect to Pclass, (raw value counts) "); plt.legend(loc='best')
plt.show()
# clearly, more ppl in class 3 (the lower class) survived ==> why? maybe where they were located? it'd be good to correlate the cabin location to survival rate? but i guess i could be other factors too

fig = plt.figure()
df.Survived[df.SibSp > 1].value_counts().plot(kind='barh',label='more than 1 sib',alpha=0.55)
df.Survived[df.SibSp <= 1].value_counts().plot(kind='barh', color='#FA2379',label='less than 1 sib',alpha=0.55)
plt.ylim(-1,2)
plt.title("Who Survived? with respect to SibSp, (raw value counts) "); plt.legend(loc='best')
plt.show()
# no clear observation..

fig = plt.figure()
df.Survived[df.Parch > 1].value_counts().plot(kind='barh',label='more than 1 parent/child',alpha=0.55)
df.Survived[df.Parch <= 1].value_counts().plot(kind='barh', color='#FA2379',label='less than 1 parent/child',alpha=0.55)
plt.ylim(-1,2)
plt.title("Who Survived? with respect to Parch, (raw value counts) "); plt.legend(loc='best')
plt.show()
# less ppl with less than 1 parent/child would survive


# plot density distribution
df.Fare[df.Pclass == 1].plot(kind='kde')    
df.Fare[df.Pclass == 2].plot(kind='kde')
df.Fare[df.Pclass == 3].plot(kind='kde')
plt.xlabel("Fare")    
plt.title("Fare Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
plt.show()
"""

# interesting to explore: cabin info and its location, SibSp and Parch aren't as accurate to describe relationships, take a look at names and see how they're related for survival prediction, if one is related and survive, the relative could be more likely to survive, add as another dependent variable or a weighting factor when training!

# drop all the null data
logittrain_df = train_df.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
logittrain_df  = logittrain_df.dropna()

# logistic regression
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
results = {}

y,x  = dmatrices(formula,data=logittrain_df,return_type='dataframe')
model = sm.Logit(y,x)  # create a model
res = model.fit()  # fit the model to the training data
results['Logit'] = [res,formula]
print res.summary()

# examine predicted vs. actual values
# Plot Predictions Vs Actual
"""fig = plt.figure()
ypred = res.predict(x)
plt.plot(x.index,ypred,'ro',x.index,y,'bo',alpha=0.4)
plt.grid(color='white',linestyle='dashed')

# plot the survival distribution
fig = plt.figure()
kde_res = KDEUnivariate(res.predict())
kde_res.fit()
plt.plot(kde_res.support,kde_res.density)
plt.fill_between(kde_res.support,kde_res.density, alpha=0.5)
plt.title("Distribution of our Predictions")

fig = plt.figure()
plt.plot(res.predict(),x['C(Sex)[T.male]'],'ro',alpha=0.5)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted chance of survival")
plt.ylabel("Gender Bool")
plt.title("The Change of Survival Probability by Gender (1 = Male)")


fig = plt.figure()
plt.scatter(res.predict(),x['C(Pclass)[T.3]'] , alpha=0.5)
plt.xlabel("Predicted chance of survival")
plt.ylabel("Class Bool")
plt.grid(b=True, which='major', axis='x')
plt.title("The Change of Survival Probability by Lower Class (1 = 3rd Class)")

fig = plt.figure()
plt.scatter(res.predict(),x['Parch'] , alpha=0.5)
plt.xlabel("Predicted chance of survival")
plt.ylabel("Class Bool")
plt.grid(b=True, which='major', axis='x')
plt.title("The Change of Survival Probability by # of Parent/child")
plt.show()
"""

# load test data
testfname = '%s/test.csv' % fdir
test_df = pd.read_csv(testfname,header=0)
test_df['Survived'] = 1.0
compareresult = ka.predict(test_df,results,'Logit')
compareresult = Series(compareresult)
outfname = '%s/logitoutput.csv' % fdir
compareresult.to_csv(outfname)
print compareresult

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
#rftrain_df['Familysize'] = rftrain_df['SibSp'] + rftrain_df['Parch']
#rftrain_df['AgeClass'] = rftrain_df['Age']*rftrain_df['Pclass']

# drop the unused columns
# when using dmatrices, one doesn't need to drop the unnecessary columns since it will pick the needed data out
#rftrain_df = rftrain_df.drop(['Name','Ticket', 'Cabin', 'PassengerId'], axis=1) 
formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
y,x = dmatrices(formula_ml, data = rftrain_df,return_type='dataframe')
y = np.asarray(y).ravel()  # random forest takes a 1D flatten array
result_rf = ske.RandomForestClassifier(n_estimators=100).fit(x,y)
score = result_rf.score(x,y)
#oh = 1 - sum(abs(y - result_rf.predict(x)))  #this is the same as score calculated above
print 'score is {0}'.format(score)

# make the prediction with the test data
# prepare the test data
rftest_df = test_df
# fill the null values with the most common place
if sum(rftest_df.Embarked.isnull()) > 0:
    common = rftest_df.Embarked.dropna().mode().values
    rftest_df.Embarked[rftest_df.Embarked.isnull()] = common
    print common

if sum(rftest_df.Age.isnull()) > 0:
    common = rftest_df.Age.dropna().median()
    rftest_df.Age[rftest_df.Age.isnull()] = common
    print common

#rftest_df['Familysize'] = rftest_df['SibSp'] + rftest_df['Parch']
#rftest_df['AgeClass'] = rftest_df['Age']*rftest_df['Pclass']

formula_test = 'PassengerId ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
junk,x = dmatrices(formula_test, data = rftest_df,return_type='dataframe')
test_x = np.asarray(x)  
output = result_rf.predict(test_x)
#print output

fig = plt.figure()
#plt.plot(x.index,result_rf.predict(x),'ro',x.index,y,'bo',alpha=0.5)
plt.plot(x.index,output,'bo',alpha=0.5)
plt.ylim(-1,2)
plt.show()

# the Age*Class and Familysize didn't really improve the score when training the data
# maybe another feature added to the models? i guess this is where data visualization is helpful...

# try support vector machine
# select the features for training
svmtrain_df = rftrain_df
y2,x2 = dmatrices(formula_ml, data = svmtrain_df,return_type='matrix')
feat1 = 3
feat2 = 6
print x2
X2 = np.asarray(x2)
X2 = X2[:,[feat1,feat2]]
print X2


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
    
    plt.figure(nfig)
    plt.scatter(X2[:,0],X2[:,1],c=y,zorder=10,cmap=color_map)
    

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
    
clf = svm.SVC(kernel='poly', gamma=3).fit(X_train, y_train)                                                            
y,x = dmatrices(formula_ml, data=test_data, return_type='dataframe')

# Change the interger values within x.ix[:,[6,3]].dropna() explore the relationships between other 
# features. the ints are column postions. ie. [6,3] 6th column and the third column are evaluated. 
res_svm = clf.predict(x.ix[:,[6,3]].dropna()) 

res_svm = DataFrame(res_svm,columns=['Survived'])
res_svm.to_csv("data/output/svm_poly_63_g10.csv") # saves the results for you, change the name as you please. 
    










