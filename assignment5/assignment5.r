# Intro to Data Science: assignment #5

# question 1 and 2
data = read.csv('seaflow_21min.csv')
summary(data)
pop = data$pop
oh = pop[pop == 'synecho']
print(length(oh))

# question 3
library(caret)
set.seed(125903)
indx <- createDataPartition(data$pop,p=0.5,list=FALSE,times=1)
data_train <- data[indx,]
data_test <- data[-indx,]

# print out the mean of the variable 'time' in the training set
#print(data_train$time)
print(mean(data_train$time))

# question 4: plot the data
#library(ggplot2)
#xx <- data_train$pe
#yy <- data_train$chl_small
#cc <- data_train$pop
#print(length(xx))
#print(length(yy))
#print(length(cc))
#p <- ggplot(data=data_train,aes(x = pe, y = chl_small, colour=pop)) + geom_point()
#print(p)

# question 5,6,7: train a decision tree
library(rpart)
fol <- formula(pop  ~ fsc_small + fsc_perp + fsc_big + pe + chl_big + chl_small)
model_cs <- rpart(fol,method="class",data=data_train)  #method = 'class' is for classification tree, 'anova' for regression tree
#print(model)
#plot(model)
#text(model)

#question 8: evaluate the decision tree on the test data
truth = data_test$pop
#df_test = data.frame(fsc_small=data_test$fsc_small,fsc_perp=data_test$fsc_perp,fsc_big=data_test$fsc_big,pe=data_test$pe,chl_big=data_test$chl_big,chl_small=data_test$chl_small)
predc1 = predict(model_cs,newdata=data_test,type='class') # this predict for the 'test data'
#predc = predict(model,data=data_test,type='class') # this predict for the 'trained data'
accuracy1 = sum(predc1 == truth)/length(truth)
print(accuracy1)

# question 9: random forest model
library(randomForest)
model_rf <- randomForest(fol,data=data_train,ntree=20)
predc2 = predict(model_rf,newdata=data_test)
accuracy2 = sum(predc2 == truth)/length(truth)
print(accuracy2)

# question 10: svm
#library(e1071)
#model_svm <- svm(fol,data=data_train)
#predc3 = predict(model_svm,newdata=data_test)
#accuracy3 = sum(predc3 == truth)/length(truth)
#print(accuracy3)

# question 14: remove the data with file_id 208
data2 = data[data$file_id != 208,]
set.seed(125903)
indx2 <- createDataPartition(data2$pop,p=0.5,list=FALSE,times=1)
data_train2 <- data2[indx,]
data_test2 <- data2[-indx,]
model_svm2 <- svm(fol,data=data_train2)
predc4 = predict(model_svm2,newdata=data_test2)
truth2 = data_test2$pop
accuracy4 = sum(predc4 == truth2)/length(truth2)
print(accuracy4)


