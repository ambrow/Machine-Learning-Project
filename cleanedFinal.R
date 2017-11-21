###Checking things for ML
#Load training and validation data
#Load RF
#Load XGB
#load("allstate_t.RData")
#load("allstate_v.RData")
#load("allstateRF.RData")
#load("xgBoost.RData")

library(readr)
library(randomForest)
library(Matrix)
library(xgboost)
library(tidyverse)

#reading in the data
allstate=read_csv("~/NC State Files/Fall/Machine Learning/Project/allstate_train.csv")

#setting categorical variables as factor variables
for(i in 2:117){
  allstate[[i]]=as.factor(allstate[[i]])
}

#selecting variables from SAS EM random forest
allstate=select(allstate,c('id','cat80',
                                  'cont2',
                                  'cat79',    
                                  'cat81',
                                  'cat100',
                                  'cat101',
                                  'cont7',
                                  'cont12',
                                  'cat111',
                                  'cat12',
                                  'cat1',    
                                  'cat114',
                                  'cat103',    
                                  'cat53',
                                  'cat87',    
                                  'cat94',
                                  'cont14',
                                  'cat4',
                                  'cat38',
                                  'cat57',
                                  'cat108',
                                  'cont11',
                                  'cat72',
                                  'cat2',    
                                  'cat5','loss'))

#separating into training and validation sets
set.seed(478)
split=sample(c(T,F),prob = c(70,30),replace = T,size=length(allstate$loss))
allstate.train=allstate[split==T,]
allstate.validation=allstate[split==F,]

#transform loss into target
allstate.train$target=log(allstate.train$loss)
allstate.validation$target=log(allstate.validation$loss)

#xgboost model
#first create sparse matrices
sparse_allstatetrain = sparse.model.matrix(target ~ . -id -loss -target, data=allstate.train)
sparse_allstatevalid = sparse.model.matrix(target ~ . -id -loss -target, data=allstate.validation)

xgb <- xgboost(data = sparse_allstatetrain,
               label = allstate.train$target,
               eta = 0.02,
               max_depth = 3,
               gamma = 6,
               nround=10000,
               subsample = 0.8,
               colsample_bytree = 0.8,
               objective = "reg:linear",
               nthread = 3,
               eval_metric = 'mae',
               verbose = 1)

#make predictions with xgboost
xgb.pvalid = predict(xgb, sparse_allstatevalid)
#exponentiate to make comparisons to actual data on loss
xgb.pvalid.actual=exp(xgb.pvalid)
#create MAE.XGB
mae.xgb=sum(abs(allstate.validation$loss - xgb.pvalid.actual))/length(xgb.pvalid.actual)

#random forest model
#first we create a design matrix
#cat57 (column 21) is dropped because it only has one level in the test set
allstate.cat.train=as.data.frame(allstate.train[,c(2,4:7,10:17,19:20,22,24:26)])
fix.train=as.data.frame(model.matrix(~.-1,data = allstate.cat.train))
#then we add the range standardized continuous variables
fix.train=as.data.frame(cbind(fix.train,allstate.train[,c(3,8,9,18,23)]))
for(i in 115:119){
  fix.train[,i]=(fix.train[,i]-min(fix.train[,i]))/(max(fix.train[,i])-min(fix.train[,i]))
}

#now do the pca on the covariance matrix
pca=prcomp(fix.train,scale=F)
plot(pca)
PC.set.t=pca$x[,c(1:10)]

#add the target variable to the dataset
PC.set.t=as.data.frame(cbind(PC.set.t,allstate.train$target))

#random forest
rf=randomForest(V11~.,data=PC.set.t,ntree=50, do.trace=TRUE)

#creating correct validation variables from validation set
xlevs.valid = lapply(allstate.cat.train[,sapply(allstate.cat.train, is.factor), drop = F], function(j){
  levels(j)
})
allstate.validation.cat=as.data.frame(allstate.validation[,c(2,4:7,10:17,19:20,22,24:26)])
fix.validation <- as.data.frame(model.matrix(~ . -1, data = allstate.validation.cat, xlev = xlevs.valid))
fix.validation=as.data.frame(cbind(fix.validation,allstate.validation[,c(3,8,9,18,23)]))
for(i in 115:119){
  fix.validation[,i]=(fix.validation[,i]-min(fix.validation[,i]))/(max(fix.validation[,i])-min(fix.validation[,i]))
}

#matrix multiplication to score the validation data with the training pca's
rf.v=scale(fix.validation,pca$center,pca$scale) %*% pca$rotation
rf.v.2=as.data.frame(rf.v[,1:10])
rf.pred=predict(rf,rf.v.2,type='response')
rf.actual=exp(rf.pred)

#rf mae
rf.mae=sum(abs(allstate.validation$loss-rf.actual))/length(allstate.validation$loss)

#ensemble
ensemble.prediction=(rf.actual+xgb.pvalid.actual)/2

#ensemble mae
ensemble.mae=sum(abs(allstate.validation$loss-ensemble.prediction))/length(allstate.validation$loss)




#######################
#recreating model on all data
#loading the test data
allstate.test=read_csv("~/NC State Files/Fall/Machine Learning/Project/allstate_test.csv")

for(i in 2:117){
  allstate.test[[i]]=as.factor(allstate.test[[i]])
}

allstate.test=select(allstate.test,c('id','cat80',
                             'cont2',
                             'cat79',    
                             'cat81',
                             'cat100',
                             'cat101',
                             'cont7',
                             'cont12',
                             'cat111',
                             'cat12',
                             'cat1',    
                             'cat114',
                             'cat103',    
                             'cat53',
                             'cat87',    
                             'cat94',
                             'cont14',
                             'cat4',
                             'cat38',
                             'cat57',
                             'cat108',
                             'cont11',
                             'cat72',
                             'cat2',    
                             'cat5'))
#sparse allstate test
#not sure I need this
#allstate.test$target=numeric(length = length(allstate.test$id))
allstate_sparse.test= sparse.model.matrix(~.-id,data=allstate.test)
#^throwing an error bc cat57 only has one level
allstate.test.xgb=allstate.test[,c(1:20,22:26)]
allstate_sparse.test=sparse.model.matrix(~.-id,data=allstate.test.xgb)
#so this one works, so I will be dropping cat57 in the final model
#log transform target
allstate$target = log(allstate$loss)
allstate.xgb=allstate[,c(1:20,22:28)]
#XGB
#sparse matrix
allstate_sparse = sparse.model.matrix(target ~ . -id -loss -target, data=allstate.xgb)

xgb.final <- xgboost(data = allstate_sparse,
               label = allstate$target,
               eta = 0.02,
               max_depth = 3,
               gamma = 6,
               nround=10000,
               subsample = 0.8,
               colsample_bytree = 0.8,
               objective = "reg:linear",
               nthread = 3,
               eval_metric = 'mae',
               verbose = 1)

#xgboost predictions
xgb.predictions.transformed=predict(xgb.final,allstate_sparse.test)
xgb.predictions=exp(xgb.predictions.transformed)

#write xgb predictions to csv
write_csv(data.frame(cbind(allstate.test$id,xgb.predictions)),path="xgb.predictions.csv")

#random forest
allstate.cat=as.data.frame(allstate[,c(2,4:7,10:17,19:20,22,24:26)])
fix.final=as.data.frame(model.matrix(~.-1,data = allstate.cat))
fix.final=as.data.frame(cbind(fix.race,allstate[,c(3,8,9,18,23)]))
for(i in 115:119){
  fix.final[,i]=(fix.final[,i]-min(fix.final[,i]))/(max(fix.final[,i])-min(fix.final[,i]))
}

#now do the pca on the covariance matrix
pca=prcomp(fix.final,scale=F)
plot(pca)
PC.set.final=pca$x[,c(1:10)]

#add the target variable to the dataset
PC.set.final=as.data.frame(cbind(PC.set,allstate$target))

#random forest
rf.final=randomForest(V11~.,data=PC.set.final,ntree=50, do.trace=TRUE)

#creating correct validation variables from test set
#subset into categorical and continuous
#run xlevs for categorical
#cbind standardized continous
xlevs = lapply(allstate.cat[,sapply(allstate.cat, is.factor), drop = F], function(j){
  levels(j)
})
allstate.test$loss=1:100
allstate.test.cat=as.data.frame(allstate.test[,c(2,4:7,10:17,19:20,22,24:26)])
fix.test <- as.data.frame(model.matrix(~ . -1, data = allstate.test.cat, xlev = xlevs))
fix.test=as.data.frame(cbind(fix.test,allstate.test[,c(3,8,9,18,23)]))
for(i in 115:119){
  fix.test[,i]=(fix.test[,i]-min(fix.test[,i]))/(max(fix.test[,i])-min(fix.test[,i]))
}


#matrix multiplication to score the validation data with the training pca's
rf.test=scale(fix.test,pca$center,pca$scale) %*% pca$rotation
rf.test.2=as.data.frame(rf.test[,1:10])
rf.predictions.transformed=predict(rf.final,rf.test.2,type='response')
rf.predictions=exp(rf.predictions.transformed)

#write to csv the random forest predictions
write_csv(data.frame(cbind(allstate.test$id,rf.predictions)),path="rf.predictions.csv")


#ensemble predictions
ensemble.predictions.final = (xgb.predictions+rf.predictions)/2

#write to csv
write_csv(data.frame(cbind(allstate.test$id,ensemble.predictions.final)),path="ensemble.predictions.csv")
