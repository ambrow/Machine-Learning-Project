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
#all categorical variables become binary flags for each category

design.t=as.data.frame(model.matrix(~allstate.train[[2]]-1))
for(i in c(4:7,10:17,19:22,24:26)){
  column.t=as.data.frame(model.matrix(~allstate.train[[i]]-1))
  design.t = as.data.frame(cbind(design.t,column.t))
}
design.t=as.data.frame(cbind(design.t,allstate.train[,c(3,8,9,18,23)]))
#all numeric variables will be range standardized
design.t[,135]=(design.t[,135]-min(design.t[,135]))/(max(design.t[,135])-min(design.t[,135]))
#summary(design[,135]) check my range standardization
for(i in 136:139){
  design.t[,i]=(design.t[,i]-min(design.t[,i]))/(max(design.t[,i])-min(design.t[,i]))
}
#now do the pca on the covariance matrix
pca=prcomp(design.t,scale=F)
plot(pca)
PC.set.t=pca$x[,c(1:10)]

#add the target variable to the dataset
PC.set.t=as.data.frame(cbind(PC.set.t,allstate.train$target))

#random forest
rf=randomForest(allstate.train$target~.,data=PC.set.t,ntree=200, do.trace=TRUE)

#creating correct validation variables from validation set
design.v=as.data.frame(model.matrix(~allstate.validation[[2]]-1))
for(i in c(4:7,10:17,19:22,24:26)){
  column.v=as.data.frame(model.matrix(~allstate.validation[[i]]-1))
  design.v = as.data.frame(cbind(design.v,column.v))
}
design.v=as.data.frame(cbind(design.v,allstate.validation[,c(3,8,9,18,23)]))
design.v[,135]=(design.v[,135]-min(design.v[,135]))/(max(design.v[,135])-min(design.v[,135]))
#summary(design[,135]) check my range standardization
for(i in 136:139){
  design.v[,i]=(design.v[,i]-min(design.v[,i]))/(max(design.v[,i])-min(design.v[,i]))
}
#matrix multiplication to score the validation data with the training pca's
rf.v=scale(design.v,pca$center,pca$scale) %*% pca$rotation
rf.v.2=as.data.frame(rf.v[,1:10])
rf.pred=predict(rf,rf.v.2,type='response')
rf.actual=exp(rf.pred)

#rf mae
rf.mae=sum(abs(allstate.validation$loss-rf.actual))/length(allstate.validation$loss)

#ensemble
ensemble.prediction=(rf.actual+xgb.pvalid.actual)/2

#ensemble mae
ensemble.mae=sum(abs(allstate.validation$loss-ensemble.prediction))/length(allstate.validation$loss)

#recreating model on all data
#loading test data
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

#random forest
design=as.data.frame(model.matrix(~allstate[[2]]-1))
for(i in c(4:7,10:17,19:20,22,24:26)){
  column=as.data.frame(model.matrix(~allstate[[i]]-1))
  design = as.data.frame(cbind(design,column))
}
design=as.data.frame(cbind(design,allstate[,c(3,8,9,18,23)]))
#all numeric variables will be range standardized
design[,133]=(design[,133]-min(design[,133]))/(max(design[,133])-min(design[,133]))
#summary(design[,135]) check my range standardization
for(i in 133:137){
  design[,i]=(design[,i]-min(design[,i]))/(max(design[,i])-min(design[,i]))
}
#now do the pca on the covariance matrix
pca=prcomp(design,scale=F)
plot(pca)
PC.set=pca$x[,c(1:10)]

#add the target variable to the dataset
PC.set=as.data.frame(cbind(PC.set,allstate$target))

#random forest
rf.final=randomForest(V11~.,data=PC.set,ntree=200, do.trace=TRUE)

#creating correct validation variables from validation set
design.test=as.data.frame(model.matrix(~allstate.test[[2]]-1))
for(i in c(4:7,10:17,19:20,22,24:26)){
  column.test=as.data.frame(model.matrix(~allstate.test[[i]]-1))
  design.test = as.data.frame(cbind(design.test,column.test))
}
design.test=as.data.frame(cbind(design.test,allstate.test[,c(3,8,9,18,23)]))
#this looks like there aren't as many levels in the test set. Interesting
#could pose a problem for the PCA solution here. I shall ask
design.test[,135]=(design.test[,135]-min(design.test[,135]))/(max(design.test[,135])-min(design.test[,135]))
#summary(design[,135]) check my range standardization
for(i in 136:139){
  design.test[,i]=(design.test[,i]-min(design.test[,i]))/(max(design.test[,i])-min(design.test[,i]))
}
#matrix multiplication to score the validation data with the training pca's
rf.test=scale(design.test,pca$center,pca$scale) %*% pca$rotation
rf.test.2=as.data.frame(rf.test[,1:10])
rf.predictions.transformed=predict(rf.final,rf.test.2,type='response')
rf.predictions=exp(rf.predictions.transformed)

#ensemble predictions
ensemble.predictions.final = (xgb.predictions+rf.predictions)/2

#write to csv
csv.predictions=cbind(allstate.test$id,ensemble.predictions.final)
