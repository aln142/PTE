rm(list=ls())
setwd("/Users/aln142/Library/Mobile Documents/com~apple~CloudDocs/R Files/PTE")

library(ggplot2)
library(randomForest)
library(dbplyr)
library(tidyverse)
library(AUCRF)
library(pROC)
library(glmnet)
library(e1071)
library(caret)
library(gbm)

set.seed(3)
TBI <- read.csv("PTE R Subset.csv")
TBI2 <- TBI
TBI2$Epilepsy <- factor(TBI2$Epilepsy)
TBI2$AIS <- factor(TBI2$AIS)
set.seed(3)
train2<-sample(1:nrow(TBI2),241)
datTrain<-TBI2[train2,]
datTest<-TBI2[-train2,]


#stepwise Logistic Regression
glm4<-glm(Epilepsy ~ 1, family = "binomial", data=datTrain)
glm5<-glm(Epilepsy ~ AIS + sz_early + Age.at.TBI + Sex + mva + non_accident + pmh_dev_delay + pmh_fam_hist_epi + pmh_adhd, family = "binomial", data=datTrain)
step_model <- step((glm(Epilepsy ~ AIS + Age.at.TBI + Sex + sz_early + mva + non_accident + pmh_dev_delay + pmh_fam_hist_epi + pmh_adhd, family = "binomial", data=datTrain)), scope = list(lower = glm4, upper = glm5), direction = "backward")
step_model_pred <-predict(step_model,newdata=datTest,type="response")
step_model_pred<-ifelse(step_model_pred>=0.5,1,0)
table(step_model_pred,datTest$Epilepsy)
mean(abs(step_model_pred-datTest$Epilepsy))
mean2 <-1-err2
mean2
error <- qt(0.975,df=length(step_model_pred==datTest$Epilepsy)-1)*sd(step_model_pred==datTest$Epilepsy)/sqrt(length(step_model_pred==datTest$Epilepsy))
left <- mean(step_model_pred==datTest$Epilepsy)-error
right <- mean(step_model_pred==datTest$Epilepsy)+error
left
right
step_model_pred <- as.numeric(step_model_pred)
ci.auc(datTest$Epilepsy, step_model_pred, boot.n=2000)
auc(datTest$Epilepsy, step_model_pred)
ci.thresholds(datTest$Epilepsy, step_model_pred, boot.n=2000)

#Support Vector Machine
tune.out <- tune(svm, Epilepsy~., data=datTrain, kernel="radial", scale=FALSE, ranges=list(cost=c(1,5,10,15,20),gamma=(2^c(-8:4))))
tune.out
summary(tune.out)
print(tune.out)

svmFit2 <- svm(Epilepsy ~ ., data=datTrain, kernel="radial", cost=20, gamma=0.125, scale=FALSE)
predsvm<-predict(svmFit2,datTest, type="class")
table(predsvm,datTest$Epilepsy)
mean(predsvm==datTest$Epilepsy)
error <- qt(0.975,df=length(predsvm==datTest$Epilepsy)-1)*sd(predsvm==datTest$Epilepsy)/sqrt(length(predsvm==datTest$Epilepsy))
left <- mean(predsvm==datTest$Epilepsy)-error
right <- mean(predsvm==datTest$Epilepsy)+error
left
right

predsvm <- as.numeric(predsvm)
datTest$Epilepsy<- as.factor(datTest$Epilepsy)
ci.auc(datTest$Epilepsy, predsvm, boot.n=10000)
1-auc(as.numeric(datTest$Epilepsy), as.numeric(predsvm))
ci.thresholds(datTest$Epilepsy, predsvm, boot.n=10000)


#Random Forest
TBI <- read.csv("PTE R Subset.csv")
TBI2 <- TBI 
TBI2$Epilepsy <- factor(TBI2$Epilepsy)
TBI2$AIS <- as.factor(TBI2$AIS)
set.seed(3)
train2<-sample(1:nrow(TBI2),172) #50% for training
datTrain2<-TBI2[train2,]
dat2<-TBI2[-train2,]
validate2<-sample(1:nrow(dat2),69) #20% for validation
datValidate2<-dat2[validate2,]
datTest2<-dat2[-validate2,] #30% for testing

rfTBI2 <-randomForest(Epilepsy ~ ., data = datTrain2, method = 'class', ntree=1000)
rfTBI2
varImpPlot(rfTBI2, sort=TRUE, n.var=min(20, nrow(rfTBI2$importance)))
varImpPlot(rfTBI2)

oobErr <- testErr <- double(length=15)
for(indx in 1:length(oobErr)){
  tempFit <- randomForest(Epilepsy ~ ., data=datTrain2, mtry=indx, ntree=1000)
  oobErr[indx] <- tempFit$err.rate[1000]
  tempPred <- predict(tempFit,datValidate2)
  testErr[indx] <- mean(tempPred!=datValidate2$Epilepsy)
  cat(indx," ")
}
matplot(1:indx,cbind(testErr,oobErr),pch=19,col=c("red","blue"),type="b",ylab="Classification Error",xlab="Size of Predictor Sets")
legend("topright",legend=c("OOB","Test"),pch=19,col=c("red","blue"))

predvalRF <- predict(rfTBI2,datTest2,type="response", mtry=3)
table(predvalRF,datTest2$Epilepsy)
mean(predvalRF==datTest2$Epilepsy)
error <- qt(0.975,df=length(predvalRF==datTest2$Epilepsy)-1)*sd(predvalRF==datTest2$Epilepsy)/sqrt(length(predvalRF==datTest2$Epilepsy))
left <- mean(predvalRF==datTest2$Epilepsy)-error
right <- mean(predvalRF==datTest2$Epilepsy)+error
left
right
predvalRF <- as.numeric(predvalRF) 
auc(datTest2$Epilepsy, predvalRF)
ci.auc(datTest2$Epilepsy, predvalRF, boot.n=2000)
ci.thresholds(datTest2$Epilepsy, predvalRF, boot.n=2000)


#Adaptive Boosting

fitgbm <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 10, 
                       search = "random")

gbmGrid <-  expand.grid(interaction.depth = c(1, 3, 6, 9, 10), 
                        n.trees = seq(100,10000,100), 
                        shrinkage = seq(.05, .1,.05),
                        n.minobsinnode = c(5,10,15,20))

gbm_fit <- train(Epilepsy ~ ., data = datTrain, 
                 distribution="adaboost",
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = fitgbm)
gbm_fit

predmat2 <- predict(gbm_ft, newdata=datTest, type="response", n.trees =, shrinkage =, interaction.depth =, n.minobsinnode = )

### EDIT HERE
predmat2_values <-ifelse(predmat2>=0.5,1,0)
table(predmat2_values, datTest3$Epilepsy)
err2 <- mean(abs(predmat2_values-datTest3$Epilepsy))
mean2 <-1-err2
mean2
error <- qt(0.975,df=length(predmat2_values==datTest3$Epilepsy)-1)*sd(predmat2_values==datTest3$Epilepsy)/sqrt(length(predmat2_values==datTest3$Epilepsy))
left <- mean(predmat2_values==datTest3$Epilepsy)-error
right <- mean(predmat2_values==datTest3$Epilepsy)+error
left
right

predmat2 <- as.numeric(predmat2)
ci.auc(datTest3$Epilepsy, predmat2, boot.n=2000)
auc(datTest3$Epilepsy, predmat2)
ci.thresholds(datTest3$Epilepsy, predmat2, boot.n=2000)
ci.thresholds(datTest3$Epilepsy, predmat2, threshold=0.5, boot.n=2000)

1-auc((as.numeric(datTest3$Epilepsy)), as.numeric(predmat2))
ci.auc(datTest2$Epilepsy, predmat2)
ci.thresholds(datTest2$Epilepsy, predmat2, threshold=00.50)

#Supplemental Figure 1.
prediction_for_GLM <- predict(glm5, x_test, type="response") 
pred_GLM <- prediction(prediction_for_GLM, datTest$Epilepsy)
perf_GLM <-performance(pred_GLM, "tpr", "fpr")
plot(perf_GLM, col='1', label = "Logistic", xlab="1-Specifity", ylab="Sensitivity")

prediction_for_BOOST <- predict(gbDemAdaboost, datTest)
prediction_for_BOOST <- as.data.frame(prediction_for_BOOST)
pred_BOOST <- prediction(prediction_for_BOOST[,1], datTest3$Epilepsy)
perf_BOOST <-performance(pred_BOOST, "tpr", "fpr")
plot(perf_BOOST, col='2', add=TRUE, label = "ADABOOST")

pred_RF <- predict(rfTBI2,datTest2,type="prob")
pred_RF <- prediction(pred_RF[,2], datTest2$Epilepsy)
perf_RF <-performance(pred_RF, "tpr", "fpr")
plot(perf_RF, col='3', add=TRUE, label = "Random Forest")
legend(0.6, 0.5, c('Stepwise Logistic Regression','Random Forest', 'Adaptive Boosting'),1:3, box.lty=0)


