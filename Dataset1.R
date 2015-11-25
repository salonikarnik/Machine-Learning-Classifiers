install.packages("rpart",dependencies = TRUE)
install.packages("e1071", dependencies=TRUE)
install.packages("klaR")
install.packages("caret",dependencies = TRUE)
install.packages("class",dependencies = TRUE)
install.packages("neuralnet",dependencies=T)
install.packages("adabag",dependencies = T)
install.packages("ipred")
install.packages("ada",dependencies = T)
install.packages("randomForest")

library("rpart")
library ("e1071")
library("klaR")
library("caret")
library("class")
library("neuralnet")
require(adabag)
library("randomForest")
library("ipred")
library(ada)


args <- commandArgs(TRUE)
dataURL<-as.character(args[1])
header<-as.logical(args[2])
d<-read.csv(dataURL,header = header)

set.seed(123)


for(i in 1:10) {
  cat("Running sample ",i,"\n")
  sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
  trainingData<-d[sampleInstances,]
  testData<-d[-sampleInstances,]
  Class<-d[,as.integer(args[3])]
  
  method="Decision Tree" 
  library("rpart")
  model <- rpart(trainingData[Class]~., data=trainingData, parms = list(split = 'information'),minsplit=2,method="class")
  prunedTree <- prune(model, cp=(model$cptable[which.min(model$cptable[, "xerror"]), "CP"]))
  pred <- predict(prunedTree,testData) 
  predTable <- table(pred,testData[Class])
  accuracy <- sum(diag(predTable))/sum(predTable)
  accuracy*100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  method="Support Vector Machines" 
  library ("e1071")
  svmModel <- svm(trainingData[Class]~., data = trainingData,cost = 100,gamma=1000) 
  SVMprediction <- predict(svmModel, testDdata[,-6])
  conf <- table(pred = SVMprediction, true = testData[,6])
  accuracy <-sum(diag(conf))/sum(conf)
  accuracy <-accuracy*100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  method="Naive Bayesian"
  library("klaR")
  library("caret")
  model <- naiveBayes(as.factor(trainingData[Class]) ~ ., data = trainingData) 
  model
  prediction <- predict(model, testData)
  prediction
  tab <- table(prediction, testData[Class])            
  tab
  accuracy <- sum(diag(tab))/sum(tab)
  accuracy <-accuracy*100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  method="kNN"
  library("class")
  pred <- knn(train=trainingData, test=testData,cl=as.factor(trainingData[Class]), k=7,prob = FALSE, use.all = TRUE)
  conf <- confusionMatrix(pred, testData[Class])
  accuracy <- conf$overall[1]*100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  method="Neural Net"
  library("neuralnet")
  netModel <- neuralnet(trainingData[Class] ~income + loan + age, data= trainingData ,hidden = 4, lifesign = "minimal",linear.output = FALSE, threshold = 0.1)
  temp_test <- subset(test_data, select = c("income", "loan","age"))
  pred <- compute(netModel, temp_test)
  result <- data.frame(actual = testData[Class], prediction = pred$net.result)
  result$prediction <- round(result$prediction)
  tab <- table(actual= result$actual,prediction=result$prediction)
  accuracy <- sum(tab[0,0],tab[1,1])/sum(tab)
  accuracy <-accuracy*100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  method="Logistic Regression"
  library("class")
  Logmodel <- glm(trainingData[Class]~.,data=trainingData,family = binomial)
  pred <- predict(Logmodel,testData,type="response")
  confint(Logmodel)
  threshold=0.65
  prediction<-sapply(pred, FUN=function(x) if (x>threshold) 1 else 0)
  actual<-trainingData
  accuracy <- sum(actual==prediction)/length(actual)
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  method="Bagging"
  library("ipred")
  model <- bagging(trainingData[Class]~., data=trainingData, coob=TRUE)
  pred <- predict(model, testData)
  result <- data.frame(actual = testData[Class], prediction = pred)
  result$prediction <- round(result$prediction)
  tab <- table(actual= result$actual,prediction=result$prediction)
  accuracy <- sum(diag(tab))/sum(tab)
  accuracy <- accuracy*100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  method="Random Forest"
  library("randomForest") 
  rfModel <- randomForest(as.factor(trainingData[Class])~., data=trainingData, importance=TRUE, proximity=TRUE, ntree=500)
  RFpred <- predict(rfModel,testData,type='response')
  predTable <- table(observed = testData[Class], predicted = RFpred)
  accuracy <- sum(diag(predTable))/sum(predTable)
  accuracy <-accuracy*100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  method="Boosting"
  library("ada")
  model <- ada(trainingData[Class] ~ ., data = trainingData, iter=20, nu=1, type="discrete")
  p=predict(model,testData)
  accuracy <- sum(testData[Class]==p)/length(p)
  accuracy <- accuracy * 100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
}