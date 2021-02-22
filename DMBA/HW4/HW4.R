set.seed(12345)
library(openxlsx)
library(FNN)
library(caret)
train <- read.xlsx("D://Homework//DMBA//HW4//user_changes.xlsx",1,colNames = T)
new <- read.xlsx("D://Homework//DMBA//HW4//new_knn.xlsx",1,colNames = T)
train.index <- sample(c(1:116974),10000)
train.rd <- train[train.index,]

train.rd$time_to_booking <- train.rd$datetime - train.rd$cdate
data <- train.rd[,c(1,6,12,14)]

train.index <- sample(c(1:10000),5000)
train <- data[train.index,]
valid <- data[-train.index,]
test.index <- sample(c(1:5000),2000)
test <- valid[test.index,]
valid <- valid[-test.index,]

train.norm <- train
valid.norm <- valid
test.norm <- test
new.norm <- new

norm.value <- preProcess(train[,c(2,4)],method = c("center","scale"),cutoff = 0.5, outcome = train[,3])
train.norm[,c(2,4)] <- predict(norm.value,train[,c(2,4)])
valid.norm[,c(2,4)] <- predict(norm.value,valid[,c(2,4)])
test.norm[,c(2,4)] <- predict(norm.value,test[,c(2,4)])
new.norm[,2:3] <- predict(norm.value,new[,2:3])

library(e1071)
# initialize a data frame with two columns: k, and accuracy.
accuracy.df <- data.frame(k = seq(1, 20, 1), accuracy = rep(0, 20))

# compute knn for different k on validation.
for(i in 1:20) {
  knn.pred <- knn(train.norm[, 1:2], test.norm[, 1:2], 
                  cl = train.norm[, 3], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, test.norm[, 3])$overall[1] 
}
accuracy.df


knn.pred.new <- knn(train.norm[, c(2,4)], new.norm[10,2:3], 
                    cl = train.norm[, 3], k = 20)
knn.pred.new


#NB
rm(list=ls())

train <- read.xlsx("D://Homework//DMBA//HW4//user_changes.xlsx",1,colNames = T)
set.seed(12345)

train$time_to_booking <- train$datetime - train$cdate
for( i in 1:nrow(SampleWS)){
  SampleWS$time_to_booking[i] <-train$time_to_booking[which(train$booking_id==SampleWS$booking_id[i])]
}
SampleWS$binned_people <-  ifelse(SampleWS$people < median(SampleWS$people), 1, 2)
SampleWS$binned_time_to_booking <-  ifelse(SampleWS$time_to_booking < median(SampleWS$time_to_booking), 1, 2)
data <- SampleWS[,c(1,8,10,6)]

data$binned_people<- factor(data$binned_people)
data$binned_time_to_booking<- factor(data$binned_time_to_booking)
data$cancel_change <- factor(data$cancel_change)

train.index <- sample(c(1:30000),15000)
train <- data[train.index,]
valid <- data[-train.index,]
test.index <- sample(c(1:15000),6000)
test <- valid[test.index,]
valid <- valid[-test.index,]

cancel.nb<- naiveBayes(cancel_change ~.,data = train,type = "raw")
cancel.nb

new <- read.xlsx("D://Homework//DMBA//HW4//new_NB.xlsx",1,colNames = T)
## predict probabilities
pred.prob <- predict(cancel.nb, newdata = new, type = "raw")
pred.prob
## predict class membership
pred.class <- predict(cancel.nb, newdata = valid)

df <- data.frame(actual = valid.df$Flight.Status, predicted = pred.class, pred.prob)
















