setwd("D:\\Homework\\DMBA\\HW7")
library(readxl)
library(dummies)
library(caret)
library(stepwise)
data <- read_xlsx("HubwayTripsByRegistered.xlsx",1)

set.seed(12345)
selected.var <- c(1,12,13,14,15)
s <- sample(sample(1:nrow(data), size=30000),replace = T)
hubway.sample <- data[s,selected.var]

aggregate(dummy(hubway.sample$trip_type),by=list(hubway.sample$hour),FUN=mean)
hubway.sample$morning <- hubway.sample$hour %in% c(4,5,6,7,8,9, 10,11)
hubway.sample$noon <- hubway.sample$hour %in% c(12, 13, 14)
hubway.sample$afternoon <- hubway.sample$hour %in% c(15, 16, 17, 18, 19)
hubway.sample$evening <- hubway.sample$hour %in% c(20, 21, 22,23,0,1,2,3)
hubway.sample$isReturnTrip<- 1*(hubway.sample$trip_type=='RoundTrip')

set.seed(12345)
train.index <- sample(c(1:dim(hubway.sample)[1]), dim(hubway.sample)[1]*0.6)  
valid.index <- setdiff(c(1:dim(hubway.sample)[1]), train.index)  
train.df <- hubway.sample[train.index, ]
valid.df <- hubway.sample[valid.index, ]


# Create dataset with dummies for inputs
hubway.sample$DOW <- as.factor(hubway.sample$DOW)
hubway.sample$gender <- as.factor(hubway.sample$gender)
hubway.sample$hour <- as.factor(hubway.sample$hour)
hubway.sample.dummies <- dummy.data.frame(hubway.sample, sep = ",")
hubway.sample.dummies <- subset(hubway.sample.dummies, select = -trip_type)

set.seed(12345)
train.index <- sample(c(1:dim(hubway.sample)[1]), dim(hubway.sample)[1]*0.6)  
valid.index <- setdiff(c(1:dim(hubway.sample)[1]), train.index)  
train.dummies.df <- hubway.sample.dummies[train.index, ]
valid.dummies.df <- hubway.sample.dummies[valid.index, ]
# do data partitioning again here and create train.dummies.df, valid.dummies.df, test.dummies.df
# run stepwise (this can take a while to run!!)
full.model <- glm(train.dummies.df$isReturnTrip ~., family="binomial", data=train.dummies.df)
stepwise.model<-step(full.model, direction = "both", trace = 1)

summary(full.model)

# evaluate
pred <- predict(full.model, valid.dummies.df)
confusionMatrix(ifelse(pred > 0.5, 1, 0), valid.dummies.df$ReturnTrip)
