library(Metrics)
library(binr)
library(rpart)
library(randomForest)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(caret)
setwd("D:\\Homework\\DMBA\\HW6")
new <- read.csv("new.csv")
new$weekdays <- weekdays(as.Date(new$cdate,'%Y-%m-%d'))
new$weekdays<- as.factor(new$weekdays)
new$weekdays2 <- as.POSIXlt(new$cdate)$wday
which(is.na(new)==TRUE)

binned <- bins.quantiles(new$quantity,target.bins = 20, max.breaks = 20)

## (a)set the seed to make your partition reproductible
set.seed(12345)

sample <- sample.int(n = nrow(new), size = floor(.5*nrow(new)), replace = F)
train <- new[sample, ]
test  <- new[-sample, ]

sample <- sample.int(n = nrow(test), size = floor(.6*nrow(test)), replace = F)
valid <- test[sample, ]
test  <- test[-sample, ]

write.csv(new, "new.df.csv")
#(b)
max(table(train$quantity))
rmse(1, test$quantity)

#regression tree
fit <- rpart(quantity ~type+weekdays2+price , data = train1, method = "anova", cp = 0.0001)
# count number of leaves
length(fit$frame$var[fit$frame$var == "<leaf>"])
# plot tree
prp(fit, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))  
fancyRpartPlot(fit,cex = 1)

bestcp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
pruned <- prune(fit, bestcp)
fancyRpartPlot(pruned,cex = 1)
length(pruned$frame$var[pruned$frame$var == "<leaf>"])

bestcp <- fit$cptable[which.min(fit$cptable[,"xstd"]),"CP"]
bspruned <- prune(fit, bestcp)
fancyRpartPlot(bspruned,cex = 1)
length(bspruned$frame$var[bspruned$frame$var == "<leaf>"])



bspruned.train <- predict(bspruned,train)
rmse(bspruned.train,train$quantity)

bspruned.valid <- predict(bspruned,valid)
rmse(bspruned.valid,valid$quantity)

bspruned.test <- predict(bspruned,test)
rmse(bspruned.test,test$quantity)


#classification tree
fit <- rpart(quantity ~type+weekdays2+price , data = train1, method = "class")
# count number of leaves
length(fit$frame$var[fit$frame$var == "<leaf>"])
# plot tree
prp(bspruned, type = 4, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))  
fancyRpartPlot(fit,cex = 1)

bestcp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
pruned <- prune(fit, bestcp)
fancyRpartPlot(pruned,cex = 0.5)
length(pruned$frame$var[pruned$frame$var == "<leaf>"])

bestcp <- fit$cptable[which.min(fit$cptable[,"xstd"]),"CP"]
bspruned <- prune(fit, bestcp)
fancyRpartPlot(bspruned)
length(bspruned$frame$var[bspruned$frame$var == "<leaf>"])



bspruned.train <- predict(bspruned,train)
rmse(bspruned.train,train$quantity)

bspruned.valid <- predict(bspruned,valid)
rmse(bspruned.valid,valid$quantity)

bspruned.test <- predict(bspruned,test)
rmse(bspruned.test,test$quantity)








printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits
# create additional plots 
par(mfrow=c(1,2)) # two plots on one page 
rsq.rpart(fit) # visualize cross-validation results  	










# plot tree 
prp(fit, type = 1, extra = 1, under = TRUE, split.font = 2, varlen = -10)





