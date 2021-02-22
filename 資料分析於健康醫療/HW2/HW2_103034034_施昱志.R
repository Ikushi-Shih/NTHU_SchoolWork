library(qcc)
library(xlsx)
library(e1071)


################################################

data <- read.xlsx("C:\\Users\\Horus\\Downloads\\SVM_20170308_data.xlsx",1,startRow=26, endRow=46)
data <- as.data.frame(data)
data <- subset(data, select = -5)
data.all <- read.xlsx("C:\\Users\\Horus\\Downloads\\SVM_20170308_data.xlsx",2)
data.all <- as.data.frame(data.all)
data.all <- subset(data.all, select = -5)
test <- read.xlsx("C:\\Users\\Horus\\Downloads\\SVM_20170308_data.xlsx",2)
test <- as.data.frame(test)


#sigma = 1
a.1 <- qcc(data, type = "xbar", nsigmas = 1)
a.2 <- qcc(data.all, type = "xbar", limits = c(a.1$limits),center = a.1$center)

test$result <- 1
for(i in 1:18){
  if(rowMeans(data.all)[[i]]>a.2$limits[2]){
    test$result[i] <- -1
  }
}
a.sen=0
a.spe=0
for(i in 1:18){
  if(test$Type[i] == 1 && test$result[i] == 1){
    a.sen <- a.sen+1
  }
  else if(test$Type[i] == -1 && test$result[i] == -1){
    a.spe <- a.spe+1
  }
}
a.sensitivity <- a.sen/9
a.specificity <- a.spe/9
a.accuracy <- (a.sen+a.spe)/18
sigma1 <- c(a.sensitivity,a.specificity,a.accuracy)
sigma1

################################################

#sigma = 2
b.1<-qcc(data, type = "xbar", nsigmas = 2)
b.2 <- qcc(data.all, type = "xbar", limits = c(b.1$limits),center = b.1$center)
test$result <- 1
test$result <- 1
for(i in 1:18){
  if(rowMeans(data.all)[[i]]>b.2$limits[2]){
    test$result[i] <- -1
  }
}
b.sen=0
b.spe=0
for(i in 1:18){
  if(test$Type[i] == 1 && test$result[i] == 1){
    b.sen <- b.sen+1
  }
  else if(test$Type[i] == -1 && test$result[i] == -1){
    b.spe <- b.spe+1
  }
}
b.sensitivity <- b.sen/9
b.specificity <- b.spe/9
b.accuracy <- (b.sen+b.spe)/18
sigma2 <-c(b.sensitivity,b.specificity,b.accuracy)
sigma2

################################################

#sigma = 2.5
c.1<-qcc(data, type = "xbar", nsigmas = 2.5)
c.2 <- qcc(data.all, type = "xbar", limits = c(c.1$limits),center = c.1$center)
test$result <- 1
test$result <- 1
for(i in 1:18){
  if(rowMeans(data.all)[[i]]>c.2$limits[2]){
    test$result[i] <- -1
  }
}
c.sen=0
c.spe=0
for(i in 1:18){
  if(test$Type[i] == 1 && test$result[i] == 1){
    c.sen <- c.sen+1
  }
  else if(test$Type[i] == -1 && test$result[i] == -1){
    c.spe <- c.spe+1
  }
}
c.sensitivity <- c.sen/9
c.specificity <- c.spe/9
c.accuracy <- (c.sen+c.spe)/18
sigma2.5 <- c(c.sensitivity,c.specificity,c.accuracy)
sigma2.5

################################################

#sigma = 3
d.1<-qcc(data, type = "xbar", nsigmas = 3)
d.2 <- qcc(data.all, type = "xbar", limits = c(d.1$limits),center = d.1$center)
test$result <- 1
test$result <- 1
for(i in 1:18){
  if(rowMeans(data.all)[[i]]>d.2$limits[2]){
    test$result[i] <- -1
  }
}
d.sen=0
d.spe=0
for(i in 1:18){
  if(test$Type[i] == 1 && test$result[i] == 1){
    d.sen <- d.sen+1
  }
  else if(test$Type[i] == -1 && test$result[i] == -1){
    d.spe <- d.spe+1
  }
}
d.sensitivity <- d.sen/9
d.specificity <- d.spe/9
d.accuracy <- (d.sen+d.spe)/18
sigma3 <- c(d.sensitivity,d.specificity,d.accuracy)
sigma3

################################################

#svm
trainset = read.xlsx("C:\\Users\\ieem\\Downloads\\SVM_20170308_data.xlsx",1)
testset = read.xlsx("C:\\Users\\ieem\\Downloads\\SVM_20170308_data.xlsx",2)
plot(trainset[1:(ncol(trainset)-1)],
     col=ifelse(trainset[ncol(trainset)]>0,1,2),pch=16,cex=2)

"==============================================================================================="
"Create Training data and Testing data"
"Use (-)  to exclude the test data"

trainset$Type <- as.factor(trainset$Type)


"==============================================================================================="
"Construct SVM  model"
svm.model <- svm(Type~.,data=trainset,kernel="radial",scale=c(),
                 type ="C-classification",gamma=0.01,cost=1)
svm.model
"See model"
"Plot model"
plot(svm.model,trainset,x_i1~x_i3)
plot(svm.model, trainset,x_i1~x_i2, grid=200,color.palette = terrain.colors)

"==============================================================================================="
"Testing part"
svm.pred <- predict(svm.model,testset)
sum(svm.pred==testset[,ncol(testset)])/nrow(testset)

print(svm.pred)
table(svm.pred)
e<- cbind(testset,svm.pred) 
e
e.sen=0
e.spe=0
for(i in 1:18){
  if(e$Type[i] == -1 && e$svm.pred[i] == -1){
    e.sen <- e.sen+1
  }
  else if(e$Type[i] == 1 && e$svm.pred[i] == 1){
    e.spe <- e.spe+1
  }
}
e.sensitivity <- e.sen/9
e.specificity <- e.spe/9
e.accuracy <- (e.sen+e.spe)/18
SVM <- c(e.sensitivity,e.specificity,e.accuracy)
SVM

result <- as.data.frame(cbind(sigma1,sigma2,sigma2.5,sigma3,SVM),row.names = c("sensitivity","specificity","accuracy"))
result








