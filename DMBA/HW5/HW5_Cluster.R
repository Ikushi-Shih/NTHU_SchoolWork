setwd("D:\\Homework\\DMBA\\HW5")
library(readxl)
library(MASS)
library(base)
data <- read_excel("EastWestAirlinesCluster.xlsx",2)

#(A)
parcoord(data, col = rainbow(length(data[,2:12])))

db <- data[1:2000,]

db.norm<- sapply(db,scale)
row.names(db.norm)<- row.names(db)

ds <- dist(db.norm[,2:12],method = "euclidean")
hc1<- hclust(ds,method = "ward.D2")
plot(hc1,hang= -1, ann= F)
memb<- cutree(hc1, k=2)
memb
parcoord(memb)

kmeans(db.norm,2)
