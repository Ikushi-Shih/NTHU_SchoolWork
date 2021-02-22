t1 <- Sys.time()
library(readxl)
ebay <- read_xlsx("D:\\Homework\\DMBA\\HW6\\eBay Logistic model selection.xlsx",sheet = 1)
colnames(ebay)[19] <- "Competitive"

#model1 : regression on all predictors
logit.reg <- glm(Competitive~ ., data = ebay) 
options(scipen=999)
summary(logit.reg)

#model2 : stepwise selection
library(MASS)
step <- stepAIC(logit.reg, direction="both")
step$anova # display results
options(scipen=999)
summary(step)
t2 <- Sys.time()
t2-t1