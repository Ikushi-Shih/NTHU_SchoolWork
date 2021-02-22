library(ggplot2)
library(jpeg)
# The iris dataset is proposed by R

softdrink <- softdrin[order(softdrin$Time),]
setwd("D:\\Homework\\DMBA\\Week 6")
# basic scatterplot
ggplot(softdrin, aes(x=Cases, y=Time)) + 
  geom_point()
ggsave("Scatter_Time_Cases.jpg")

ggplot(softdrin, aes(x=Distance, y=Time)) + 
  geom_point()
ggsave("Scatter_Time_Distance.jpg")

ggplot(softdrin, aes(x=Cases, y=Distance)) + 
  geom_point()
ggsave("Scatter_Cases_Distance.jpg")

regression <- lm(Time~Cases + Distance, data = softdrin)
summary(regression)

Time_Cases <- lm(Time~Cases, data = softdrin)
summary(Time_Cases)

Time_Distance <- lm(Time~Distance, data = softdrin)
summary(Time_Distance)
