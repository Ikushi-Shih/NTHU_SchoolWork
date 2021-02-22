library(qcc)
RN <- rnorm(30, 10, 1)
qcc(RN,type = "xbar.one",data.name = "Shewhart Chart")
sd(RN)

RN1 <- rnorm(10, 10, 1)
RN2 <- rnorm(10, 11, 1)
RN3 <- rnorm(10, 12, 1)

RN4 <- c(RN1,RN2,RN3)
x <-c(1:30)
y <- RN4
UCL <-  12.73372
CL <- 9.920845
LCL <- 7.107967

Control_Chart_Data <- data.frame(x,y,UCL,CL,LCL)
ggplot(data = Control_Chart_Data,aes(x = x,y = y))+
  geom_point()+geom_line()+
  geom_line(aes(y = LCL),color = "red")+
  geom_line(aes(y = CL),color = "green")+
  geom_line(aes(y = UCL),color = "red")
