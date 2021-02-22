setwd("D:\\Homework\\DMBA\\HW3\\purchase_item")
data <- purchase_item
nrow(table(data$id))
nrow(data)

nrow(table(data$restaurant_id))
sort_rest_id <- sort(table(data$restaurant_id),decreasing=T)
(sort_rest_id[1]+sort_rest_id[2])/nrow(data)
barplot(sort_rest_id)

nrow(table(data$purchase_id))
sort_purchase_id <- sort(table(data$purchase_id),decreasing=T)
(sort_purchase_id[1]+sort_purchase_id[2])/nrow(data)
barplot(sort_purchase_id)

nrow(table(data$type))
sort_type <- sort(table(data$type),decreasing=T)
(sort_type[1]+sort_type[2])/nrow(data)
barplot(sort_type)

nrow(table(data$product_id))
sort_product_id <- sort(table(data$product_id),decreasing=T)
(sort_product_id[1]+sort_product_id[2])/nrow(data)
barplot(sort_product_id)

nrow(table(data$name))
sort_name <- sort(table(data$name),decreasing=T)
(sort_name[1]+sort_name[2])/nrow(data)
barplot(sort_name)

hist(table(data$quantity))
hist(table(data$price))
hist(table(data$cdate))


new<-data[which(data$restaurant_id=="da5aebea74a06b5001809d64e99d502774fa3c3a"),]
write.csv(new,"new.csv")
new<-new[-c(which(new$quantity==0)),]
new$Weekdays <- weekdays(new$cdate)
new$ln <- log(new$quantity)
new$type[which(new$type=="PREPAY")] <- 0
new$type[which(new$type=="COUPON")] <- 1

library(plyr)
new$Weekdays <-revalue(new$Weekdays, c("Monday"="1","Tuesday"="2","Wednesday"="3","Thursday"="4","Friday"="5","Saturday"="6","Sunday"="7"))

