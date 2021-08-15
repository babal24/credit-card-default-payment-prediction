#################################################
#Advanced Analytics & Machine Learning
#MGT7179
#Assignment 2 - Classification
#Vasileios Gounaris Bampaletsos-40314803

#Classification Problem
#Default Payments of Credit Card Clients in Taiwan

#The dataset contains information on default  payment
#demographic factors, credit data, payments' history
#bill statements of credit clients in Taiwan from
#April 2005 to September 2006


#Analyze and predict the default payment for the next month
#Our clients(banks) will have a good tool to predict
#their clients behavior about payments

#methods which used:
#Logistic Regression 
#Random Forest  (with cross validation)
#Decision Tree
#SVM 
#SVM Kernel
#Boosted Trees (XGBOOST method & cross validation)
#Naive Bayes
###################################################

# ~

#set the working directory
setwd("/Users/basilisgounarismpampaletsos/Desktop/PROJECTS 2/30:04 analytics")
options(scipen = 9)

#load the libraries
library(readxl)
library(psych)
library(ggplot2)
library(caTools)
library(statsr)
library(dplyr)
library(BAS)
library(car)
library(tidyr)
library(purrr)
library(gridExtra)
library(forcats)
library(corrplot)
library(magrittr)
library(caret)
library(Hmisc)
library(tidyverse)
library(ggpubr)
library(ROCR)
library(broom)
library(lubridate)
library(GGally)
library(ISLR)
library(hrbrthemes)
library(viridis)
library(e1071)
library(plyr)
library(readr)
library(repr)
library(glmnet)
library(ggthemes)
library(scales)
library(wesanderson)
library(styler)
library(xgboost)
library(randomForest)
library(rsample)      
library(gbm)          
library(h2o)          
library(pdp)          
library(lime)
library(naniar)
library(leaps)
library(tree)
library(MASS)
library(class)
library(data.table)
#library(gam)
library(sandwich)
library(rpart.plot)
library(lmtest)
library(ranger)
library(nnet)
library(pROC)
library(kernlab)

#load the data
data <- read.csv("Credit_Card.csv")

###########################################################
#summarize the data for the first time
#check the data's distribution and descriptive measures
summary(data)

#check the structure of our dataset
str(data)

#check the dataset if it has missing values
vis_miss(data)
sum(is.na(data))


#check some basic statistics in the dataset
#looking for the minimum, median and maximum values
#understand the data's distribution
#looking for outliers
summary(data$ID) #client ID number
summary(data$LIMIT_BAL) #stats for amount of given credit in NT dollars
summary(data$SEX) #stats for GENDER(1=male, 2=female)
summary(data$EDUCATION) #stats for education level(1,2,3,4,5,6)
summary(data$MARRIAGE) #stats for marital status(1=married, 2=single,3=others)
summary(data$AGE) #stats for the ages
summary(data$PAY_0) #stats for repayment status for September 2005(1 to 9 months)
summary(data$PAY_2) #stats for repayment status for August 2005(1 to 9 months)
summary(data$PAY_3) #stats for repayment status for July 2005(1 to 9 months)
summary(data$PAY_4) #stats for repayment status for June 2005(1 to 9 months)
summary(data$PAY_5) #stats for repayment status for May 2005(1 to 9 months)
summary(data$PAY_6) #stats for repayment status for April 2005(1 to 9 months)
summary(data$BILL_AMT1) #stats for bill statement for September 2005 (NT dollar)
summary(data$BILL_AMT2) #stats for repayment status for August 2005 (NT dollar)
summary(data$BILL_AMT3) #stats for repayment status for July 2005 (NT dollar)
summary(data$BILL_AMT4) #stats for repayment status for June 2005 (NT dollar)
summary(data$BILL_AMT5) #stats for repayment status for May 2005 (NT dollar)
summary(data$BILL_AMT6) #stats for repayment status for April 2005 (NT dollar)
summary(data$PAY_AMT1) #stats for repayment status for September 2005 (NT dollar)
summary(data$PAY_AMT2) #stats for repayment status for August 2005 (NT dollar)
summary(data$PAY_AMT3) #stats for repayment status for July 2005 (NT dollar)
summary(data$PAY_AMT4) #stats for repayment status for June 2005 (NT dollar)
summary(data$PAY_AMT5) #stats for repayment status for May 2005 (NT dollar)
summary(data$PAY_AMT6) #stats for repayment status for April 2005 (NT dollar)
summary(data$default.payment.next.month) #default payment (1=yes, 0=no)


#make basic visualisations for better understanding
#make simple histograms for numerice variables

#histograms help to check the data quality
#find the outliers and problems in the dataset

#data at this stage are all intiger/numeric
#NUMERIC VARIABLES
hist(data$ID)
hist(data$LIMIT_BAL)
hist(data$SEX)
hist(data$EDUCATION)
hist(data$MARRIAGE)
hist(data$AGE)
hist(data$PAY_0)
hist(data$PAY_2)
hist(data$PAY_3)
hist(data$PAY_4)
hist(data$PAY_5)
hist(data$PAY_6)
hist(data$BILL_AMT1)
hist(data$BILL_AMT2)
hist(data$BILL_AMT3)
hist(data$BILL_AMT4)
hist(data$BILL_AMT5)
hist(data$BILL_AMT6)
hist(data$PAY_AMT1)
hist(data$PAY_AMT2)
hist(data$PAY_AMT3)
hist(data$PAY_AMT4)
hist(data$PAY_AMT5)
hist(data$PAY_AMT6)
hist(data$default.payment.next.month)


##################################################################################################
#CORRELATIONS
##################################################################################################
#at this point we want to see the correlations how the variables connect each other
#check this connections/associations because they are important for models' building

#use 2 different ways to check the correlation
#very useful graphs
#full cor matrix, half cor matrix
#correlation matrix
#using pearson method 
continuous_var.cor = cor(na.omit(data), method = "pearson")
corrplot(continuous_var.cor)

#half correlation matrix
#using pearson method 
ggcorr(na.omit(data), method = c("everything", "pearson"))


##################################################################################################
#FIX THE DATA
##################################################################################################
#change the name of target variable because it is very big for our visualisations, correlation visualisations

#from default.payment.next.month to TARGET
names(data)[25] <- "TARGET"
#change the name PAY_0 to PAY_1 because we dont want a gap between our variables(0 to 2)
names(data)[7] <- "PAY_1"


#fix pay1
data$PAY_1[data$PAY_1 == -2] <- -1
data$PAY_1[data$PAY_1 == 0] <- 1
#fix pay2
data$PAY_2[data$PAY_2 == -2] <- -1
data$PAY_2[data$PAY_2 == 0] <- 1
#fix pay3
data$PAY_3[data$PAY_3 == -2] <- -1
data$PAY_3[data$PAY_3 == 0] <- 1
#fix pay4
data$PAY_4[data$PAY_4 == -2] <- -1
data$PAY_4[data$PAY_4 == 0] <- 1
#fix pay5
data$PAY_5[data$PAY_5 == -2] <- -1
data$PAY_5[data$PAY_5 == 0] <- 1
#fix pay6
data$PAY_6[data$PAY_6 == -2] <- -1
data$PAY_6[data$PAY_6 == 0] <- 1


#data$TARGET[data$TARGET == "1"] <- "Yes"
#data$TARGET[data$TARGET == "0"] <- "No"


#change the values of variables
#from numbers to characters

#variable: sex
#check how many unique values it has
unique(data$SEX)
#rename the values
data$SEX[data$SEX == "1"] <- "Male"
data$SEX[data$SEX == "2"] <- "Female"

#variable: education
#check how many unique values it has
unique(data$EDUCATION)
#combine the 3 unknown values(0,5,6) with value others(4)
data$EDUCATION <- ifelse(data$EDUCATION == 0 | data$EDUCATION == 5 | data$EDUCATION == 6,4, data$EDUCATION)
#rename the values of education
data$EDUCATION[data$EDUCATION == "1"] <- "Gradute School"
data$EDUCATION[data$EDUCATION == "2"] <- "University"
data$EDUCATION[data$EDUCATION == "3"] <- "High School"
data$EDUCATION[data$EDUCATION == "4"] <- "Others"
#variable: marriage
#check how many unique values it has
unique(data$MARRIAGE)
#combine the 0 value with 3(others) value
data$MARRIAGE <- ifelse(data$MARRIAGE == 0, 3, data$MARRIAGE)

data$MARRIAGE[data$MARRIAGE == "1"] <- "Married"
data$MARRIAGE[data$MARRIAGE == "2"] <- "Single"
data$MARRIAGE[data$MARRIAGE == "3"] <- "Others"

#change the type of variables
#INTIGER TO FACTOR
data$PAY_1 <- as.factor(data$PAY_1)
data$PAY_2 <- as.factor(data$PAY_2)
data$PAY_3 <- as.factor(data$PAY_3)
data$PAY_4 <- as.factor(data$PAY_4)
data$PAY_5 <- as.factor(data$PAY_5)
data$PAY_6 <- as.factor(data$PAY_6)
data$SEX <- as.factor(data$SEX)
data$EDUCATION <- as.factor(data$EDUCATION)
data$MARRIAGE <- as.factor(data$MARRIAGE)
data$TARGET <- as.factor(data$TARGET)

#delete the ID variable because is useless
data$ID <- NULL

#check the distribution of clean data
#only the variables as factor (no the numeric)
table(data$TARGET)
table(data$SEX)
table(data$AGE)
table(data$MARRIAGE)
table(data$EDUCATION)
table(data$PAY_1)
table(data$PAY_2)
table(data$PAY_3)
table(data$PAY_4)
table(data$PAY_5)
table(data$PAY_6)


##################################################################################################
#FINAL VISUALISATIONS
#VISUALISE THE DATA FOR BETTER UNDERSTANDING
##################################################################################################

#create some demographics visualisations
#include education, age, marital status, limit balance and target
#help the audiance(clients) to understand the data

#1 - Gender(SEX) combining with default payment(target variable)
#see how many people choose which option
ggplot(data = data, mapping = aes(x = SEX, fill = TARGET)) +
  geom_bar() +
  labs(fill="default payment", x="Gender", y= "count", 
       title="Gender distribution per target variable", 
       caption="2 types of answers") +
  stat_count(aes(label = ..count..), geom = "label")

#Distribution of credit balance per target and age
ggplot(data = data, aes(x= AGE, y = LIMIT_BAL, fill=TARGET)) + 
  geom_boxplot(alpha=0.3) +
  theme(legend.position="none") +
  labs(x="Age", y= "Credit Balance (NT dollars)", 
       title="Distribution of credit balance per target and age")

#boxplot to check the education level of the customers
ggplot(data = data, mapping = aes(x = EDUCATION, fill = TARGET)) +
  geom_bar() +
  ggtitle("EDUCATION") +
  stat_count(aes(label = ..count..), geom = "label") +
  labs(x="Education level", y= "count", 
       title="Education level distribution per target")

#Distribution of credit balance per age and education level
ggplot(data = data, mapping = aes(x=AGE, y=LIMIT_BAL))+
  geom_boxplot(aes(fill = EDUCATION))+
  facet_wrap(~TARGET) +
  labs(x="Age", y= "Amount of Credit Balance (NT dollar)", 
       title="Distribution of credit balance per age and education level 1")

#graph to understand better the distribution of credit balance
#in ages and education level
ggplot(data = data, mapping = aes(x=AGE, y=LIMIT_BAL)) +
  geom_point()+
  geom_smooth(aes(color = EDUCATION))+
  facet_wrap(~EDUCATION) +
  labs(x="Age", y= "Credit Balance (NT dollars)", 
       title="Distribution of credit balance per age and education level 2")


#marital status
ggplot(data = data, mapping = aes(x = MARRIAGE, fill = TARGET)) +
  geom_bar() +
  stat_count(aes(label = ..count..), geom = "label") +
  labs(x="Marrital Status", y= "count", 
       title="Marrital Status distribution per target")

#martial status and balance
ggplot(data = data, mapping = aes(x=MARRIAGE, y=LIMIT_BAL))+
  geom_boxplot(aes(fill = SEX))+
  facet_wrap(~TARGET) +
  labs(x="Marrital Status", y= "Amount of Credit Balance (NT dollar)", 
       title="Distribution of credit balance per marital status and gender")

#CONTINUOUS VARIABLES
#CHECK the economy variables
#limit balance (LIMIT_BAL)
#payment delays (PAY_1-6)
#bill statement (BILL_AMT1-6)
#payments (PAY_AMT1-6)

#BILL_AMT1
#check the distribution of the amount of bills in dataset
#how much money customers need to pay
ggplot(data, aes(x = BILL_AMT1))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(BILL_AMT1)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of bill statement in September (BILL_AMT1)", 
       caption="With the mean line in red")

#BILL_AMT2
ggplot(data, aes(x = BILL_AMT2))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(BILL_AMT2)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of bill statement in August (BILL_AMT2)", 
       caption="With the mean line in red")

#BILL_AMT3
ggplot(data, aes(x = BILL_AMT3))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(BILL_AMT3)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of bill statement in July (BILL_AMT3)", 
       caption="With the mean line in red")

#BILL_AMT4
ggplot(data, aes(x = BILL_AMT4))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(BILL_AMT4)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of bill statement in June (BILL_AMT4)", 
       caption="With the mean line in red")

#BILL_AMT5
ggplot(data, aes(x = BILL_AMT5))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(BILL_AMT5)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of bill statement in May (BILL_AMT5)", 
       caption="With the mean line in red")

#BILL_AMT6
ggplot(data, aes(x = BILL_AMT6))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(BILL_AMT6)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of bill statement in April (BILL_AMT6)", 
       caption="With the mean line in red")

#PAY_AMT1
#check the payment statement 
ggplot(data, aes(x = PAY_AMT1))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(PAY_AMT1)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of previous payment in September (PAY_AMT1)", 
       caption="With the mean line in red")

#PAY_AMT2
ggplot(data, aes(x = PAY_AMT2))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(PAY_AMT2)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of previous payment in August (PAY_AMT2)", 
       caption="With the mean line in red")

#PAY_AMT3
ggplot(data, aes(x = PAY_AMT3))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(PAY_AMT3)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of previous payment in July (PAY_AMT3)", 
       caption="With the mean line in red")

#PAY_AMT4
ggplot(data, aes(x = PAY_AMT4))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(PAY_AMT4)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of previous payment in June (PAY_AMT4)", 
       caption="With the mean line in red")

#PAY_AMT5
ggplot(data, aes(x = PAY_AMT5))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(PAY_AMT5)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of previous payment in May (PAY_AMT5)", 
       caption="With the mean line in red")

#PAY_AMT6
ggplot(data, aes(x = PAY_AMT6))+ 
  geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(PAY_AMT6)), linetype = "dashed", size = 0.6, color = "#FC4E07") +
  labs(x="NT dollar", y= "count", 
       title="Distribution of previous payment in April (PAY_AMT6)", 
       caption="With the mean line in red")

#PAY_1-6
#check the distribution of delays
#how people payed their bills(no delay, 1month, 2,.....)
#and also how they choose to pay next month's payment

#PAY_1
ggplot(data, aes(PAY_1)) +
  geom_bar(colour="black", mapping = aes(fill = TARGET)) +
  labs(x="months of delay", y= "count", fill="default payment next month",
       title="Payment Delays Distribution in September(PAY_1)")

#PAY_2
ggplot(data, aes(PAY_2)) +
  geom_bar(colour="black", mapping = aes(fill = TARGET)) +
  labs(x="months of delay", y= "count", fill="default payment next month",
       title="Payment Delays Distribution in August(PAY_2)")

#PAY_3
ggplot(data, aes(PAY_3)) +
  geom_bar(colour="black", mapping = aes(fill = TARGET)) +
  labs(x="months of delay", y= "count", fill="default payment next month",
       title="Payment Delays Distribution in July(PAY_3)")

#PAY_4
ggplot(data, aes(PAY_4)) +
  geom_bar(colour="black", mapping = aes(fill = TARGET)) +
  labs(x="months of delay", y= "count", fill="default payment next month",
       title="Payment Delays Distribution in June(PAY_4)")

#PAY_5
ggplot(data, aes(PAY_5)) +
  geom_bar(colour="black", mapping = aes(fill = TARGET)) +
  labs(x="months of delay", y= "count", fill="default payment next month",
       title="Payment Delays Distribution in May(PAY_5)")

#PAY_6
ggplot(data, aes(PAY_6)) +
  geom_bar(colour="black", mapping = aes(fill = TARGET)) +
  labs(x="months of delay", y= "count", fill="default payment next month",
       title="Payment Delays Distribution in April(PAY_6)")

#violin graph
#check the distribution of target variable in the scale of limit balance variable
#how balance(NT dollars) they have and how they pay(default or not)
ggplot(data = data, aes(x = TARGET, y = LIMIT_BAL)) +
  geom_violin(aes(fill = TARGET), trim = FALSE, alpha = 0.3) +
  geom_boxplot(aes(fill = TARGET), width = 0.2, outlier.colour = NA) +
  theme(legend.position = "NA") +
  labs(fill="Payment next month", x="Default Payment Next Month", y= "Limit Balance (NT dollars)", 
       title="Distribution of credit balance per customers choose for payment", 
       caption="The boxes present the mean")





###############################################################
#Prepare the data for the methods - SPLIT THE DATA
###############################################################
#split the data
#train_set 75%
#test_set 25%
set.seed(123)
split = sample.split(data$TARGET, SplitRatio = 0.75)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

##################################################################################################
#Find the best variables for our models
#Use Bayesian Information Criterion (BIC) to find the best variables
#using forward selection
best_sel = regsubsets(TARGET ~. , data = train_set, method = "forward",
                      nvmax = length(data)-1)
best_sel_sum = summary(best_sel)

plot(best_sel_sum$bic, type = 'b', col = "blue", pch = 19, 
     xlab = "Number of Variables",
     ylab = "Cross-Validated Prediction Error",
     main = "Forward Stepwise Selection using BIC")
points(which.min(best_sel_sum$bic), best_sel_sum$bic[which.min(best_sel_sum$bic)],
       col = "red", pch = 19)

#list all variable which are important to use
final = t(best_sel_sum$which)[,which.min(best_sel_sum$bic)]
final_names = names(data)[-24]
final_names[final[24:length(data)]]
final_names


##################################################################################################
#lOGISTIC REGRESSION
##################################################################################################
#create the logistic regression model
log_reg <- glm(formula = TARGET ~., 
                 family = binomial(link="logit"), 
               data = train_set)
summary(log_reg)

#make predictions 
log_pred <- predict(log_reg, newdata = test_set[-24], type = 'response') 
log_pred

#set the probabilities >0.5 = yes and <0.5 = no
class_pred <- as.factor(ifelse(log_pred > 0.5, 1, 0))


#build the odds for the final model's variables
exp(log_reg$coefficients)

#confusion matrix and other statistics
confusionMatrix(class_pred, test_set$TARGET)

#R^2 
logisticPseudoR2s <- function(LogModel) {
  dev <- LogModel$deviance 
  nullDev <- LogModel$null.deviance 
  modelN <- length(LogModel$fitted.values)
  R.l <-  1 -  dev / nullDev
  R.cs <- 1- exp ( -(nullDev - dev) / modelN)
  R.n <- R.cs / ( 1 - ( exp (-(nullDev / modelN))))
  cat("Pseudo R^2 for logistic regression\n")
  cat("Hosmer and Lemeshow R^2  ", round(R.l, 3), "\n")
  cat("Cox and Snell R^2        ", round(R.cs, 3), "\n")
  cat("Nagelkerke R^2           ", round(R.n, 3),    "\n")
}

logisticPseudoR2s(log_reg)



head(log_pred)
class(log_pred)


#ROC
#transform the input data into a standardized format.
pr <- prediction(log_pred, test_set$TARGET)
#All kinds of predictor evaluations are performed using this function.
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
prf
plot(prf)

#AUC
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]   
auc

#Precision and Recall
precision_recall <- performance(pr, "prec", "rec")
plot(precision_recall)



##################################################################################################
#CHECK ASSUMPTIONS
##################################################################################################

#assumption 1
#linearity 
train_set$LIMIT_BAL_log <- log(train_set$LIMIT_BAL)*train_set$LIMIT_BAL
train_set$AGE_log <- log(train_set$AGE)*train_set$AGE
train_set$BILL_AMT1_log <- log(train_set$BILL_AMT1)*train_set$BILL_AMT1
train_set$BILL_AMT2_log <- log(train_set$BILL_AMT2)*train_set$BILL_AMT2
train_set$BILL_AMT3_log <- log(train_set$BILL_AMT3)*train_set$BILL_AMT3
train_set$BILL_AMT4_log <- log(train_set$BILL_AMT4)*train_set$BILL_AMT4
train_set$BILL_AMT5_log <- log(train_set$BILL_AMT5)*train_set$BILL_AMT5
train_set$BILL_AMT6_log <- log(train_set$BILL_AMT6)*train_set$BILL_AMT6
train_set$PAY_AMT1_log <- log(train_set$PAY_AMT1)*train_set$PAY_AMT1
train_set$PAY_AMT2_log<- log(train_set$PAY_AMT2)*train_set$PAY_AMT2
train_set$PAY_AMT3_log<- log(train_set$PAY_AMT3)*train_set$PAY_AMT3
train_set$PAY_AMT4_log<- log(train_set$PAY_AMT4)*train_set$PAY_AMT4
train_set$PAY_AMT5_log<- log(train_set$PAY_AMT5)*train_set$PAY_AMT5
train_set$PAY_AMT6_log<- log(train_set$PAY_AMT6)*train_set$PAY_AMT6
train_set$PAY_AMT1<- log(train_set$PAY_AMT1)*train_set$PAY_AMT1

formula <- TARGET ~.

model <- glm(formula, family = "binomial", data = train_set)
summary(model)


#assumption 2
#infuential values
#cook's distance graph
plot(log_reg, which = 4, id.n = 3)

# Extract model results
model_data <- augment(log_reg) %>% mutate(index = 1:n()) 

#the data for the top 3 values as we can see in the cook's graph
model_data %>% top_n(3, .cooksd)

#plot the standardised Residuals
ggplot(model_data, aes(index, .std.resid)) + 
  geom_point(aes(color = TARGET), alpha = .5) +
  theme_bw()

#filter potential infuential data points
model_data %>% filter(abs(.std.resid) > 3)

#how many variables violate the cook's distance
cook <- cooks.distance(log_reg)
sum(cook > 1)

#residuals fit
x <- rstandard(log_reg)
sum(x > 1.96)


#assumption 3
#multicollinearity
vif(log_reg)


##################################################################################################
#Random Forest x2 with tuning (cross validation k-folds = 10)
##################################################################################################
#create the random forest model
rf_class <- randomForest(x = train_set[-24],
                         y = train_set$TARGET,
                         ntree = 10)
summary(rf_class)

#make predictions in the test set
rf_pred <- predict(rf_class, newdata = test_set[-24])
rf_pred

#create the confusion matrix to see the model's performance
confusionMatrix(rf_pred, test_set$TARGET)


#evaluate the model using k-fold cross-validation
#search if we can boost the model's performance
fold = createFolds(y = train_set$TARGET, k=10)
rf_cross = lapply(fold, function(x){
  train_fold = train_set[-x, ]
  test_fold = train_set[x, ]
  rf_class_cross = randomForest(formula = TARGET ~. ,
                               data = train_fold, ntree = 20)
  rf_pred_cross = predict(rf_class_cross, newdata = test_fold,
                          type = "class")
  rf_conf = confusionMatrix(rf_pred_cross, test_fold$TARGET)
  rf_conf
  accuracy = rf_conf$overall[1]
  return(accuracy)
})

#check the tuning model's accuracy
mean(as.numeric(rf_cross))



##################################################################################################
#Decision Tree Classification
##################################################################################################
#create the decision tree model
dt_class <- rpart(formula = TARGET ~. , data = train_set, method = 'class')
summary(dt_class)

#plot the decision tree
plot(dt_class)
text(dt_class)

rpart.plot(dt_class, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)

#make predictions to test set
dt_pred <- predict(dt_class, newdata = test_set, type = 'class')
dt_pred

#confusion matrix
confusionMatrix(dt_pred, test_set$TARGET)


##################################################################################################
#SUPPORT VECTOR MACHINE (SVM)
##################################################################################################
#create the svm model
svm_class <- svm(formula = TARGET ~. , 
                 data = train_set,
                 type = 'C-classification', 
                 kernel = 'linear')
summary(svm_class)

#make predictions to test set
svm_pred <- predict(svm_class, newdata = test_set[-24])
svm_pred
#confusion matrix
confusionMatrix(svm_pred, test_set$TARGET)


##################################################################################################
#KERNEL SUPPORT VECTOR MACHINE (KERNEL SVM) (gaussian)
##################################################################################################
#create the svm kernel model
svm_kernel <- svm(formula = TARGET ~. , 
                  data = train_set,
                 type = 'C-classification', 
                 kernel = 'radial')
summary(svm_kernel)

#make predictions to test set
svm_pred_kernel <- predict(svm_kernel, newdata = test_set[-24])
svm_pred_kernel

#confusion matrix for svm kernel(gaussian)
confusionMatrix(svm_pred_kernel, test_set$TARGET)


##################################################################################################
#NAIVE BAYES
##################################################################################################
#create the naive bayes model
nv_class <- naiveBayes(x = train_set[-24], 
                       y = train_set$TARGET)
summary(nv_class)

#make predictions to test set
nv_pred <- predict(nv_class, newdata = test_set[-24], type = "class")
nv_pred

#confusion matrix for naive bayes
confusionMatrix(nv_pred, test_set$TARGET)

##################################################################################################
#XGBOOST DART (cross validation k-folds = 5)
##################################################################################################
#create the xgboost model
xgb_class <- train(TARGET ~., 
               data = train_set, 
               method = "xgbTree", trControl = trainControl("cv", number = 5))
summary(xgb_class)

# Best tuning parameter mtry
xgb_class$bestTune
# Make predictions on the test data
xgb_pred <- predict(xgb_class, newdata = test_set$TARGET)
xgb_pred
head(xgb_pred)

confusionMatrix(xgb_pred, test_set$TARGET)