#Import needed libraries

library(funModeling) 
library(tidyverse)
library(reshape2)
library(Hmisc)
library(ggplot2)
library(randomForest) #for random forest
library(scales)
library(caret)
library(data.table)
library(readr)
library(knitr)
library(leaps)
library(car)
library(mice)
library(RColorBrewer)
library(plotly)
library(nortest)
library(lmtest)
library(Amelia)
library(ggcorrplot)
library(e1071) #Contains svm, naive bayes
library(ISLR)
library(xgboost)
library(plyr)
library(gbm)
library(ROCR)
library(minerva) # contains MIC statistic
library(rpart) #decison tree
library(rpart.plot) #decision tree

#--------------------------------------------------------------------------------------------->
#Getting the data

# import CSV file for Bank loan data
loan_data <- read.csv("BANK-LOAN_Data.csv", header = TRUE)
View(loan_data)

#--------------------------------------------------------------------------------------------->
# EXPLORATORY DATA ANALYSIS

#Quick Cleaning
loan_data <- loan_data[,-18]
loan_data <- loan_data[,-12:-14]
loan_data <- loan_data[,-1:-2]


str(loan_data)
colnames(loan_data)
summary(loan_data)

# Removing the an observation from loan_data
loan_data <- na.omit(loan_data, cols=c("Maximum.Open.Credit"))
summary(loan_data)

# Check the distribution for catgeorial variables
freq(loan_data)
# Check the distribution for continuous variables
plot_num(loan_data, bins=40)
missmap(loan_data, main = "Missing vs Observed values")

#Check scatterplot of continuous variable
qplot(Credit.Score, Annual.Income, data = loan_data, color = Home.Ownership)
qplot(Credit.Score, Years.of.Credit.History, data = loan_data, color = Years.in.current.job)
qplot(Credit.Score, Number.of.Open.Accounts, data = loan_data, color = Years.in.current.job)

dim(loan_data)
# Retrieve R metric (or Pearson coefficient) for all numeric variables, skipping the categoric ones.
correlation_table(loan_data, "Loan.Status")

# Visualizing the correlation
ggcorrplot(cor(loan_data))

# Retrieving the relative and absolute distribution between an input and target variable. 
# To identify if a variable is important or not.
cross_plot(loan_data, input="Term", target="Loan.Status")
cross_plot(loan_data, input="Years.in.current.job", target="Loan.Status")
cross_plot(loan_data, input="Home.Ownership", target="Loan.Status")
cross_plot(loan_data, input="Years.of.Credit.History", target="Loan.Status")

# Turn categoricals of Home Ownership & Years in current Job into booleans
categories = unique(loan_data$Home.Ownership)
categories2 = unique(loan_data$Years.in.current.job)

# Modifying the area name variable
# split the categories off
cat_loan = data.frame(Home.Ownership = loan_data$Home.Ownership)

for(i in categories){
  cat_loan[,i] = rep(0, times= nrow(cat_loan))
}
head(cat_loan) #New columns are generated to the right

for(k in 1:length(cat_loan$Home.Ownership)){
  coke = as.character(cat_loan$Home.Ownership[k])
  cat_loan[,coke][k] = 1
}
head (cat_loan)

cat_columns = names(cat_loan)
keep_columns = cat_columns[cat_columns != 'Home.Ownership']
cat_loan = select(cat_loan,one_of(keep_columns))
tail(cat_loan)

# Modifying Years.in.current.job variable
# split the categories off
cat_loan2 = data.frame(Years.in.current.job = loan_data$Years.in.current.job)

for(j in categories2){
  cat_loan2[,j] = rep(0, times= nrow(cat_loan2))
}
head(cat_loan2) #New columns are generated to the right

for(x in 1:length(cat_loan2$Years.in.current.job)){
  coke = as.character(cat_loan2$Years.in.current.job[x])
  cat_loan2[,coke][x] = 1
}
head (cat_loan2)

cat_columns2 = names(cat_loan2)
keep_columns2 = cat_columns2[cat_columns2 != 'Years.in.current.job']
cat_loan2 = select(cat_loan2,one_of(keep_columns2))
tail(cat_loan2)


#merge back the categorical dataframes and altered numerical variables
cleaned_loan_data = cbind(loan_data,cat_loan2,cat_loan)
cleaned_loan_data <- cleaned_loan_data[,-11:-12]
View(cleaned_loan_data)
head(cleaned_loan_data)
str(cleaned_loan_data)

# Rename the column names
colnames(cleaned_loan_data)[10:17] <- c("Loan.Term", "YCJ_5to9yrs", "YCJ_Under4yrs", "YCJ_10plusyrs", "Home.Mortgage", "Own.Home", "Rent", "Have.Mortgage")
head(cleaned_loan_data)
View(cleaned_loan_data)


# Change data type of cleaned data to factor
str(cleaned_loan_data)
cleaned_loan_data$YCJ_5to9yrs <- as.factor(cleaned_loan_data$YCJ_5to9yrs)
cleaned_loan_data$YCJ_Under4yrs <- as.factor(cleaned_loan_data$YCJ_Under4yrs)
cleaned_loan_data$YCJ_10plusyrs <- as.factor(cleaned_loan_data$YCJ_10plusyrs)
cleaned_loan_data$Home.Mortgage <- as.factor(cleaned_loan_data$Home.Mortgage)
cleaned_loan_data$Own.Home <- as.factor(cleaned_loan_data$Own.Home)
cleaned_loan_data$Rent <- as.factor(cleaned_loan_data$Rent)
cleaned_loan_data$Have.Mortgage <- as.factor(cleaned_loan_data$Have.Mortgage)

freq(cleaned_loan_data)
contrasts(cleaned_loan_data$Loan.Status)
#------------------------------------------------------------------>
# Further Exploration

cross_plot(cleaned_loan_data, input="Loan.Term", target="Loan.Status")
cross_plot(cleaned_loan_data, input="YCJ_5to9yrs", target="Loan.Status")
cross_plot(cleaned_loan_data, input="YCJ_Under4yrs", target="Loan.Status")
cross_plot(cleaned_loan_data, input="YCJ_10plusyrs", target="Loan.Status")
cross_plot(cleaned_loan_data, input="Home.Mortgage", target="Loan.Status")
cross_plot(cleaned_loan_data, input="Own.Home", target="Loan.Status")
cross_plot(cleaned_loan_data, input="Rent", target="Loan.Status")
cross_plot(cleaned_loan_data, input="Have.Mortgage", target="Loan.Status")

cleaned_loan_data <- cleaned_loan_data[,-17]
cleaned_loan_data <- cleaned_loan_data[,-15]

str(cleaned_loan_data)
summary(cleaned_loan_data)


#------------------------------------------------------------------>
# Creating Test and Train Data

# Split into Train and Testing set
# Training Set : Testing Set = 70 : 30 (randomly)
set.seed(120)
train_LNsample <- sample(nrow(cleaned_loan_data), 0.7*nrow(cleaned_loan_data), replace = FALSE)
train_loan <- cleaned_loan_data[train_LNsample,]
test_loan <- cleaned_loan_data[-train_LNsample,]
summary(train_loan)
View(train_loan)
summary(test_loan)
View(test_loan)

#--------------------------------------------------------------------------------------------------------------------->
# DATA MODELLING AND EVALUATION OF MODELS

#-------------------------------------------------------------------------------------------------|
# Random Forest

#Find the best value for mtry
bestmtry_loan <- tuneRF(train_loan,train_loan$Loan.Status, stepFactor = 1.2, improve = 0.1, trace=T, plot=T)
# mtry is 3

# Create a Random Forest model_loan with default parameters
rfmodel_loan <- randomForest(Loan.Status ~ ., data = train_loan, importance = TRUE)
rfmodel_loan
# checking the Importance of the variables
importance(rfmodel_loan)
varImpPlot(rfmodel_loan)

# Predicting on test data
predLoanTest_rf <- predict(rfmodel_loan, newdata = test_loan, type = "class")

# Check confusion Matrix
tabloan_rf <- table(Actual = test_loan$Loan.Status, Predicted = predLoanTest_rf)
tabloan_rf
# Checking classification accuracy
accuracyloan_rf <- mean(predLoanTest_rf == test_loan$Loan.Status)
accuracyloan_rf
# Check misclassification Error
misclassRateloan_rf <- 1-sum(diag(tabloan_rf))/sum(tabloan_rf)
misclassRateloan_rf

confusionMatrix(table(Actual = test_loan$Loan.Status, Predicted = predLoanTest_rf))
#-------------------------------------------------------------------------------------------------||
# Decision Trees

# Create a decision tree model_loan with default parameters
dtmodel_loan <- train(Loan.Status ~ ., data = train_loan, method = "rpart")
dtmodel_loan
loantree.rpart <- rpart(Loan.Status ~., data = train_loan)
rpart.plot(loantree.rpart)

# Predicting on test data
predLoanTest_dt <- predict(dtmodel_loan, test_loan)

# Check confusion Matrix
tabloan_dt <- table(Actual = test_loan$Loan.Status, Predicted = predLoanTest_dt)
tabloan_dt
# Checking classification accuracy
accuracyloan_dt <- mean(predLoanTest_dt == test_loan$Loan.Status)
accuracyloan_dt
# Check misclassification Error
misclassRateloan_dt <- 1-sum(diag(tabloan_dt))/sum(tabloan_dt)
misclassRateloan_dt


#-------------------------------------------------------------------------------------------------|
# Support Vector Machine

# Create a Support vector machine model with default parameters
svmmodel_loan <- svm(Loan.Status ~ ., data = train_loan)
summary(svmmodel_loan)
# Visualise the model
plot(svmmodel_loan, data = train_loan)

# Predict with Test Data
predLoanTest_svm <- predict(svmmodel_loan, test_loan)

# Check Confusion Matrix
tabloan_svm <- table(Actual = test_loan$Loan.Status, Predicted = predLoanTest_svm)
tabloan_svm
# Check classification Accuracy
accuracyloan_svm <- mean(predLoanTest_svm == test_loan$Loan.Status)
accuracyloan_svm
# Check misclassification Error
misclassRateloan_svm <- 1-sum(diag(tabloan_svm))/sum(tabloan_svm)
misclassRateloan_svm

str(cleaned_crime_data)

## Tuning #HyperParameter Optimisation which helps to select the best model
#set.seed(123)
#tmodel_loan <- tune(svmmodel_loan, Loan.Status~., data=train_loan, ranges = list (epsilon = seq(0,1,0.1), cost = 2^(2:4)))
#plot(tmodel_loan)
#summary(tmodel_loan)
## best Model
#tmodel_loan$best.model


#-------------------------------------------------------------------------------------------------|
# Logistic Regression

lrmodel_loan <- glm(Loan.Status ~., family=binomial(link='logit'), data = train_loan)

# Check the summary of the model_loan
summary(lrmodel_loan)

# Inspecting for correlation in the model_loan
alias(lrmodel_loan)

# Predict with Test Data
predLoanTest_lr <- predict(lrmodel_loan, newdata=test_loan, type='response')
# Assessing the predictive ability of the model
predLoanTest_lr <- ifelse(predLoanTest_lr > 0.5,1,0)

# Check Confusion matrix
tabloan_lr <- table(Actual = test_loan$Loan.Status, Predicted = predLoanTest_lr)
tabloan_lr
# Checking classification accuracy
accuracyloan_lr <- mean(predLoanTest_lr == test_loan$Loan.Status)
accuracyloan_lr
# Check misclassification Error
misclassRateloan_lr <- 1-sum(diag(tabloan_lr))/sum(tabloan_lr)
misclassRateloan_lr
# Checking classification accuracy # Alternate Method
#misClassificError <- mean(predLoanTest_lr != test_loan$Loan.Status)
#print(paste('Accuracy: ', 1-misClassificError))

# analyze the table of deviance using anova function
anova(lrmodel_loan, test="Chisq")

# To get the McFadden r-squared index
library(pscl)
pR2(lrmodel_loan)
# Plot the ROC curve -- performance measurements for a binary classifier
predLoanTest_lr2<- predict(lrmodel_loan, newdata=test_loan, type="response")
prloan_lr <- prediction(predLoanTest_lr2, test_loan$Loan.Status)
ROC_loan <- performance(prloan_lr, measure = "tpr", x.measure = "fpr")
plot(ROC_loan)

# calculate the AUC (area under the curve) -- performance measurements for a binary classifier
auc_loan <- performance(prloan_lr, measure = "auc")
auc_loan <- auc_loan@y.values[[1]]
auc_loan


#-------------------------------------------------------------------------------------------------|
# Naive Bayes Model

# Create a Support vector machine model with default parameters
nbmodel_loan <- naiveBayes(Loan.Status ~ ., data = train_loan)
summary(nbmodel_loan)
# Visualise the model
plot(nbmodel_loan, data = train_loan)

# Predict with Test Data
predLoanTest_nb <- predict(nbmodel_loan, test_loan)

# Check Confusion Matrix
tabloan_nb <- table(Actual = test_loan$Loan.Status, Predicted = predLoanTest_nb)
tabloan_nb
# Check classification Accuracy
accuracyloan_nb <- mean(predLoanTest_nb == test_loan$Loan.Status)
accuracyloan_nb
# Check misclassification Error
misclassRateloan_nb <- 1-sum(diag(tabloan_nb))/sum(tabloan_nb)
misclassRateloan_nb


#-------------------------------------------------------------------------------------------------------|
#comparing metrics

metrics_loan <- data.frame(ML_Algorithm = rep(c("Random Forest","Support Vector Machine","Decision Tree","Naïve Bayes","Logistic Regression"), each=2),
                           Metric = c("Accuracy", "Misclassification Error"), 
                           Values = c(accuracyloan_rf,misclassRateloan_rf,accuracyloan_svm,misclassRateloan_svm,accuracyloan_dt,misclassRateloan_dt,accuracyloan_nb,misclassRateloan_nb,accuracyloan_lr,misclassRateloan_lr ))
metrics_loan
#bar plot comparing metrics
# Use position=position_dodge()
ggplot(data=metrics_loan, aes(x=Metric, y=Values, fill=ML_Algorithm)) +
  geom_bar(stat="identity", position=position_dodge())

#----------------------------------------------------------------------------------------------|
# Calculate Sensitivity, Precision,  F1 Score 
Sensitivityloan_rf <- 18357/(18357+15) # TP/(TP + FN)
precisionloan_rf <- 18357/(18357+3523) # TP/(TP + FP)
F1Scoreloan_rf <- (2*precisionloan_rf*Sensitivityloan_rf)/(precisionloan_rf+Sensitivityloan_rf)

# Calculate Sensitivity, Precision,  F1 Score 
Sensitivityloan_svm <- 18372/(18372+0) # TP/(TP + FN)
precisionloan_svm <- 18372/(18372+3544) # TP/(TP + FP)
F1Scoreloan_svm <- (2*precisionloan_svm*Sensitivityloan_svm)/(precisionloan_svm+Sensitivityloan_svm)

# Calculate Sensitivity, Precision,  F1 Score 
Sensitivityloan_dt <- 18372/(18372+0) # TP/(TP + FN)
precisionloan_dt <- 18372/(18372+3545) # TP/(TP + FP)
F1Scoreloan_dt <- (2*precisionloan_dt*Sensitivityloan_dt)/(precisionloan_dt+Sensitivityloan_dt)

# Calculate Sensitivity, Precision,  F1 Score 
Sensitivityloan_nb <- 6198/(6198+12174) # TP/(TP + FN)
precisionloan_nb <- 6198/(6198+1292) # TP/(TP + FP)
F1Scoreloan_nb <- (2*precisionloan_nb*Sensitivityloan_nb)/(precisionloan_nb+Sensitivityloan_nb)

# Calculate Sensitivity, Precision,  F1 Score 
Sensitivityloan_lr <- 18372/(18372+0) # TP/(TP + FN)
precisionloan_lr <- 18372/(18372+3546) # TP/(TP + FP)
F1Scoreloan_lr <- (2*precisionloan_lr*Sensitivityloan_lr)/(precisionloan_lr+Sensitivityloan_lr)

# COmpare the metrics
metrics2_loan <- data.frame(ML_Model = rep(c("Random Forest","Support Vector Machine","Decision Tree","Naïve Bayes","Logistic Regression"), each=3),
                            Metric = c("Sensitivity", "Precision", "F1 Score"), 
                            Values = c(Sensitivityloan_rf,precisionloan_rf,F1Scoreloan_rf,Sensitivityloan_svm,precisionloan_svm,F1Scoreloan_svm,
                                       Sensitivityloan_dt,precisionloan_dt,F1Scoreloan_dt,Sensitivityloan_nb,precisionloan_nb,F1Scoreloan_nb,
                                       Sensitivityloan_lr,precisionloan_lr,F1Scoreloan_lr ))
metrics2_loan

#bar plot comparing metrics
# Use position=position_dodge()
ggplot(data=metrics2_loan, aes(x=Metric, y=Values, fill=ML_Model)) +
  geom_bar(stat="identity", position=position_dodge())
