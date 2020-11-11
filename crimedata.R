# Import needed libraries

library(funModeling) 
library(tidyverse)
library(reshape2)
library(Hmisc)
library(ggplot2)
library(randomForest)
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
library(e1071)
library(ISLR)
library(xgboost)
library(plyr)
library(gbm)
library(ROCR)
library(rpart) #decison tree
library(rpart.plot) #decision tree

#--------------------------------------------------------------------------------------------->
#Getting the Data

# import CSV file for Crime data
crime_data <- read.csv("CRIME_CPW-BHW_Data.csv", header = TRUE)
View(crime_data)

#--------------------------------------------------------------------------------------------->
# EXPLORATORY DATA ANALYSIS

str(crime_data)
# Removing unused attribute
crime_data <- crime_data[,-3]
# Convert crimeType data type
crime_data$Crime_Type <- as.factor(crime_data$Crime_Type)

colnames(crime_data)
summary(crime_data)

# Check the distribution for catgeorial variables
freq(crime_data)
# Check the distribution for continuous variables
plot_num(crime_data, bins=40)
# Checking for missing values
missmap(crime_data, main = "Missing vs Observed values")
sapply(crime_data, function(x) sum(is.na(x))) # An alternative method
# checking the factor varoables
contrasts(crime_data$AREA.NAME)


#------------------------------------------------------------------>
# Data Cleaning and transformation

# Turn categoricals of Area Name into booleans
categories = unique(crime_data$AREA.NAME)
categories2 = unique(crime_data$Vict_Descent)

# Modifying the area name variable
# split the categories off
cat_crime = data.frame(AREA.NAME = crime_data$AREA.NAME)

for(i in categories){
  cat_crime[,i] = rep(0, times= nrow(cat_crime))
}
head(cat_crime) #New columns are generated to the right

for(k in 1:length(cat_crime$AREA.NAME)){
  coke = as.character(cat_crime$AREA.NAME[k])
  cat_crime[,coke][k] = 1
}
head (cat_crime)

cat_columns = names(cat_crime)
keep_columns = cat_columns[cat_columns != 'AREA.NAME']
cat_crime = select(cat_crime,one_of(keep_columns))
tail(cat_crime)

# Modifying Vict_descent variable
# split the categories off
cat_crime2 = data.frame(Vict_Descent = crime_data$Vict_Descent)

for(j in categories2){
  cat_crime2[,j] = rep(0, times= nrow(cat_crime2))
}
head(cat_crime2) #New columns are generated to the right

for(x in 1:length(cat_crime2$Vict_Descent)){
  coke = as.character(cat_crime2$Vict_Descent[x])
  cat_crime2[,coke][x] = 1
}
head (cat_crime2)

cat_columns2 = names(cat_crime2)
keep_columns2 = cat_columns2[cat_columns2 != 'Vict_Descent']
cat_crime2 = select(cat_crime2,one_of(keep_columns2))
tail(cat_crime2)


#merge back the categorical dataframes and altered numerical variables
cleaned_crime_data = cbind(crime_data,cat_crime2,cat_crime)
cleaned_crime_data <- cleaned_crime_data[,-8]
cleaned_crime_data <- cleaned_crime_data[,-5]
head(cleaned_crime_data)
str(cleaned_crime_data)

contrasts(cleaned_crime_data$Crime_Type)

# Rename the column names
colnames(cleaned_crime_data)[7:12] <- c("Hispanics", "Blacks", "Whites", "CentralArea", "PacificArea", "WestValleyArea")
str(cleaned_crime_data)
summary(cleaned_crime_data)
cleaned_crime_data
View(cleaned_crime_data)


# Change data type of cleaned data to factor
str(cleaned_crime_data)
cleaned_crime_data$Hispanics <- as.factor(cleaned_crime_data$Hispanics)
cleaned_crime_data$Blacks <- as.factor(cleaned_crime_data$Blacks)
cleaned_crime_data$Whites <- as.factor(cleaned_crime_data$Whites)
cleaned_crime_data$CentralArea <- as.factor(cleaned_crime_data$CentralArea)
cleaned_crime_data$PacificArea <- as.factor(cleaned_crime_data$PacificArea)
cleaned_crime_data$WestValleyArea <- as.factor(cleaned_crime_data$WestValleyArea)

freq(cleaned_crime_data)
#------------------------------------------------------------------>
# Further Exploration

cross_plot(cleaned_crime_data, input="Vict_Sex", target="Crime_Type")
cross_plot(cleaned_crime_data, input="Hispanics", target="Crime_Type")
cross_plot(cleaned_crime_data, input="Blacks", target="Crime_Type")
cross_plot(cleaned_crime_data, input="Whites", target="Crime_Type")
cross_plot(cleaned_crime_data, input="CentralArea", target="Crime_Type")
cross_plot(cleaned_crime_data, input="PacificArea", target="Crime_Type")
cross_plot(cleaned_crime_data, input="WestValleyArea", target="Crime_Type")


#------------------------------------------------------------------>
# Creating Test and Train Data

# Split into Train and Testing set
# Training Set : Testing Set = 70 : 30 (randomly)
set.seed(150)
train_CRsample <- sample(nrow(cleaned_crime_data), 0.7*nrow(cleaned_crime_data), replace = FALSE)
train_crime <- cleaned_crime_data[train_CRsample,]
test_crime <- cleaned_crime_data[-train_CRsample,]
summary(train_crime)
View(train_crime)
summary(test_crime)
View(test_crime)

#--------------------------------------------------------------------------------------------------------------------->
# DATA MODELLING AND EVALUATION OF MODELS

#-------------------------------------------------------------------------------------------------|
# Random Forest Model

#Find the best value for mtry
bestmtry_crime <- tuneRF(train_crime,train_crime$Crime_Type, stepFactor = 1.2, improve = 0.1, trace=T, plot=T)
# mtry is 3

# Create a Random Forest model_crime with default parameters
rfmodel_crime <- randomForest(Crime_Type ~ ., data = train_crime, importance = TRUE)

# checking the Importance of the variables
importance(rfmodel_crime)
varImpPlot(rfmodel_crime)

# Predicting on test data
predCrimeTest_rf <- predict(rfmodel_crime, newdata = test_crime, type = "class")
# Check confusion Matrix
tabcrime_rf <- table(Actual = test_crime$Crime_Type, Predicted = predCrimeTest_rf)
tabcrime_rf
# Checking classification accuracy
accuracycrime_rf <- mean(predCrimeTest_rf == test_crime$Crime_Type)
accuracycrime_rf
# Check misclassification Error
misclassRatecrime_rf <- 1-sum(diag(tabcrime_rf))/sum(tabcrime_rf)
misclassRatecrime_rf


#-------------------------------------------------------------------------------------------------|
# Decision Tree Model

# Create a decision tree model_crime with default parameters
dtmodel_crime = train(Crime_Type ~ ., data = train_crime, method = "rpart")
dtmodel_crime

# Predicting on test data
predCrimeTest_dt <- predict(dtmodel_crime, newdata = test_crime)
# Check confusion Matrix
tabcrime_dt <- table(Actual = test_crime$Crime_Type, Predicted = predCrimeTest_dt)
tabcrime_dt
# Checking classification accuracy
accuracycrime_dt <- mean(predCrimeTest_dt == test_crime$Crime_Type)
accuracycrime_dt
# Check misclassification Error
misclassRatecrime_dt <- 1-sum(diag(tabcrime_dt))/sum(tabcrime_dt)
misclassRatecrime_dt


#-------------------------------------------------------------------------------------------------|
# Support Vector Machine Model

# Create a Support Vector Machine model with default parameters
svmmodel_crime <- svm(Crime_Type ~ ., data = train_crime)
summary(svmmodel_crime)
# Visualise the model
plot(svmmodel_crime, data = train_crime)

# Predict with Test Data
predcrimeTest_svm <- predict(svmmodel_crime, test_crime)

# Check Confusion matrix
tabcrime_svm <- table(Actual = test_crime$Crime_Type, Predicted = predcrimeTest_svm)
tabcrime_svm
# Checking classification accuracy
accuracycrime_svm <- mean(predcrimeTest_svm == test_crime$Crime_Type)
accuracycrime_svm
# Check misclassification Error
misclassRatecrime_svm <- 1-sum(diag(tabcrime_svm))/sum(tabcrime_svm)
misclassRatecrime_svm

# Tuning #HyperParameter Optimisation which helps to select the best model
set.seed(123)
tmodel_crime <- tune(svmmodel_crime, Crime_Type~., data=train_crime, ranges = list (epsilon = seq(0,1,0.1), cost = 2^(2:4)))
plot(tmodel_crime)
summary(tmodel_crime)
# best Model
tmodel_crime$best.model


#-------------------------------------------------------------------------------------------------|
# Logistic Regression Model

lrmodel_crime <- glm(Crime_Type ~., family=binomial(link='logit'), data=train_crime)
# Check the summary of the model_crime
summary(lrmodel_crime)

# Inspecting for correlation in the model_crime
alias(lrmodel_crime)

# Predict with Test Data
predcrimeTest_lr <- predict(lrmodel_crime, newdata=test_crime, type='response')
# Assessing the predictive ability of the model
predcrimeTest_lr <- ifelse(predcrimeTest_lr > 0.5,1,0)

# Check Confusion matrix
tabcrime_lr <- table(Actual = test_crime$Crime_Type, Predicted = predcrimeTest_lr)
tabcrime_lr
# Checking classification accuracy
accuracycrime_lr <- mean(predcrimeTest_lr == test_crime$Crime_Type)
accuracycrime_lr
# Check misclassification Error
misclassRatecrime_lr <- 1-sum(diag(tabcrime_lr))/sum(tabcrime_lr)
misclassRatecrime_lr
# Checking classification accuracy
#misClassificError <- mean(predcrimeTest_lr != test_crime$Crime_Type)
#print(paste('Accuracy: ', 1-misClassificError))

# analyze the table of deviance using anova function
anova(lrmodel_crime, test="Chisq")

# To get the McFadden r-squared index
library(pscl)
pR2(lrmodel_crime)
# Plot the ROC curve -- performance measurements for a binary classifier
predcrimeTest_lr2<- predict(lrmodel_crime, newdata=test_crime, type="response")
prcrime_lr <- prediction(predcrimeTest_lr2, test_crime$Crime_Type)
ROC_crime <- performance(prcrime_lr, measure = "tpr", x.measure = "fpr")
plot(ROC_crime)

# calculate the AUC (area under the curve) -- performance measurements for a binary classifier
auc_crime <- performance(prcrime_lr, measure = "auc")
auc_crime <- auc_crime@y.values[[1]]
auc_crime


#-------------------------------------------------------------------------------------------------|
# Naive Bayes Model

# Create a Support vector machine model with default parameters
nbmodel_crime <- naiveBayes(Crime_Type ~ ., data = train_crime)
summary(nbmodel_crime)
# Visualise the model
plot(nbmodel_crime, data = train_crime)

# Predict with Test Data
predcrimeTest_nb <- predict(nbmodel_crime, test_crime)

# Check Confusion Matrix
tabcrime_nb <- table(Actual = test_crime$Crime_Type, Predicted = predcrimeTest_nb)
tabcrime_nb
# Check classification Accuracy
accuracycrime_nb <- mean(predcrimeTest_nb == test_crime$Crime_Type)
accuracycrime_nb
# Check misclassification Error
misclassRatecrime_nb <- 1-sum(diag(tabcrime_nb))/sum(tabcrime_nb)
misclassRatecrime_nb


#-----------------------------------------------------------------------------------------------------------------------|
#comparing metrics

metrics_crime <- data.frame(ML_Algorithm = rep(c("Random Forest","Support Vector Machine","Decision Tree","Naïve Bayes","Logistic Regression"), each=2),
                            Metric = c("Accuracy", "Misclassification Error"), 
                            Values = c(accuracycrime_rf,misclassRatecrime_rf,accuracycrime_svm,misclassRatecrime_svm,accuracycrime_dt,misclassRatecrime_dt,accuracycrime_nb,misclassRatecrime_nb,accuracycrime_lr,misclassRatecrime_lr ))
metrics_crime

#bar plot comparing metrics
# Use position=position_dodge()
ggplot(data=metrics_crime, aes(x=Metric, y=Values, fill=ML_Algorithm)) +
  geom_bar(stat="identity", position=position_dodge())

#----------------------------------------------------------------------------------------------|
# Calculate Sensitivity, Precision,  F1 Score 
Sensitivitycrime_rf <- 6938/(6938+7342) # TP/(TP + FN)
precisioncrime_rf <- 6938/(6938+4336) # TP/(TP + FP)
F1Scorecrime_rf <- (2*precisioncrime_rf*Sensitivitycrime_rf)/(precisioncrime_rf+Sensitivitycrime_rf)

# Calculate Sensitivity, Precision,  F1 Score 
Sensitivitycrime_svm <- 6211/(6211+8069) # TP/(TP + FN)
precisioncrime_svm <- 6211/(6211+4483) # TP/(TP + FP)
F1Scorecrime_svm <- (2*precisioncrime_svm*Sensitivitycrime_svm)/(precisioncrime_svm+Sensitivitycrime_svm)

# Calculate Sensitivity, Precision,  F1 Score 
Sensitivitycrime_nb <- 5403/(5403+8877) # TP/(TP + FN)
precisioncrime_nb <- 5403/(5403+3950) # TP/(TP + FP)
F1Scorecrime_nb <- (2*precisioncrime_nb*Sensitivitycrime_nb)/(precisioncrime_nb+Sensitivitycrime_nb)

# Calculate Sensitivity, Precision,  F1 Score 
Sensitivitycrime_dt <- 4325/(4325+9955) # TP/(TP + FN)
precisioncrime_dt <- 4325/(4325+2943) # TP/(TP + FP)
F1Scorecrime_dt <- (2*precisioncrime_dt*Sensitivitycrime_dt)/(precisioncrime_dt+Sensitivitycrime_dt)

# Calculate Sensitivity, Precision,  F1 Score 
Sensitivitycrime_lr <- 7151/(7151+7129) # TP/(TP + FN)
precisioncrime_lr <- 7151/(7151+5443) # TP/(TP + FP)
F1Scorecrime_lr <- (2*precisioncrime_lr*Sensitivitycrime_lr)/(precisioncrime_lr+Sensitivitycrime_lr)

# Compare metrics
metrics2_crime <- data.frame(ML_Model = rep(c("Random Forest","Support Vector Machine","Decision Tree","Naïve Bayes", "Logistic Regression"), each=3),
                             Metric = c("Sensitivity", "Precision", "F1 Score"), 
                             Values = c(Sensitivitycrime_rf,precisioncrime_rf,F1Scorecrime_rf,Sensitivitycrime_svm,precisioncrime_svm,F1Scorecrime_svm,
                                        Sensitivitycrime_dt,precisioncrime_dt,F1Scorecrime_dt,Sensitivitycrime_nb,precisioncrime_nb,F1Scorecrime_nb,
                                        Sensitivitycrime_lr,precisioncrime_lr,F1Scorecrime_lr ))
metrics2_crime

#bar plot comparing metrics
# Use position=position_dodge()
ggplot(data=metrics2_crime, aes(x=Metric, y=Values, fill=ML_Model)) +
  geom_bar(stat="identity", position=position_dodge())



