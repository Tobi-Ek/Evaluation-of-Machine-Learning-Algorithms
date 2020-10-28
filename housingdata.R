#Import needed libraries

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

#--------------------------------------------------------------------------------------------->

# import CSV file for housing data
housingpr_data <- read.csv("HOUSING-PRICE_Data.csv", header = TRUE)
View(housingpr_data)

#--------------------------------------------------------------------------------------------->

# EXPLORATORY DATA ANALYSIS

str(housingpr_data)
summary(housingpr_data)

# Check the distribution for continuous variables
plot_num(housingpr_data, bins=40)
# Check the distribution for catgeorial variables
freq(housingpr_data)
# Check for missing values
missmap(housingpr_data, main = "Missing vs observed values")

# Display the map plot of median house value with respect to the latitude and longitude
map_plot <- ggplot(housingpr_data, aes(x = longitude, y = latitude, hma = housing_median_age, color = median_house_value, tr = total_rooms, tb = total_bedrooms, hh = households, mi = median_income)) +
  geom_point(aes(size = population), shape = 15, alpha = 0.4) + xlab("Longitude") + ylab("Latitude") +
  ggtitle("Data Map - Longtitude vs Latitude and its Corresponding Variables") + theme(plot.title = element_text(hjust = 0.5)) +
  scale_color_distiller(palette = "Paired", labels = comma) +
  labs(color = "Median House Value (in $USD)", size = "Population")
map_plot


#--------------------------------------------------------------------------------------------->
# Data Cleaning

# Exlporing Ocean_proximity
levels(housingpr_data$ocean_proximity)

# Exempt the Island category from the variable 
housingpr_data = housingpr_data[housingpr_data$ocean_proximity != "ISLAND", ]
summary(housingpr_data$ocean_proximity)
View(housingpr_data)

# Check current Ocean_proximity plot
ggplot(housingpr_data, aes(x = factor(ocean_proximity))) + geom_bar(stat = "count", color = "black", fill = "light blue")

# Removing the 207 observations from total_bedrooms
#housingpr_data <- na.omit(housingpr_data, cols=c("total_bedrooms"))
#summary(housingpr_data)

# Taking the mean and median of the total_bedrooms
bedroom_mean = mean(housingpr_data$total_bedrooms, na.rm = TRUE)
bedroom_median = median(housingpr_data$total_bedrooms, na.rm = TRUE)

# Plotting the histogram of the total bedrooms with the mean & median
ggplot(housingpr_data, aes(x = total_bedrooms)) +
  geom_histogram(bins = 40, color = "black", fill = "green") +
  geom_vline(aes(xintercept = bedroom_mean, color = "Mean"), lwd = 1.5) +
  geom_vline(aes(xintercept = bedroom_median, color = "Median"), lwd = 1.5) +
  xlab("Total Bedrooms") +
  ylab("Frequency") +
  ggtitle("Histogram of Total Bedrooms") +
  scale_color_manual(name = "Summary Stats", labels = c("Mean", "Median"), values = c("red", "blue"))

# Impute missing values for total bedrooms
housingpr_data$total_bedrooms[is.na(housingpr_data$total_bedrooms)] = bedroom_median

# Fix the total columns - make them means
housingpr_data$mean_bedrooms = housingpr_data$total_bedrooms/housingpr_data$households
housingpr_data$mean_rooms = housingpr_data$total_rooms/housingpr_data$households
drops = c('total_bedrooms', 'total_rooms')
housingpr_data = housingpr_data[ , !(names(housingpr_data) %in% drops)]
View(housingpr_data)

#--------------------------------------------------------------------------------------------->
# Data Transformation

# Turn categoricals into booleans
categories = unique(housingpr_data$ocean_proximity)

# split the categories off
cat_housing = data.frame(ocean_proximity = housingpr_data$ocean_proximity)

for(ocean in categories){
  cat_housing[,ocean] = rep(0, times= nrow(cat_housing))
}
head(cat_housing) #New columns are generated to the right

for(k in 1:length(cat_housing$ocean_proximity)){
  cat = as.character(cat_housing$ocean_proximity[k])
  cat_housing[,cat][k] = 1
}
head (cat_housing)

cat_columns = names(cat_housing)
keep_columns = cat_columns[cat_columns != 'ocean_proximity']
cat_housing = select(cat_housing,one_of(keep_columns))
tail(cat_housing)

# Scale all numerical variables except median_house_value
dropcolumn = c('ocean_proximity','median_house_value')
housingpr_num =  housingpr_data[ , !(names(housingpr_data) %in% dropcolumn)]
head(housingpr_num)

scaled_housingpr_num = scale(housingpr_num)
head(scaled_housingpr_num)

#merge back the categorical dataframes and altered numerical variables
cleaned_housing_data = cbind(cat_housing, scaled_housingpr_num, median_house_value=housingpr_data$median_house_value)
head(cleaned_housing_data)
tail(cleaned_housing_data)
summary(cleaned_housing_data)

# Retrieve R metric (or Pearson coefficient) for all numeric variables, skipping the categoric ones.
correlation_table(cleaned_housing_data, "median_house_value")
# Only Median Income had good correletion with house value

# Looking more at the correlations between the variables of the cleaned data
corr_matrix = cor(cleaned_housing_data)
kable(t(corr_matrix))
# Visualizing the correlation
ggcorrplot(cor(cleaned_housing_data))

# Rename Columns
colnames(cleaned_housing_data)
colnames(cleaned_housing_data)[1:4] <- c("NEAR_BAY", "L_1H_OCEAN", "INLAND", "NEAR_OCEAN")

#------------------------------------------------------------------>
# Further Exploration

# For a better sense of the distribution of the nine numeric variables, we look at histograms for each of them.
par(mfrow = c(3,3))
hist(cleaned_housing_data$longitude, breaks = 20, main = "longitude", border="orange", col="purple")
hist(cleaned_housing_data$latitude, breaks = 20, main = "latitude", border="orange", col="purple")
hist(cleaned_housing_data$housing_median_age, breaks = 20, main = "housing_median_age", border="orange", col="purple")
hist(cleaned_housing_data$mean_rooms, breaks = 20, main = "mean_rooms", border="orange", col="purple")
hist(cleaned_housing_data$mean_bedrooms, breaks = 20, main = "mean_bedrooms", border="orange", col="purple")
hist(cleaned_housing_data$population, breaks = 20, main = "population", border="orange", col="purple")
hist(cleaned_housing_data$households, breaks = 20, main = "households", border="orange", col="purple")
hist(cleaned_housing_data$median_income, breaks = 20, main = "median_income", border="orange", col="purple")
hist(cleaned_housing_data$median_house_value, breaks = 20, main = "median_house_value", border="orange", col="purple")
#hist(cleaned_housing_data$NEAR_BAY , breaks = 20, main = "NEAR_BAY ", border="orange", col="purple")
#hist(cleaned_housing_data$L_1H_OCEAN, breaks = 20, main = "L_1H_OCEAN", border="orange", col="purple")
#hist(cleaned_housing_data$INLAND, breaks = 20, main = "INLAND", border="orange", col="purple")
#hist(cleaned_housing_data$NEAR_OCEAN , breaks = 20, main = "NEAR_OCEAN ", border="orange", col="purple")

# Looking at pairs to better understand the relationship between the all the variables.
pairs(cleaned_housing_data, col = "purple")

#--------------------------------------------------------------------------------------------->
# Creating Test and Train Data

# Split into Train and Testing set
# Training Set : Testing Set = 70 : 30 (randomly)
set.seed(100)
train_HPsample <- sample(nrow(cleaned_housing_data), 0.7*nrow(cleaned_housing_data), replace = FALSE)
train_house <- cleaned_housing_data[train_HPsample,]
test_house <- cleaned_housing_data[-train_HPsample,]
summary(train_house)
head(train_house)
summary(test_house)
View(test_house)


#--------------------------------------------------------------------------------------------------------------------->
# DATA MODELLING AND EVALUATION OF MODELS

#-------------------------------------------------------------------------------------------------|
# Random Forest Model

#Find the best value for mtry
bestmtry_house <- tuneRF(train_house,train_house$median_house_value, stepFactor = 1.2, improve = 0.1, trace=T, plot=T)
# mtry is 4

# Create a Random Forest model_house with default parameters
rfmodel_house <- randomForest(median_house_value ~ ., data = train_house, importance = TRUE)
rfmodel_house$importance
names(rfmodel_house)

# Check important variables
importance(rfmodel_house)        
varImpPlot(rfmodel_house)

# Predicting on test data
predHouseTest_rf <- predict(rfmodel_house, newdata = test_house, type = "class")

# Generate test scatterplot
ggplot( data = test_house, aes(x=predHouseTest_rf, y=median_house_value)) + geom_point() + geom_smooth(method=lm, se=FALSE)
# Generate evaluation metrics
test_MSE_rf <- mean(( predHouseTest_rf - test_house$median_house_value)^2)
test_MAE_rf <- mean(abs( predHouseTest_rf - test_house$median_house_value))
test_MAE_rf
test_RMSE_rf <- sqrt(test_MSE_rf)
test_RMSE_rf


#-------------------------------------------------------------------------------------------------|
# Decision Tree Model

# Create a decision tree model with default parameters
dtmodel_house <- train(median_house_value ~ ., data = train_house, method = "rpart")
dtmodel_house

# Predicting on test data
predHouseTest_dt <- predict(dtmodel_house, test_house)

# Generate test scatterplot
ggplot( data = test_house, aes(x=predHouseTest_dt, y=median_house_value)) + geom_point() + geom_smooth(method=lm, se=FALSE)
# Generate evaluation metrics
test_MSE_dt <- mean(( predHouseTest_dt - test_house$median_house_value)^2)
test_MAE_dt <- mean(abs( predHouseTest_dt - test_house$median_house_value))
test_MAE_dt
test_RMSE_dt <- sqrt(test_MSE_dt)
test_RMSE_dt


#-------------------------------------------------------------------------------------------------|
# Support Vector Machine Model

# Create a Support vector machine model with default parameters
svmmodel_house <- svm(median_house_value ~ ., data = train_house)
summary(svmmodel_house)
# Visualise the model
plot(svmmodel_house, data = train_house)

# Predicting on test data
predHouseTest_svm <- predict(svmmodel_house, test_house)

# Generate test scatterplot
ggplot( data = test_house, aes(x=predHouseTest_svm, y=median_house_value)) + geom_point() + geom_smooth(method=lm, se=FALSE)
# Generate evaluation metrics
test_MSE_svm <- mean(( predHouseTest_svm - test_house$median_house_value)^2)
test_MAE_svm <- mean(abs( predHouseTest_svm - test_house$median_house_value))
test_MAE_svm
test_RMSE_svm <- sqrt(test_MSE_svm)
test_RMSE_svm

##Tuning #HyperParameter Optimisation which helps to select the best model
#set.seed(123)
#tmodel_house <- tune(svmmodel_house, median_house_value~., data=train_house, ranges = list (epsilon = seq(0,1,0.1), cost = 2^(2:4)))
#plot(tmodel_house)
#summary(tmodel_house)
#
##best Model
#tmodel_house$best.model


#-------------------------------------------------------------------------------------------------|
# Multiple Linear Regression

lm_model_house <- lm(median_house_value ~ .-NEAR_OCEAN-INLAND-latitude-longitude-population-mean_rooms-mean_bedrooms, data = train_house)
summary(lm_model_house)
par (mfrow=c(2,2))
plot(lm_model_house)


# Predicting on test data
predHouseTest_lm <- predict(lm_model_house, test_house)
# Generate test scatterplot
ggplot( data = test_house, aes(x=predHouseTest_lm, y=median_house_value)) + geom_point() + geom_smooth(method=lm, se=FALSE)
# Generate evaluation metrics
test_MSE_lm <- mean((predHouseTest_lm - test_house$median_house_value)^2)
test_MAE_lm <- mean(abs(predHouseTest_lm - test_house$median_house_value))
test_MAE_lm
test_RMSE_lm <- sqrt(test_MSE_lm)
test_RMSE_lm


#-------------------------------------------------------------------------------------------------|
# Naive bayes model (failed)

nbmodel_house <- naiveBayes(median_house_value ~., data=train_house)
summary(nbmodel_house)

#Prediction on the dataset
predHouseTest_nb <- predict(nbmodel_house,test_house)

# Generate test scatterplot
ggplot( data = test_house, aes(x=predHouseTest_lm, y=median_house_value)) + geom_point() + geom_smooth(method=lm, se=FALSE)
# Generate evaluation metrics
test_MSE_nb <- mean((predHouseTest_nb - test_house$median_house_value)^2)
test_MAE_nb <- mean(abs(predHouseTest_nb - test_house$median_house_value))
test_MAE_nb
test_RMSE_nb <- sqrt(test_MSE_nb)


#-------------------------------------------------------------------------------------------------|
# Stochastic Gradient Boosting model (failed)

#Train model_house with preprocessing & repeated cv
gbmmodel_house <- caret::train(median_house_value ~ ., data = train_house, method = "gbm", preProcess = c("scale", "center"),
                          trControl = trainControl(method = "repeatedcv", number = 5, repeats = 3, verboseIter = FALSE), verbose = 0)
gbmmodel_house

#Prediction on the dataset
predHouseTest_gbm <- predict(nbmodel_house,test_house)
# Generate test scatterplot
ggplot( data = test_house, aes(x=predHouseTest_gbm, y=median_house_value)) + geom_point() + geom_smooth(method=lm, se=FALSE)
# Generate evaluation metrics
test_MSE_gbm <- mean((predHouseTest_gbm - test_house$median_house_value)^2)
test_MAE_gbm <- mean(abs(predHouseTest_gbm - test_house$median_house_value))
test_MAE_gbm
test_RMSE_gbm <- sqrt(test_MSE_gbm)


#-------------------------------------------------------------------------------------------------------|
#comparing metrics

metrics_housing <- data.frame(ML_Algorithm = rep(c("Random Forest", "Support vector Machine", "Multiple Linear Regression","Decision Tree" ), each=2),
                              Metric = c("MAE", "RMSE"), 
                              Values = c(test_MAE_rf,test_RMSE_rf,test_MAE_svm,test_RMSE_svm,test_MAE_lm,test_RMSE_lm,test_MAE_dt,test_RMSE_dt ))
metrics_housing
#bar plot comparing metrics
# Use position=position_dodge()
ggplot(data=metrics_housing, aes(x=Metric, y=Values, fill=ML_Algorithm)) +
  geom_bar(stat="identity", position=position_dodge())




#Deleting Dat variables
data$date =NULL
#Deleting zero values

library(dplyr)
data<-data%>%filter(Sales>0)

#one-hot Encoding()

data_ohe <- as.data.frame(model_house.matrix(~.-1, data=train))
ohe_label <- data.ohe[,'Sales'] # Target variables - Sales


#EXTRAS
#To generate barplot of Importance
feature_Importance <- data.frame(importance(rfmodel_house))
colnames(feature_Importance)[1:2] <-  c("Feature", "Percentage_IncMSE")
ggplot(data = feature_Importance) + geom_bar(mapping = aes(x = Percentage_IncMSE))


