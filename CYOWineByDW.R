##############################################################################
#                                                                            #
# Author: David Wong                                                         #
# Course: HarvardX: PH125.9x Data Science: Capstone                          #
# Course provider: edX                                                       #
# Assignment: "Choose-Your-Own" CYO Project Using Wine Quality Data from UCI #
# When: June 2020                                                            #
#                                                                            #
##############################################################################

#######################################################################################
#                                                                                     #
# *** CAUTION ***                                                                     #
#                                                                                     #
# The whole script usually takes around 90 minutes to complete running                # 
# (and sometimes up to 2 hours 30 minutes) depending on the resource availability     #
# during execution and the overall hardware configuration of the machine.             #
#                                                                                     #
# The following is the specification of the machine used for reference:               #
#                                                                                     #
# CPU: 1.6 GHz Dual-Core Intel Corei5                                                 #
# Memory: 16GB 2133 MHz LPDDR3                                                        #
#                                                                                     #
#######################################################################################

#########################################
#                                       #
# Preparation of packages and libraries #
#                                       #
#########################################

# Install packages automatically if required
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(kknn)) install.packages("kknn", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org") #for kknn too
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

# Load libraries required
library(tidyverse)
library(dplyr)
library(caret)
library(corrplot)
library(kknn)
library(randomForest)
library(kernlab)
library(gridExtra)

####################
#                  #
# Data preparation #
#                  #
####################

# Download Red Wine data file from UCI and load into data frame
redwine_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
redwine_raw <- read.csv(redwine_url, header = TRUE, sep = ";")
redwine <- redwine_raw

# Download White Wine data file from UCI and load into data frame
whitewine_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
whitewine_raw <- read.csv(whitewine_url, header = TRUE, sep = ";")
whitewine <- whitewine_raw

######################################
#                                    #
# Data Exploration and Visualisation #
#                                    #
######################################

# Confirm summary data matches with information from UCI website
str(redwine)
str(whitewine)

# Visually insepct the data
head(redwine)
head(whitewine)

# Check for "NA"s
sum(is.na(redwine))
sum(is.na(whitewine))

# Analyse distribution of quality values using histogram
par(mfrow=c(1,2))
hist(redwine$quality, main="Red Wine", xlab = "Quality", col = "red")
hist(whitewine$quality, main="White Wine", xlab = "Quality", col = "white")
par(mfrow=c(1,1))

# Analyse distribution of quality values using table to show the actual figures
table(redwine$quality)
table(whitewine$quality)

# Plot Quality against each of the 11 input variables
# On Red Wine
redwineplots <- list()  # Create a list to store plots
for (i in 1:11) {
  p1 <- eval(substitute(
    ggplot(data=redwine,aes(x=redwine[ ,i], y=redwine[,"quality"])) +
      geom_point(alpha = 0.8, size =1) +
      geom_smooth(method = "lm", se = FALSE, size = 1) +
      labs(y="Red Wine Quality", x=names(redwine)[i])
    ,list(i = i)))
  redwineplots[[i]] <- p1  # add each plot into plot list
}
# Present plots
do.call(grid.arrange, c(redwineplots, ncol = 4))

# On White Wine
whitewineplots <- list()  # Create a list to store plots
for (i in 1:11) {
  p1 <- eval(substitute(
    ggplot(data=whitewine,aes(x=whitewine[ ,i], y=whitewine[,"quality"])) +
      geom_point(alpha = 0.8, size =1) +
      geom_smooth(method = "lm", se = FALSE, size = 1) +
      labs(y="White Wine Quality", x=names(whitewine)[i])
    ,list(i = i)))
  whitewineplots[[i]] <- p1  # add each plot into plot list
}
# Present plots
do.call(grid.arrange, c(whitewineplots, ncol = 4))

# Plot correlation matrix using number method to display correlation coefficient
cor_redwine <- cor(redwine)
corrplot(cor_redwine, number.cex=0.7, method = 'number', title="Red Wine Correlation Matrix", mar=c(0,0,1,0))
cor_whitewine <- cor(whitewine)
corrplot(cor_whitewine, number.cex=0.7, method = 'number', title = "White Wine Correlation Matrix", mar=c(0,0,1,0))

###################################################
#                                                 #
# Model Building - Red Wine                       #
#                                                 #
# Note:                                           #
# Use caret package's train function              #
# Use all and standardize input variables         #
# Repeated cross-validation 5 times using 5 folds #
#                                                 #
###################################################

# Prepare training and test data sets
redwine$quality <- as.factor(redwine$quality)
set.seed(2006, sample.kind="Rounding") #Use Year - Month - day of today's date, 2020-06-08, the first day this script is drafted
train_index_redwine <- createDataPartition(redwine$quality, p = 0.8, list = F)
train_set_redwine <- redwine[train_index_redwine,]
test_set_redwine <- redwine[-train_index_redwine,]

# Model 1 - k-nearest neighbours on Red Wine
# 5 kmax, 2 distances, 3 kernels

# Create a log to record execution time
run_log <- tibble(Activity = "KNN Red Start:", Time = Sys.time())

control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
grid_kknn <- expand.grid(kmax = c(3, 5, 7, 9, 11), distance = c(1, 2),
                         kernel = c("rectangular", "cos", "gaussian"))
train_kknn_redwine <- train(quality ~ ., data = train_set_redwine, method = "kknn",
                    trControl = control, tuneGrid = grid_kknn,
                    preProcess = c("center", "scale"))
plot(train_kknn_redwine)
train_kknn_redwine$bestTune

# Record execution end-time
run_log <- bind_rows(run_log,tibble(Activity = "KNN Red End:", Time = Sys.time()))

# Model 2 - Random Forest on Red Wine
# mtry from 1 through 11

# Record execution start-time
run_log <- bind_rows(run_log,tibble(Activity = "RF Red Start:", Time = Sys.time()))

control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
grid_rf <- expand.grid(mtry = 1:11)
train_rf_redwine <- train(quality ~ ., data = train_set_redwine, method = "rf",
                  trControl = control, tuneGrid = grid_rf, 
                  preProcess = c("center", "scale"))
plot(train_rf_redwine)
train_rf_redwine$bestTune

# Record execution end-time
run_log <- bind_rows(run_log,tibble(Activity = "RF Red End:", Time = Sys.time()))

# Model 3 - Support Vector Machine on Red Wine
# The Radial Basis Function (svmRadial) used
# 10 sigma values between 0.1 and 1
# Cost: 4 values between 2 and 16

# Record execution start-time
run_log <- bind_rows(run_log,tibble(Activity = "SVM Red Start:", Time = Sys.time()))

grid_svm <- expand.grid(C = 2^(1:4), sigma = seq(0.1, 1, length = 10))
train_svm_redwine <- train(quality ~ ., data = train_set_redwine, method = "svmRadial",
                   trControl = control, tuneGrid = grid_svm,
                   preProcess = c("center", "scale"))
plot(train_svm_redwine)
train_svm_redwine$bestTune

# Record execution end-time
run_log <- bind_rows(run_log,tibble(Activity = "SVM Red End:", Time = Sys.time()))

#####################################################
#                                                   #
# Model Building - White Wine                       #
#                                                   #
# Note:                                             #
# Use caret package's train function                #
# Use all and standardize input variables           #
# Repeated cross-validation 5 times using 5 folds   #
#                                                   #
#####################################################

# Prepare training and test data sets
whitewine$quality <- as.factor(whitewine$quality)
set.seed(2006, sample.kind="Rounding") #Use Year - Month - day of today's date, 2020-06-08, the first day this script is drafted
train_index_whitewine <- createDataPartition(whitewine$quality, p = 0.8, list = F)
train_set_whitewine <- whitewine[train_index_whitewine,]
test_set_whitewine <- whitewine[-train_index_whitewine,]

# Model 1 - k-nearest neighbours on White Wine
# 5 kmax, 2 distances, 3 kernels

# Record execution start-time
run_log <- bind_rows(run_log,tibble(Activity = "KNN White Start:", Time = Sys.time()))

control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
grid_kknn <- expand.grid(kmax = c(3, 5, 7, 9, 11), distance = c(1, 2),
                         kernel = c("rectangular", "cos", "gaussian"))
train_kknn_whitewine <- train(quality ~ ., data = train_set_whitewine, method = "kknn",
                            trControl = control, tuneGrid = grid_kknn,
                            preProcess = c("center", "scale"))
plot(train_kknn_whitewine)
train_kknn_whitewine$bestTune

# Record execution end-time
run_log <- bind_rows(run_log,tibble(Activity = "KNN White End:", Time = Sys.time()))

# Model 2 - Random Forest on White Wine
# mtry from 1 through 11

# Record execution start-time
run_log <- bind_rows(run_log,tibble(Activity = "RF White Start:", Time = Sys.time()))

control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
grid_rf <- expand.grid(mtry = 1:11)
train_rf_whitewine <- train(quality ~ ., data = train_set_whitewine, method = "rf",
                          trControl = control, tuneGrid = grid_rf, 
                          preProcess = c("center", "scale"))
plot(train_rf_whitewine)
train_rf_whitewine$bestTune

# Record execution end-time 
run_log <- bind_rows(run_log,tibble(Activity = "RF White End:", Time = Sys.time()))

# Model 3 - Support Vector Machine on White Wine
# The Radial Basis Function (svmRadial) used
# 8 sigma values between 0.25 and 2
# Cost: 3 values between 2 and 8

# Record execution start-time
run_log <- bind_rows(run_log,tibble(Activity = "SVM White Start:", Time = Sys.time()))

grid_svm <- expand.grid(C = 2^(1:3), sigma = seq(0.25, 2, length = 8))
train_svm_whitewine <- train(quality ~ ., data = train_set_whitewine, method = "svmRadial",
                   trControl = control, tuneGrid = grid_svm,
                   preProcess = c("center", "scale"))
plot(train_svm_whitewine)
train_svm_whitewine$bestTune

# Record execution end-time
run_log <- bind_rows(run_log,tibble(Activity = "SVM White End:", Time = Sys.time()))

#################
#               #
# Model Testing #
#               #
#################

# Model 1 - k-nearest neighbours (Red)
predict_kknn_redwine <- predict(train_kknn_redwine, test_set_redwine)
cm_kknn_redwine <- confusionMatrix(predict_kknn_redwine, test_set_redwine$quality)
accuracy_kknn_redwine <- cm_kknn_redwine$overall['Accuracy']

# Show result
cm_kknn_redwine
accuracy_kknn_redwine

# Model 1 - k-nearest neighbours (White)
predict_kknn_whitewine <- predict(train_kknn_whitewine, test_set_whitewine)
cm_kknn_whitewine <- confusionMatrix(predict_kknn_whitewine, test_set_whitewine$quality)
accuracy_kknn_whitewine <- cm_kknn_whitewine$overall['Accuracy']

# Show result
cm_kknn_whitewine
accuracy_kknn_whitewine

# Model 2 - Random Forest (Red)
predict_rf_redwine <- predict(train_rf_redwine, test_set_redwine)
cm_rf_redwine <- confusionMatrix(predict_rf_redwine, test_set_redwine$quality)
accuracy_rf_redwine <- cm_rf_redwine$overall['Accuracy']

# Show result
cm_rf_redwine
accuracy_rf_redwine

# Model 2 - Random Forest (White)
predict_rf_whitewine <- predict(train_rf_whitewine, test_set_whitewine)
cm_rf_whitewine <- confusionMatrix(predict_rf_whitewine, test_set_whitewine$quality)
accuracy_rf_whitewine <- cm_rf_whitewine$overall['Accuracy']

# Show result
cm_rf_whitewine
accuracy_rf_whitewine

# Model 3 - SVM (Red)
predict_svm_redwine <- predict(train_svm_redwine, test_set_redwine)
cm_svm_redwine <- confusionMatrix(predict_svm_redwine, test_set_redwine$quality)

accuracy_svm_redwine <- cm_svm_redwine$overall['Accuracy']

# Show result
cm_svm_redwine
accuracy_svm_redwine

# Model 3 - SVM (White)
predict_svm_whitewine <- predict(train_svm_whitewine, test_set_whitewine)
cm_svm_whitewine <- confusionMatrix(predict_svm_whitewine, test_set_whitewine$quality)

accuracy_svm_whitewine <- cm_svm_whitewine$overall['Accuracy']

# Show result
cm_svm_whitewine
accuracy_svm_whitewine

#########################
#                       #
# Final results/Summary #
#                       #
#########################

# Consolidate results on red wine and white wine in respective lists
result_redwine <- c(accuracy_kknn_redwine, accuracy_rf_redwine, accuracy_svm_redwine)
result_whitewine <- c(accuracy_kknn_whitewine, accuracy_rf_whitewine, accuracy_svm_whitewine)

# Create a summary tibble
result_summary <- tibble(
  Model = c("KNN", "RF","SVM"),
  "Accurarcy on Red Wine" = result_redwine,
  "Accuracy on White Wine" = result_whitewine
)

# Present final results
result_summary %>% knitr::kable()

#################
#               #
# End of script #
#               #
#################

###################################################
#                                                 #
# Appendix: Execution times during model training #
#                                                 #
###################################################

run_log

# Execution times recorded in the latest model training run only 

## # A tibble: 12 x 2
## Activity Time
## <chr> <dttm>
## 1 KNN Red Start: 2020-06-10 11:18:31
## 2 KNN Red End: 2020-06-10 11:23:57
## 3 RF Red Start: 2020-06-10 11:23:57
## 4 RF Red End: 2020-06-10 11:30:04
## 5 SVM Red Start: 2020-06-10 11:30:04
## 6 SVM Red End: 2020-06-10 12:07:39
## 7 KNN White Start: 2020-06-10 12:07:39
## 8 KNN White End: 2020-06-10 12:39:43
## 9 RF White Start: 2020-06-10 12:39:43
## 10 RF White End: 2020-06-10 13:02:03
## 11 SVM White Start: 2020-06-10 13:02:03
## 12 SVM White End: 2020-06-10 13:41:55