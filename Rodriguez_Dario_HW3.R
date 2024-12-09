##################################################
# ECON 418-518 Homework 2
# Dario Rodriguez 
# The University of Arizona
# drodriguez10@arizona.edu
# 08 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table, margins, sandwich, lmtest, tidyr, stringr, dplyr, 
               car, knitr, tidyverse, ISLR2, glmnet, randomForest, caret)

# Set seed
set.seed(418518)


#####################
# Problem 1
#####################
# Extract the data into data table from file
dt <- read.csv("ECON_418-518_HW3_Data.csv", header = TRUE, sep = "," )

# Ensure the data is in the right format
dt <- data.table(dt)

#################
# Question (i)
#################
# Code
# removing columns from dt 
dt <- dt %>% select(-fnlwgt, -occupation, -relationship, -capital.gain,
                    -capital.loss, -educational.num)

#################
# Question (ii)
#################
# Cursory data cleaning, making many variables a binary measure
##############
# Part (a)
##############
# Convert income to a binary indicator, ">50k" = 1, use ifelse() statements
dt[, income := ifelse(income == ">50K", 1, 0)]

##############
# Part (b)
##############
# Convert race to binary indicator : white = 1, else = 0
dt[, race := ifelse(race == "White", 1, 0)]

##############
# Part (c)
##############
# Convert gender to binary indicator : male = 1, else = 0
dt[, gender := ifelse(gender == "Male", 1, 0)]

##############
# Part (d)
##############
# Convert workclass to binary indicator : private = 1, else = 0
dt[, workclass := ifelse(workclass == "Private", 1, 0)]

##############
# Part (e)
##############
# Convert native.country to binary indicator : United-States = 1, else = 0
dt[, native.country := ifelse(native.country == "United-States", 1, 0)]

##############
# Part (f)
##############
# Convert marital.status to binary indicator : Married-civ-spouse = 1, else = 0
dt[, marital.status := ifelse(marital.status == "Married-civ-spouse", 1, 0)]

##############
# Part (g)
##############
# Convert education to binary indicator : Masters | Bachelors | Doctorate = 1, else = 0
dt[, education := ifelse(education == "Bachelors" | education == "Masters"
                         | education == "Doctorate", 1, 0)]

##############
# Part (h)
##############
# Crate an age_sq variable based on the ages, add this to dt
dt[, age_sq := age^2]

##############
# Part (i)
##############
# Standardize the age, age_sq, and hours/week variables (x - mean(x)/ sd(x))
dt[,':='(
  age_std = (age - mean(age)) / (sd(age)),
  age_sq_std = (age_sq - mean(age_sq)) / (sd(age_sq)),
  hours.per.week_std = 
    (hours.per.week - mean(hours.per.week)) / (sd(hours.per.week))
)]

#################
# Question (iii)
#################
# Create a summary of the data table to understand some cursory information
summary(dt)
paste0("There are ", nrow(dt), " observations in dt.")

##############
# Part (a)
##############

# Count the proportion of individuals with over 50k annual incomes
sum(dt$income)
mean(dt$income)
paste0("There are ", sum(dt$income), 
       " individuals who make over $50k per year, a proportion of ", 
       mean(dt$income)*100, "%")

##############
# Part (b)
##############
# Find the proportion of individuals in the private sector
sum(dt$workclass)
mean(dt$workclass)
paste0("There are ", sum(dt$workclass), 
       " individuals working in the private sector, a proportion of ", 
       mean(dt$workclass)*100, "%")

##############
# Part (c)
##############
# Count the proportion of married individuals
sum(dt$marital.status)
mean(dt$marital.status)
paste0("There are ", sum(dt$marital.status), 
       " married individuals, a proportion of ", 
       mean(dt$marital.status)*100, "%")

##############
# Part (d)
##############
# Count the proportion of females
sum(dt$gender == 0)
mean(dt$gender == 0)
paste0("There are ", sum(dt$gender == 0), 
       " females, a proportion of ", 
       mean(dt$gender == 0)*100, "%")

##############
# Part (e)
##############
# Pull the # of observations in dt
paste0("There are ", nrow(dt), " observations in dt.")

# Pull the number of NAs in dt in comparison to how much isn't missing
paste0("There are ", sum(is.na(dt)), " NAs in dt")
paste0("There are ", sum(rowSums(!is.na(dt)) > 0), 
       " total non-missing rows in dt")

##############
# Part (f)
##############
# Convert the income variable to a factor data type so it is discrete
dt[,income := factor(dt$income, levels = c(0,1), labels = c("<=50k/yr", ">50k/yr"))]
str(dt$income)

#################
# Question (iv)
#################
# Split into a 70:30 train/testing set split
##############
# Part (a)
##############
# Use floor to find the last training set observation value
train.size <- floor(0.7 * nrow(dt))
train.size

##############
# Part (b)
##############
# Create the training data table to be the 1st row until the last training value
shuffle.index <- sample(nrow(dt))
train.index <- shuffle.index[1:train.size]
train.dt <- dt[train.index, ]

##############
# Part (c)
##############
# Create the test data table after the last training set data point to end of dt
test.size <- ceiling(0.3 * nrow(dt))
test.size
test.index <- shuffle.index[(train.size + 1):nrow(dt)]
test.dt <- dt[test.index, ]

#################
# Question (v)
#################
##############
# Part (b)
##############
# Estimate a LASSO regression model on income using train
# Each model will use 10-fold cross-validation, 50 evenly spaced values: 10^5 - 10^-2
lambda.grid <- 10^seq(5,-2, length = 50)
train.control <- trainControl(method = "cv", number = 10)
lasso.model <- train(
  income ~ .,
  data = train.dt,
  method = "glmnet",
  trControl = train.control,
  tuneGrid = expand.grid(alpha = 1, lambda = lambda.grid),
  metric = "Accuracy"
)

# Display the result of the LASSO 
print(lasso.model)

##############
# Part (c)
##############
# Evaluate the accuracy and return the lambda that gives the highest accuracy
# pull the bestTune and results slots of the model object
best.lambda <- lasso.model$bestTune$lambda
best.accuracy <- lasso.model$results$Accuracy[
  which(lasso.model$results$lambda == best.lambda)
]

paste0("The value of λ that gives the highest classification accuracy: ", best.lambda)
paste0("The highest classification accuracy: ", best.accuracy)

##############
# Part (d)
##############
# Which variables have coefficient estimates that are approximately 0
# Since I used train(), I find these by coef(cv$finalModel, s = cv$bestTune$lambda)
coef(lasso.model$finalModel, s = lasso.model$bestTune$lambda)

##############
# Part (e)
##############
# Pull the non-zero coefficients out of the prior command
selected.vars <- c("income", "age",
                   "education", "marital.status", "hours.per.week")

# Subset data to only include the non-zero variables from prior
train.subset <- train.dt[, selected.vars, with = FALSE]

# Develop the LASSO model with the newly selected variables
lasso.model2 <- train(
  income ~ .,
  data = train.subset,
  method = "glmnet",
  trControl = train.control,
  tuneGrid = expand.grid(alpha = 1, lambda = lambda.grid),
  metric = "Accuracy"
)

# Train a new Ridge model using the selected variables
ridge.model <- train(
  income ~ .,
  data = train.subset,
  method = "glmnet",
  trControl = train.control,
  tuneGrid = expand.grid(alpha = 0, lambda = lambda.grid),
  metric = "Accuracy"
)

# Pull and compare the results, including the best accuracy and lambda for each
print(lasso.model2)
print(lasso.model2$bestTune)
best.lambda.l2 <- lasso.model2$bestTune$lambda
best.accuracy.l2 <- lasso.model2$results$Accuracy[
  which(lasso.model2$results$lambda == best.lambda.l2)
]
paste0("The value of λ that gives the highest classification accuracy: ",
       best.lambda.l2)
paste0("The highest classification accuracy: ", best.accuracy.l2)

print(ridge.model)
print(ridge.model$bestTune)
best.lambda.r <- ridge.model$bestTune$lambda
best.accuracy.r <- ridge.model$results$Accuracy[
  which(ridge.model$results$lambda == best.lambda.r)
]
paste0("The value of λ that gives the highest classification accuracy: ",
       best.lambda.r)
paste0("The highest classification accuracy: ", best.accuracy.r)

# By using these outputs, we can also compare the two to see which has stronger accuracy
if (max(lasso.model2$results$Accuracy) > max(ridge.model$results$Accuracy)) {
  paste0("Lasso regression has the best classification accuracy.")
} else {
  paste0("Ridge regression has the best classification accuracy.")
}

#################
# Question (vi)
#################
# Lastly, we want to estimate a random forest model on income
##############
# Part (b)
##############
# Use both the randomForest and caret packages to train()
# We will do three models, with 100, 200, and 300 trees. 
# Each tree should be estimated using splits of two, five, and nine random possible features
# train with 5-fold cv
mtry.grid <- expand.grid(mtry = c(2, 5, 9))

models.rf <- list()

for (t in c(100, 200, 300))
{
  # print the current # of trees in the forest
  print(paste0(t, " trees in the forest."))
  
  # Define the model type
  model.rf <- train(
    income ~.,
    data = train.dt,
    method = "rf", 
    tuneGrid = mtry.grid,
    trControl = trainControl(method = "cv", number = 5),
    ntree = t
  )
  
  # store the model in the list
  models.rf[[paste0("ntree_", t)]] <- model.rf
  
  # show the list
  print(models.rf)
  print("---------------------------------------------------------------------")
}

##############
# Part (e)
##############
# The model with 300 trees was found to be the most accurate at mtry = 2
# Thus, we want to make a confusion matrix to make predictions
model.rf.acc <- models.rf[["ntree_300"]]
predictions <- predict(model.rf.acc, newdata = test.dt)
confusion.matrix <- confusionMatrix(predictions, test.dt[, income])
print(confusion.matrix)

#################
# Question (vii)
#################
# Make some confusion matrices on the lasso and ridge models
# Let's see how they work with the testing data
predictions2 <- predict(lasso.model2, newdata = test.dt)
predictions3 <- predict(ridge.model, newdata = test.dt)
confusion.matrix.l <- confusionMatrix(predictions2, test.dt[, income])
confusion.matrix.r <- confusionMatrix(predictions3, test.dt[, income])
print(confusion.matrix.l)
print(confusion.matrix.r)

