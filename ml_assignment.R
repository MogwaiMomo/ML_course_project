setwd(dirname(parent.frame(2)$ofile))


# MACHINE LEARNING ASSIGNMENT

### In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to identify the class of movement they are performing. Here are the 5 classes of movement: 

# A - Correct movement
# B - Throwing elbows to the front
# C - Lifting only halfway
# D - Lowering only halfway
# E - Throwing hips to the front


### STEP 0: LOAD REQUIRED LIBRARIES

library(caret)
library(ggplot2)


### STEP 1: LOAD THE DATA

raw_train <- read.table("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", sep=",", header=T, na.strings=c("NA"," ", ""))

raw_test <- read.table("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", sep=",", header=T, na.strings=c("NA"," ", ""))

### STEP 2: CLEAN THE DATA

# First we take a look at the data to see what we're dealing with:

dim(raw_train)
head(raw_train)
str(raw_train)

# It appears that several columns in the data set have lots of NAs in them. Let's find out how prevalent these NAs — and by extension, whether we should remove them — by calculating the % of NAs for each variable. 

colMeans(is.na(raw_train))

# Variables in this data set appear to either be completely full (NAs = 0) or nearly empty (NAs = 0.98%). Obviously keeping variables that are 98% empty would be problematic, so let's remove them from both the test and training datasets: 

fullVars <- names(which(colMeans(is.na(raw_train)) == 0))
fullVars2 <- names(which(colMeans(is.na(raw_test)) == 0))

training <- raw_train[,fullVars] 
testing <- raw_test[,fullVars2] 

# Let's also remove non-quantitative variables such as timestamps, usernames, etc., since our model needs to classify an observed movement based on accelerometer measurements alone:

training <- training[,-(1:7)]
testing <- testing[,-(1:7)]

# We now have reasonably clean data sets to build our model with. 

### STEP 3: SPLIT THE DATA

# The n of our training set is quite large (~19,000), so let's break off some data to use as a validation set for checking our model and estimating our out-of-sample error before using it to classify observations from our testing set: 

set.seed(123)
splitIndex <- createDataPartition(training[,"classe"], p=.75, list=F, times=1)
train <- training[splitIndex,]
test <- training[-splitIndex,]


### STEP 4: PRELIMINARY MODELING: LDA

# Our goal for this project is to predict what type of movement is being observed based on a range of quantitative predictors. In other words, we need to build a classification model as opposed to a regression model. 

# For classification problems with a multi-class outcome (as opposed to a binary yes/no outcome), 2 standard options to try are Linear or Quadratic Discrimant Analysis and K-Nearest Neighbours. 

# First, let's try LDA, out of the box, using the default cross-validation settings on the trianing data and centering & scaling the data using the pre-processing function: 
set.seed(213)
lda_fit <- train(classe ~ .,
                 data = train,
                 method = "lda", 
                 preProcess=c("center", "scale"), 
                 trControl = trainControl(method = "cv"))

lda_fit

# Our model gives in-sample accuracy of ~0.7. Decent, but not great, especially considering in-sample accuracy will likely be overly optimistic compared to out-of-sample accuracy. 

# LDA assumes Gaussian distributions of predictor variables and common co-variance, and we don't know this is the case, which may be why the accuracy is subpar. Let's try a non-parametric classifier like K-Nearest-Neighbours and see what we get. 

### STEP 5: NON-PARAMETRIC MODELING: KNN

# KNN is an extremely low-bias/highly-flexible option, but may result in over-fitting the data. In addition, while it can be highly accurate in predictions, it's generally pretty hard to get any interpretation of relationships from it. But we're only interested in getting accurate predictions, so let's give it a shot. 

# Note that we also use principal components analysis (PCR) to reduce the dimensionality of our predictors in this model and hopefully get rid of some unnecessary noise. PCA pre-processing scales and centers the data as well. 
set.seed(435)
knn_fit <- train(classe ~ .,
                 data = train,
                 method = "knn", 
                 preProcess=c("pca"), 
                 trControl = trainControl(method = "cv"))
knn_fit

# We now get in-sample accuracy estimates of ~0.96 with k = 5, which is a HUGE improvement over LDA. Let's further validate our model by running predictions against the validation set we created.  

fit_classes1 <- predict(knn_fit, newdata = test)

# To calculate overall prediction accuracy and the out-of-sample error rate, we generate a Confusion Matrix:

Conf_Matrix1 <- confusionMatrix(data = fit_classes1, test$classe)
Conf_Matrix1

# Our out-of-sample error rate is only ~4% (i.e. 1-Accuracy). That should be good enough to pass our prediction test. That being said, this model does not give us much with which to INTERPRET the relationships between the predictors and the outcome. We can interpret accurately, but we have no idea how we're doing it!


### STEP 6 (BONUS): Generalized Boosted Regression Models

# GBM models are great because they combine high predictive accuracy with highly interpretable results. The plot() function allows you to actually visually see the importance of certain variables over others. The one disadvantage is that a GBM Model can be computationally expensive and take time to run. 

set.seed(555)
gbm_fit <- train(classe ~ .,
  data = train,
  method = "gbm",
  trControl = trainControl(method = "cv"),
  verbose = F
)

# The in-sample accuracy of this model is only marginally better than KNN (0.961 vs 0.956). Let's test it with our validation set: 

fit_classes2 <- predict(gbm_fit, newdata = test)
Conf_Matrix2 <- confusionMatrix(fit_classes2, test$classe)
Conf_Matrix2

# Again, our out-of-sample accuracy is only marginally better than what we got with KNN (0.9613 vs 0.9592). 

# But now let's dig into our gbm model a bit to understand it: 

var_importance <- summary(gbm_fit)
tail(var_importance, 10)

# Wow, there are a lot of predictors we've included in our model that have ZERO influence on its predictive power. Now if we look at the most influential variables: 

head(var_importance, 10)

# We can see that just the top few variables — namely roll_belt, pitch_forearm, yaw_belt, and a few others, have by far the biggest influence on the model. If we wanted to in the future, we could try creating simpler multiple regression models to quantify the mean difference in these predictor variables between classes and get a more granular understanding of what constitutes "good" and "bad" weight-lifting form. 


