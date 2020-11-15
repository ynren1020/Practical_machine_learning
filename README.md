# Practical_machine_learning final project

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Download Data
```
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url, "./pml-training.csv")
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url, "./pml-testing.csv")
```

## Read in Data
```
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
dim(training)
dim(testing)
head(training)
unique(training$user_name)
names(training)
training <- training[,-1]
```

## Load required packages
```
library(caret)
library(rattle)

```

## Preprocess 
```
set.seed(2020)
# remove NA or ""columns
discard <- which(colSums(is.na(training) |training=="")>0.9*dim(training)[1]) 
training.sub <- training[,-discard]

# remove no variance variables
nsv <- nearZeroVar(training.sub, saveMetrics = TRUE)
head(nsv) 

# plot predictors
featurePlot(x = training.sub[,c("roll_belt", "pitch_belt", "yaw_belt")],
            y = training.sub$classe,
            plot = "pairs")

# PCA (dimention reduction)
preProc <- preProcess(training.sub[,-59], method = "pca", thresh = 0.8)
trainPC <- predict(preProc, training.sub[,-59])
trainPC <- cbind(training.sub$classe, trainPC)
names(trainPC)[1] <- "classe"
testing.sub <- testing[,names(testing)%in%names(training.sub)]
testPC <- predict(preProc, testing.sub)

```

## Fit models
### Tree based prediction
```
trControl <- trainControl(method = "cv", number = 3)
model1 <- train(classe ~ ., 
                method = "rpart", 
                data = trainPC,
                trControl = trControl)
print(model1$finalModel)
# prediction
res1 <- predict(model1,testPC)
res1
# plot 
fancyRpartPlot(model1$finalModel)
# with bagging
predictors <- trainPC[,-c(1:4)]
class <- trainPC$classe
treebag <- bag(predictors, class, B = 10,
               bagControl = bagControl(fit = ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))

res.treebag <- predict(treebag, testPC[-c(1:3)])
res.treebag
```

## random forest 
### Random forest takes an unusual long time....
```
model2 <- train(classe ~ ., 
                data = trainPC, 
                method = "rf", 
                prox = TRUE,
                trControl = trControl)
#getTree(model2$finalModel, k =2)
res2 <- predict(model2, testPC)
res2
```

## Boosting with tree
### take too long ...
```
model3 <- train(classe ~ ., 
                data = trainPC, 
                method = "gbm", 
                verbose = FALSE)
print(model3)
res3 <- predict(model3, testPC)
res3
```

## model based prediction

```
# lda
model4 <- train(classe ~ ., 
                data = trainPC, 
                method = "lda",
                trControl = trControl)
print(model4)
res4<- predict(model4, testPC)
res4
```

# Agreement among different methods
```
table(res1, res.treebag, res4)
output.df <- data.frame(tree=res1, treebag = res.treebag, lda = res4)
output.df
irr::kappam.light(output.df[, 1:3])
```