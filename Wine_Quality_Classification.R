
suppressMessages(library(tidyverse))


# Read the file
df = read.csv(file = 'C:\\Users\\kantg\\OneDrive\\Desktop\\CMU\\Essentials of statistics\\Project 3\\wineQuality.csv',
              header= TRUE,
              stringsAsFactors = TRUE)
head(df)

# Get rows, columns, summary
print(ncol(df)) # no. of columns is 12
print(nrow(df)) # no. of records is 6497

#Obtaining a summary of the columns. Looking at the summary of the columns, as well as the data.
summary(df)

# Checking if there are any rows with null values.There are no missing values in the data
sum(is.na(df))

#EDA

suppressMessages(library(tidyverse))

df.new <- df %>%                    
  dplyr::select(.,-label) %>% 
  gather(.)
# facet charts
ggplot(data = df.new, aes(value)) + 
  geom_histogram(color="black",fill="orange",bins = 30) +
  facet_wrap(~key, scale='free_x') + labs(x = "Feature Variables")

# fix.acid, free.sd, sugar, sulphates, total.sd, vol.acid are skewed by looking at the charts above

df$fix.acid <- log(df$fix.acid)
hist(df$fix.acid)  #histogram of log transformed fix.acid

df$free.sd <- log(df$free.sd)
hist(df$free.sd)  #histogram of log transformed free.sd

df$sugar <- log(df$sugar)
hist(df$sugar)  #histogram of log transformed sugar

df$sulphates <- sqrt(df$sulphates)
hist(df$sulphates)  #histogram of log transformed sulphates

df$total.sd <- sqrt(df$total.sd)
hist(df$total.sd) #histogram of log transformed total.sd

df$vol.acid <- log(df$vol.acid)
hist(df$vol.acid) #histogram of log transformed vol.acid


#Correlation plot

suppressMessages(library("corrplot"))

df %>%
  dplyr::select(.,-label) %>%
  cor(.) %>% corrplot(.,method="shade", order = 'AOE', type = 'lower')

#Classification Models

predictors <- df %>% select(-c("label")) 
response <- df %>% select("label")
# Split into training and test data
set.seed(35)
s = sample(nrow(predictors),round(0.7*nrow(predictors)))
predictors.train = predictors[s,]
predictors.test= predictors[-s,]
resp.train = response[s,]
resp.test= response[-s,]

df.train <-  cbind(predictors.train,resp.train)

# Logistic Regression

out.log = glm(resp.train~., data=predictors.train, family=binomial)
summary(out.log)

##Threshold
t = sum(df$label == "GOOD")/length(df$label)
resp.prob = predict(out.log,newdata=predictors.test,type="response") 
resp.pred = ifelse(resp.prob > t,"GOOD","BAD")

# Mis-classification Rate
mean(resp.pred!=resp.test)

# Confusion Matrix
tab = table(resp.pred,resp.test)
print(tab)

suppressMessages(library(pROC))
roc.log = roc(resp.test,resp.prob)
roc.log$thresholds[1:10]

# Determining AUC values

cat("AUC: ",round(roc.log$auc,4),"\n")
logreg_auc = round(roc.log$auc,4)
plot(roc.log, col="chartreuse2", xlim=c(1,0), ylim=c(0,1))

#best glm

suppressMessages(library(bestglm))

bg.out = bestglm(df.train, family=binomial,IC="BIC")
bg.out$BestModel

resp.prob = predict(bg.out$BestModel, newdata=predictors.test, type="response") 
resp.pred = ifelse(resp.prob > t,"GOOD","BAD")

# Misclassification Rate
mean(resp.pred!=resp.test)

# Confusion Matrix
tab = table(resp.pred,resp.test)
print(tab)

#Creating ROC Curve
roc.bglm = roc(resp.test,resp.prob)
roc.bglm$thresholds[1:10]

# Determining AUC values
cat("AUC: ", round(roc.bglm$auc,4), "\n")
bglm_auc = round(roc.bglm$auc,4)
plot(roc.bglm, col="chartreuse2", xlim=c(1,0), ylim=c(0,1))


#Decision Tree

suppressMessages(library(rpart))
rpart.out = rpart(resp.train~.,data=predictors.train)
summary(rpart.out)

resp.prob = predict(rpart.out, newdata=predictors.test, type="prob")[,2]
resp.pred = ifelse(resp.prob > t,"GOOD","BAD")

# Misclassification Rate
mean(resp.pred!=resp.test)

# Confusion Matrix
tab = table(resp.pred,resp.test)
print(tab)

#Creating ROC Curve
roc.dt = roc(resp.test,resp.prob)

names(roc.dt)

roc.dt$thresholds[1:10]

# Determining AUC values
cat("AUC: ", round(roc.dt$auc,4), "\n")
dt_auc = round(roc.dt$auc,4)
plot(roc.dt, col="chartreuse2", xlim=c(1,0), ylim=c(0,1))


# Random forest model

suppressMessages(library(randomForest))

rf.out = randomForest(resp.train~., data=predictors.train, importance = TRUE)
summary(rf.out)

varImpPlot(rf.out, type=1)

resp.prob = predict(rf.out, newdata=predictors.test, type="prob")[,2]
resp.pred = ifelse(resp.prob > t,"GOOD","BAD")

# Misclassification Rate
mean(resp.pred!=resp.test)

# Confusion Matrix
tab = table(resp.pred,resp.test)
print(tab)

#Creating ROC Curve
roc.rf = roc(resp.test,resp.prob)
roc.rf$thresholds[1:10]

# Determining AUC values
cat("AUC: ", round(roc.rf$auc,4), "\n")
rf_auc= round(roc.rf$auc,4)
plot(roc.rf, col="chartreuse2", xlim=c(1,0), ylim=c(0,1))

#KNN

suppressMessages(library(FNN))
k.max = 29
mcr.k = rep(NA, k.max)

for ( kk in 2:k.max ) {
  knn.out = knn.cv(train=predictors.train, cl=resp.train, k=kk, prob=TRUE)
  knn.prob = attributes(knn.out)$prob
  mcr.k[kk] = mean((knn.out != resp.train))
}
k.min = which.min(mcr.k)

ggplot(data=data.frame("k"=1:k.max,"mcr"=mcr.k),mapping=aes(x=k,y=mcr)) + 
  geom_point() + geom_line() +
  xlab("Number of Nearest Neighbors k") + ylab("Validation MCR") + 
  geom_vline(xintercept=k.min,color="red")

cat("The optimal number of nearest neighbors is ",k.min,"\n")

knn.out = knn(train=predictors.train, test=predictors.test, cl = resp.train, k=k.min, prob = TRUE)
resp.prob = attributes(knn.out)$prob
w = which(knn.out=="BAD")
resp.prob[w] = 1 - resp.prob[w]

#Creating ROC Curve
roc.knn = roc(resp.test,resp.prob)
roc.knn$thresholds[1:10]

# Determining AUC values
cat("AUC: ", round(roc.knn$auc,4), "\n")
knn_auc = round(roc.knn$auc,4)
plot(roc.knn, col="chartreuse2", xlim=c(1,0), ylim=c(0,1)) 

# Creating a consolidated table
suppressMessages(library(DT))
models <- c("Logistic Regression", "Best GLM","Random Forest", "Decision Tree", "KNN")
auc_values <- c(logreg_auc, bglm_auc, rf_auc, dt_auc, knn_auc)
new_auc_values <- cbind(models, auc_values)
datatable(new_auc_values)

# From AUC values. random forest is the best. Next, we determine the optimal class-separation threshold by maximizing
#the Youden’s J-statistic (senstivity + specicity - 1) and finding the threshold associated with that maximum value

set.seed(35)
rf.out = randomForest(resp.train~.,data=predictors.train, importance = TRUE)
resp.prob = predict(rf.out, newdata=predictors.test,type="prob")[,2] 
roc.rf = roc(resp.test,resp.prob)

# Youden's J Calculation
J = roc.rf$sensitivities + roc.rf$specificities - 1
w = which.max(J)
cat("Optimum threshold for RF: ", round(roc.rf$thresholds[w],4),"\n")
t_opt = round(roc.rf$thresholds[w],4)
resp.pred = ifelse(resp.prob > t_opt ,"GOOD","BAD") 

# Confusion Matrix
tab = table(resp.pred, resp.test)
tab

# Misclassification Rate with Optimal Threshold for Best Model
mean(resp.pred!=resp.test)

## Thus the misclassification rate has decreased from 18.5 to 17.6 in random forest and optimizaed random forest
