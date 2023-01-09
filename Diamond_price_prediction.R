
suppressMessages(library(tidyverse))


# Read the file
df = read.csv(file = 'C:\\Users\\kantg\\OneDrive\\Desktop\\CMU\\Essentials of statistics\\Project 2\\diamonds.csv',
              header= TRUE,
              stringsAsFactors = TRUE)
head(df)

# Get rows, columns, summary
print(ncol(df)) # no. of columns is 11
print(nrow(df)) # no. of records is 53940

#Obtaining a summary of the columns. Looking at the summary of the columns, as well as the data, it looks like column
# X is a label/identifier and will not be useful for prediction. Thus we have 9 predictor variables and price is
# the response variable. There are 3 character variables.
summary(df)


# Removing the first column X
df <- subset(df, select = -c(X) )
names(df)[8:10] <- c("xx","yy","zz")
head(df)


## EDA

# Checking if there are any rows with null values.There are no missing values in the data
sum(is.na(df))

# Here is the correlation plot. We can see that carat, xx, yy, zz are highly correlated with each other.
# Thus, multicollinearity exists in the data set.

suppressMessages(library(corrplot))
df %>%
  dplyr::select(.,carat, depth, table, xx,yy,zz, price) %>%
  cor(.) %>%
  corrplot(.,method="shade")

# Identifying and transforming/removing outliers

# Creating facet charts for numerical variables. We can see that carat and the response variable are skewed towards
# the right. Thus, I have performed non-linear transformation for both the variables (log transformation) and the
# distribution looks more centered after transformation.

df.new <- df %>%                  
 dplyr::select(.,carat, depth, table, xx,yy,zz, price) %>% 
 gather(.)
ggplot(data=df.new,mapping=aes(x=value)) + 
  geom_histogram(color="grey",fill="purple",bins=25) +
  facet_wrap(~key,scales='free_x')

#Log transforming predictor variable carat since it was earlier right skewed. Looks more uniform now, 
# no need to remove outliers I believe. For now I have not transformed the response variable although it does look
# skewed to the right.

df$carat <- log(df$carat)
hist(df$carat)  #histogram of log transformed 'carat'

new.df = df
head(new.df)

# Running the regression model. 

# First step is to split into training and test data. I have taken the training:test split to be 70:30
# so that there is sufficient data not seen by the model for testing

set.seed(67897)
s <- sample(nrow(new.df),round(0.7*nrow(new.df)))
df.train <- new.df[s,]
df.test <- new.df[-s,]

# Linear regression with analysis (without transforming the response variable)

lm.out <- lm(price~.,data=df.train)
summary(lm.out)
price.pred <- predict(lm.out,newdata=df.test)

library(ggplot2)
ggplot(data=df.test,mapping=aes(x=price,y=price.pred)) + geom_point(color='blue',size=0.5) + labs(y = "Predicted price", x = "Price") +  geom_abline()


mse = mean((df.test$price - price.pred)^2) # Mean square error is computed for the test data set
mse
# adjusted R squared value is 0.86

# Plotting the residuals. We observe that the plot of residuals is skewed. Thus residuals are not normally distributed
# and we need to transform the response variable.

predicted_value = price.pred
actual_value = df.test$price
Residuals = predicted_value - actual_value
hist(Residuals,col='green')

#Log transforming the response variable price

new.df$price <- log(new.df$price) 
hist(new.df$price)  #histogram of log transformed response variable 'price'

# Linear regression after non-linear transformation of the response variable.

# Split into training and test data

set.seed(67897)
s <- sample(nrow(new.df),round(0.7*nrow(new.df)))
df.train <- new.df[s,]
df.test <- new.df[-s,]

# Linear regression with analysis

lm.out <- lm(price~.,data=df.train)
summary(lm.out)
price.pred <- predict(lm.out,newdata=df.test)

library(ggplot2)
ggplot(data=df.test,mapping=aes(x=price,y=price.pred)) + geom_point(color='blue',size=0.5) + labs(y = "Predicted price", x = "Price") +  geom_abline()


mse = mean((df.test$price - price.pred)^2) # Mean square error is computed for the test data set
mse
# After transforming the response variable, the adjusted R squared is 0.98. Thus this model is definitely fit better than without transformation.

# Plotting the residuals after transforming the response variable. We observe that the plot of residuals is now normally distributed.

predicted_value = price.pred
actual_value = df.test$price
Residuals = predicted_value - actual_value
hist(Residuals,col='green')

#Checking for multi collinearity using VIF

suppressMessages(library(car))
vif(lm.out)


# Best subset selection analysis of the data set

suppressMessages(library(bestglm))

bg.out <- bestglm(df.train,family=gaussian,IC="BIC")
bg.out$BestModel
summary(bg.out$BestModel)
price.pred <- predict(bg.out$BestModel,newdata=df.test)
mse = mean((df.test$price - price.pred)^2) # Mean square error is computed for the test data set
mse

# Adjusted R squared value is 0.94 which has actually dropped from the model with all predictor variables. Here we have a total of 8 predictor variables that are used for prediction.


