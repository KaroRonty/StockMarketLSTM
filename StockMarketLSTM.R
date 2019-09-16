library(dplyr)
library(keras)
library(ggplot2)
library(tensorflow)
source("https://raw.githubusercontent.com/KaroRonty/ShillerGoyalDataRetriever/master/ShillerGoyalDataRetriever.r")

# Set the training set and test set size
train_size <- 1000
test_size <- 500

# Set the amount of lags to use in months, the batch size and epochs
lag_n <-  12 * 5
batch_size <-  50
epochs <- 25

# Get a specified amount of rows, do log transformation and select columns
data <- full_data %>% 
  slice(I(nrow(full_data) - (train_size + test_size + lag_n * 2)):
          nrow(full_data)) %>% 
  mutate(dates = as.Date(paste0(dates, "-01"))) %>%
  mutate(P = log(P)) %>% 
  select(dates, P)

# Split into training and test sets
train <- data %>% 
  slice(1:I(train_size + lag_n))

test <- data %>% 
  slice(I(train_size + lag_n + 1):
          I(train_size + test_size + lag_n * 2)) %>% 
  # Scale the test set by according to the training set
  mutate(P = scale(P, center = mean(train$P), scale = sd(train$P)) %>% 
           as.vector())

# Scale the training set
train <- train %>% 
  mutate(P = scale(P) %>% 
           as.vector())

# Make lagged variables for the training set
train_lagged <- c()
for(i in 1:lag_n){
  train_lagged <- cbind(train_lagged, lag(train$P, i)[I(lag_n + 1):
                                                        I(train_size + lag_n)])
  colnames(train_lagged)[i] <- paste0("lag_", i)
}

# Make arrays for the training set with the dimensions needed by tensorflow
train_x <- array(train_lagged,
                 dim = c(nrow(train_lagged), lag_n, 1))
train_y <- array(data = train$P[I(lag_n + 1):I(train_size + lag_n)],
                 dim = c(nrow(train[I(lag_n + 1):I(train_size + lag_n), ]), 1))

# Make lagged variables for the test set
test_lagged <- c()
for(i in 1:lag_n){
  test_lagged <- cbind(test_lagged, lag(test$P, i)[I(lag_n + 1):
                                                     I(test_size + lag_n)])
  colnames(test_lagged)[i] <- paste0("lag_", i)
}

# Make arrays for the test set with the dimensions needed by tensorflow
test_x <- array(test_lagged,
                dim = c(nrow(test_lagged), lag_n, 1))
test_y <- array(data = test$P[I(lag_n + 1):I(test_size + lag_n)],
                dim = c(nrow(train[I(lag_n + 1):I(test_size + lag_n), ]), 1))

# Make the model
model <- keras_model_sequential()

model %>%
  layer_lstm(units = 100,
             input_shape = c(lag_n, 1),
             batch_size = batch_size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = "mse",
          optimizer = "adam",
          metrics = "mae")

# Print the loss and accuracy while training the model
history <- model %>% fit(x = train_x,
                         y = train_y,
                         batch_size = batch_size,
                         validation_data = list(test_x, test_y),
                         epochs = epochs,
                         verbose = 1,
                         shuffle = FALSE)
plot(history)

# Make predictions using both the training and test set and combine them
train_pred <- model %>% 
  predict(train_x, batch_size = batch_size) %>% .[, 1]

test_pred <- model %>%
  predict(test_x, batch_size = batch_size) %>% .[, 1]

pred <- c(train_pred, test_pred)

# Format the training and test set dates
train_dates <- train$dates %>% 
  lead(lag_n) %>% 
  na.omit() %>% 
  as.Date()

test_dates <- test$dates %>% 
  lead(lag_n) %>% 
  na.omit() %>% 
  as.Date()

# Make a data frame containing the actuals and predictions
res <- rbind(data.frame(train, data = "train"),
             data.frame(test, data = "test"),
             data.frame(dates = c(train_dates, test_dates),
                        P = pred,
                        data = "pred"))

# Plot the actuals and predictions
res %>% 
  ggplot(aes(x = dates, y = P, color = data)) +
  geom_line()