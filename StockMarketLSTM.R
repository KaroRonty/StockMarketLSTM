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
epochs <- 100

# Get a specified amount of rows and select columns
data <- full_data %>% 
  slice(I(nrow(full_data) - (train_size + test_size + lag_n * 2 + 1)):
          nrow(full_data)) %>% 
  mutate(dates = as.Date(paste0(dates, "-01"))) %>%
  #mutate(P = log(P)) %>% 
  select(dates, P)

# Split into training and test sets ----
train_unscaled <- data %>% 
  # + 1 because of the differencing
  slice(lag_n:I(train_size + lag_n * 2 + 1))

# Take the differences
train_differenced <- train_unscaled %>% 
  mutate(P = c(NA, diff(P)) / P)

# Slice the test data
test_unscaled <- data %>% 
  slice(I(train_size + lag_n + 1):
          I(train_size + test_size + lag_n * 2 + 1)) 

# Min-max scale the training set
train <- train_differenced %>% 
  mutate(P = ((P - min(P, na.rm = TRUE)) /
                (max(P, na.rm = TRUE) -
                   min(P, na.rm = TRUE)))) %>% 
  slice(2:nrow(train_unscaled))

# Min-max scale the test set by according to the training set
test <- test_unscaled %>% 
  # Take the differences
  mutate(P = c(NA, diff(P)) / P) %>% 
  mutate(P = ((P - min(train_differenced$P, na.rm = TRUE)) /
                (max(train_differenced$P, na.rm = TRUE) -
                   min(train_differenced$P, na.rm = TRUE)))) %>% 
  slice(2:nrow(test_unscaled))

# Make lagged variables for the training set
train_lagged <- c()
for(i in 1:lag_n){
  train_lagged <- cbind(train_lagged, lag(train$P, i)[I(lag_n + 1):
                                                    I(train_size + lag_n)])
  colnames(train_lagged)[i] <- paste0("lag_", i)
}

# Make arrays for the training set with the dimensions required by tensorflow
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

# Make arrays for the test set with the dimensions required by tensorflow
test_x <- array(test_lagged,
                dim = c(nrow(test_lagged), lag_n, 1))
test_y <- array(data = test$P[I(lag_n + 1):I(test_size + lag_n)],
                dim = c(nrow(train[I(lag_n + 1):I(test_size + lag_n), ]), 1))

# Make the model ----
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
          optimizer = optimizer_adam(lr = 1e-5),
          metrics = "mae")

# Print the loss and accuracy while training the model
history <- model %>% fit(x = train_x,
                         y = train_y,
                         batch_size = batch_size,
                         epochs = epochs,
                         verbose = 1,
                         validation_split = 0.1,
                         validation_data = list(test_x, test_y),
                         shuffle = FALSE)
plot(history)

# Make predictions using both the training and test set and combine them ----
train_pred <- model %>% 
  predict(train_x, batch_size = batch_size) %>% .[, 1]

test_pred <- model %>%
  predict(test_x, batch_size = batch_size) %>% .[, 1]

pred <- c(train_pred, test_pred)

# Function for unscaling the prices
unscale <- function(x){
  (max(train_differenced$P, na.rm = TRUE) -
     min(train_differenced$P, na.rm = TRUE)) * x +
    min(train_differenced$P, na.rm = TRUE)
}
# Calculate the cumulative price for the predictions
pred_unscaled <- first(train_unscaled$P) * cumprod(unscale(pred) + 1)

# Format the training and test set dates
train_dates <- train %>%
  # Delete the first obeservation with a NA value
  slice(-1) %>% 
  pull(dates) %>% 
  lead(lag_n) %>% 
  na.omit() %>% 
  as.Date()

test_dates <- test %>% 
  pull(dates) %>% 
  lead(lag_n) %>% 
  na.omit() %>% 
  as.Date()

# Make a data frame containing the actuals and predictions
res <- rbind(data.frame(train %>% 
                          slice(I(lag_n + 2 + 1):nrow(train_unscaled)),
                        data = "train"),
             data.frame(test %>% 
                          slice(I(lag_n + 1 + 1):nrow(test_unscaled)),
                        data = "test"),
             data.frame(dates = c(train_dates, test_dates),
                        P = pred,
                        data = "pred"))
# Make a data frame containing the correctly scaled actuals and predictions
res2 <- rbind(data.frame(train_unscaled %>% 
                           slice(I(lag_n + 2 + 1):nrow(train_unscaled)),
                         data = "train"),
              data.frame(test_unscaled %>% 
                           slice(I(lag_n + 1 + 1):nrow(test_unscaled)),
                         data = "test"),
              data.frame(dates = c(train_dates, test_dates),
                         P = pred_unscaled,
                         data = "pred"))

# Plot the actuals and predictions
res %>% 
  ggplot(aes(x = dates, y = P, color = data)) +
  geom_line()

# Plot the correctly scaled actuals and predictions
res2 %>% 
  ggplot(aes(x = dates, y = P, color = data)) +
  geom_line()