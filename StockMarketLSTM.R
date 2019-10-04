library(ggplot2)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(keras)
library(httr) # downloading the xls(x) files
library(readxl) # reading xls(x) files
library(dplyr) # data formatting
source("https://raw.githubusercontent.com/KaroRonty/ShillerGoyalDataRetriever/master/ShillerGoyalDataRetriever.r")

# Set the training set and test set size
train_size <- 1000
test_size <- 500

# Set the amount of lags in months, the batch size and epochs
lag_n <- 12 * 5
batch_size <- 3
epochs <- 2

# Get a specified amount of rows and select columns
data <- full_data %>% 
    slice((nrow(full_data) - (train_size + test_size) + 1):nrow(full_data)) %>% 
    mutate(dates = as.Date(paste0(dates, "-01"))) %>%
    select(dates, P)

# Split into training and test sets ----
train_unscaled <- data %>% 
    slice(1:train_size)

test_unscaled <- data %>% 
    slice((train_size + 1):(train_size + test_size))

# Difference training and test set values
# TODO: percentages instead of plain differences
train_differenced <- diff(train_unscaled %>% pull(P))
test_differenced <- diff(test_unscaled %>% pull(P))

# Functions for normalizing and denormalizing values
normalize <- function(x) {
    (x - min(train_differenced, na.rm = TRUE)) /
        (max(train_differenced, na.rm = TRUE) -
             min(train_differenced, na.rm = TRUE))
}

denormalize <- function(x) {
    x * (max(train_differenced, na.rm = TRUE) -
             min(train_differenced, na.rm = TRUE)) +
        min(train_differenced, na.rm = TRUE)
}

# Normalize the training and test sets
train_differenced <- normalize(train_differenced)
test_differenced <- normalize(test_differenced)

# Make lagged variables and put the values to matching positions
train_x <- t(sapply(1:(length(train_differenced) - lag_n),
                    function(x) train_differenced[x:(x + lag_n - 1)]))

train_y <- sapply((lag_n + 1):(length(train_differenced)),
                  function(x) train_differenced[x])

test_x <- t(sapply(1:(length(test_differenced) - lag_n),
                   function(x) test_differenced[x:(x + lag_n - 1)]))

test_y <- sapply((lag_n + 1):(length(test_differenced)),
                 function(x) test_differenced[x])

# Set the dimensions as required by TensorFlow
dim(train_x) <- c(nrow(train_x), ncol(train_x), 1)
dim(test_x) <- c(nrow(test_x), ncol(test_x), 1)

# Make the model ----
model <- keras_model_sequential()

model %>% 
    layer_lstm(units = 4, input_shape = c(lag_n, 1)) %>% 
    layer_dense(units = 1)

model %>% 
    compile(loss = "mse",
            optimizer = "adam",
            metrics = "mae")

# Print the loss and accuracy while training the model
history <- model %>% fit(x = train_x,
                         y = train_y,
                         batch_size = batch_size,
                         epochs = epochs,
                         verbose = 1)
plot(history)

# Make predictions using both the training and test set and combine them ----
pred_train <- predict(model, train_x, batch_size = 1)
pred_test <- predict(model, test_x, batch_size = 1)

# Denormalize the training and test sets
pred_train <- denormalize(pred_train)
pred_test <- denormalize(pred_test)

# Revert the differencing
pred_train_undiff <- pred_train +
    train_unscaled %>% 
    slice((lag_n + 1):(dim(train_unscaled)[1] - 1)) %>% 
    pull(P)

pred_test_undiff <- pred_test +
    test_unscaled %>% 
    slice((lag_n + 1):(dim(test_unscaled)[1] - 1)) %>% 
    pull(P)

# Make a data frame containing the correctly scaled actuals and predictions
res <- tibble(time_id = c(train_unscaled$dates, test_unscaled$dates),
              train = c(train_unscaled$P, rep(NA, dim(test_unscaled)[1])),
              test = c(rep(NA, dim(train_unscaled)[1]), test_unscaled$P),
              pred_train = c(rep(NA, lag_n + 1),
                             pred_train_undiff,
                             rep(NA, dim(test_unscaled)[1])),
              pred_test = c(rep(NA, dim(train_unscaled)[1]),
                            rep(NA, lag_n+1),
                            pred_test_undiff))

# Gather for plotting
res2 <- res %>%
    gather(key = 'data', value = 'P', train:pred_test)

# Plot the actuals and predictions
ggplot(res2, aes(x = time_id,
                 y = P)) +
    geom_line(aes(color = data))