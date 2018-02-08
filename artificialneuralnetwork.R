#  Creating an Artificial Neural Network

# Splitting iris dataset training and testing data
set.seed(1)
n <- dim(iris)[1]
rows <- sample(1:n, 0.8*n)
train <- iris[rows,]
test <- iris[-rows,]

# Scaling all numeric variables in training data so their values range from 0 to 1
colmax <- apply(train[,1:4], 2, max)
colmin <- apply(train[,1:4], 2, min)
X_train <- t((t(train[,1:4]) - colmin)/(colmax - colmin))
# Using same min and max values to standardize test data
X_test <- t((t(test[,1:4]) - colmin)/(colmax - colmin))

# Checking that all input variables are properly scaled
summary(X_train)

# Turning the Species category into a vector of 0's and 1's
library(dplyr)
train <- mutate(train,
                setosa = ifelse(Species == 'setosa', 1, 0),
                versicolor = ifelse(Species == 'versicolor', 1, 0),
                virginica = ifelse(Species == 'virginica', 1, 0)) 
y_train <- train[,c('setosa','versicolor','virginica')]

test <- mutate(test,
               setosa = ifelse(Species == 'setosa', 1, 0),
               versicolor = ifelse(Species == 'versicolor', 1, 0),
               virginica = ifelse(Species == 'virginica', 1, 0))
y_test <- test[,c('setosa','versicolor','virginica')]

## Setting up our network

# We will be using the sigmoid function as our activation function
sigmoid <- function(Z){ 1/(1 + exp(-Z)) }
sigmoidprime <- function(z){ exp(-z)/((1+exp(-z))^2) }

# There are 4 nodes in the input layer since we have 4 input variables
# We have two nodes in our hidden layer, which is arbitarily chosen
# We have 2 bias values for the hidden layer and 3 for the output layer

input_layer_size <- 4
output_layer_size <- 3
hidden_layer_size <- 2

# Setting some inital weights
# For weights applied to input layer values
W_1 <- matrix(runif(input_layer_size * hidden_layer_size)-.5, nrow = input_layer_size, ncol = hidden_layer_size)
# For weights applied to hidden layer values
W_2 <- matrix(runif(hidden_layer_size * output_layer_size)-.5, nrow = hidden_layer_size, ncol = output_layer_size)

# Setting bias matrices
# For weights applied before activation in hidden layer
B1 <- matrix(runif(hidden_layer_size), ncol = 1)
# For weights applied before activation in output layer
B2 <- matrix(runif(output_layer_size), ncol = 1)

# Forward propagation
n <- nrow(train)
Z_2 <- X_train %*% W_1
A_2 <- sigmoid(Z_2 + t(B1 %*% rep(1,n)))
Z_3 <- A_2 %*% W_2
y_hat <- sigmoid(Z_3 + t(B2 %*% rep(1,n)))

# The initial prediction for y is not very good
print(y_hat)

# Creating a placeholder for y-hat
y_hat <- matrix(rep(0,3*n),nrow=n)

# Applying gradient descent to train our neural network
for(i in 1:2000){
  # Forward propagation
  Z_2 <- X_train %*% W_1
  A_2 <- sigmoid(Z_2 + t(B1 %*% rep(1,n)))
  Z_3 <- A_2 %*% W_2
  y_hat <- sigmoid(Z_3 + t(B2 %*% rep(1,n)))
  
  # Back propagation
  delta_3 <- -as.matrix((y_train - y_hat) * sigmoidprime(Z_3 + t( B2 %*% rep(1,n) ))) 
  djdb2 <- rep(1, n) %*% delta_3
  djdw2 <- t(A_2) %*% delta_3
  delta_2 <- delta_3 %*% t(W_2) * sigmoidprime(Z_2 + t( B1 %*% rep(1,n) ) )
  djdb1 <- rep(1, n) %*% delta_2
  djdw1 <- t(X_train) %*% delta_2
  
  # Setting constant value for learning rate of gradient descent
  scalar <- 0.0125
  
  # Gradient descent parameter update
  W_1 <- W_1 - scalar * djdw1
  B2  <- B2  - scalar * t(djdb2)
  W_2 <- W_2 - scalar * djdw2
  B1  <- B1  - scalar * t(djdb1)
}

# Applying neural network to test data 
n_test <- nrow(test)
Z_2 <- X_test %*% W_1
A_2 <- sigmoid(Z_2 + t(B1 %*% rep(1,n_test)))
Z_3 <- A_2 %*% W_2
y_pred <- sigmoid(Z_3 + t(B2 %*% rep(1,n_test)))

# Quantifying performance of network
highest_prob <- apply(y_pred,1,which.max) # obtains index with highest value
species_categories <- c('setosa','versicolor','virginica')
predictions <- species_categories[highest_prob]
actual <- species_categories[test$Species] 
sum(predictions == test$Species)/nrow(test) # calculates percentage of correct predictions
sum(predictions != actual) # calculates total number of incorrect predictions

# Our neural network correctly predicted the species for every observation
cbind(round(y_pred,5), predictions, actual) 
