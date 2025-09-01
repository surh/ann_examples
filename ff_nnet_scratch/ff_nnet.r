library(tidyverse)

#' Simulate two datasets, a simple one and a harder one
set.seed(654765)
m <- 1000
noise <- 0.01
Dat1 <- tibble(x1 = runif(n = m),
              x2 = runif(n = m)) %>%
  mutate(y = ifelse(x1 + rnorm(n = m, sd = noise) >= x2 + rnorm(n = m, sd = noise), 
                    1, 0))
Dat1 %>%
  ggplot(aes(x = x1, y = x2)) +
  geom_point(aes(col = factor(y)))

Dat2 <- tibble(x1 = runif(n = m),
               x2 = runif(n = m)) %>%
  mutate(y = ifelse((x1 + rnorm(n = m, sd = noise) >= 0.5 & x2 + rnorm(n = m, sd = noise) > 0.5) |
                      (x1 + rnorm(n = m, sd = noise) < 0.5 & x2 + rnorm(n = m, sd = noise) < 0.5), 
                    1, 0))
Dat2 %>%
  ggplot(aes(x = x1, y = x2)) +
  geom_point(aes(col = factor(y)))

#' Try binomial model on both datasets
m1.binom <- glm(y ~ x1 + x2, data = Dat1, family = binomial(link = "logit"))
Dat1 %>%
  ggplot(aes(x = x1, y = x2)) +
  geom_point(aes(col = factor(y))) +
  geom_abline(intercept = -coef(m1.binom)[1] / coef(m1.binom)[2],
              slope = -coef(m1.binom)[3] / coef(m1.binom)[2],
              linewidth = 3, linetype = "dashed")


m2.binom <- glm(y ~ x1 + x2, data = Dat2, family = binomial(link = "logit"))
Dat2 %>%
  ggplot(aes(x = x1, y = x2)) +
  geom_point(aes(col = factor(y))) +
  geom_abline(intercept = -coef(m2.binom)[1] / coef(m2.binom)[2],
              slope = -coef(m2.binom)[3] / coef(m2.binom)[2],
              linewidth = 3, linetype = "dashed")


#' # Feed forward neural network

#' Define cost
cross_entropy_binom_cost <- function(y, y_hat){
  
  loss <- sum(-(y * log(y_hat) + (1-y)*log(1-y_hat)))
  return(loss)
}

#' Define activation function
g_logistic <- function(x){
  a <- 1 / (1 + exp(-x))
  
  return(a)
}

#' Feed forward over three layers
ff_nnet <- function(A_0, W, B, g = match.fun(g_logistic)){
    # layer 1
    Z_1 <- W$W_1 %*% A_0 + matrix(B$B_1, nrow = nrow(W$W_1), ncol = ncol(A_0)) # w/broadcasting
    A_1 <- g(Z_1)
    
    # layer 2
    Z_2 <- W$W_2 %*% A_1 + matrix(B$B_2, nrow = nrow(W$W_2), ncol = ncol(A_1))
    A_2 <- g(Z_2)
    
    # layer 3
    Z_3 <- W$W_3 %*% A_2 + matrix(B$B_3, nrow = nrow(W$W_3), ncol = ncol(A_2))
    A_3 <- g(Z_3)
    
    hidden <- list(A_0 = A_0,
                   A_1 = A_1,
                   A_2 = A_2)
    
    return(list(Y_hat = A_3, hidden = hidden)) 
}

# Y_hat <- ff_nnet(A_0 = A_0, W = W, B = B, g = g_logistic)
# cross_entropy_binom_cost(Y, ff_nnet(A_0 = A_0, W = W, B = B, g = g_logistic)$Y_hat)

#' Do backpropagation by layer
backpropagation <- function(Y_hat, Y, m, A_2, A_1, A_0, W_3, W_2, W_1){
  # Layer 3
  A_3 <- Y_hat
  
  dC_dZ3 <- (1/m) * (A_3 - Y)
  dZ3_dW3 <- A_2
  dC_dW3 <- dC_dZ3 %*% t(dZ3_dW3)
  
  dZ3_dA2 <- W_3 
  dC_dA2 <- t(W_3) %*% dC_dZ3
  
  dC_dB3 <- matrix(rowSums(dC_dZ3), ncol = 1)
  
  # Layer 2
  dA2_dZ2 <- A_2 * (1 - A_2)
  dC_dZ2 <- dC_dA2 * dA2_dZ2
  
  dZ2_dW2 <- A_1
  dC_dW2 <- dC_dZ2 %*% t(dZ2_dW2)
  
  dZ2_dA1 <- W_2
  dC_dA1 <- t(W_2) %*% dC_dZ2
  
  dC_dB2 <- matrix(rowSums(dC_dW2), ncol = 1)
  
  # Layer 1
  dA1_dZ1 <- A_1 * (1 - A_1)
  dC_dZ1 <- dC_dA1 * dA1_dZ1
  
  dZ1_dW1 <- A_0
  dC_dW1 <- dC_dZ1 %*% t(dZ1_dW1)
  
  dC_dB1 <- matrix(rowSums(dC_dW1), ncol = 1)
  
  return(list(dC_dW3 = dC_dW3, 
              dC_dW2 = dC_dW2,
              dC_dW1 = dC_dW1,
              dC_dB3 = dC_dB3,
              dC_dB2 = dC_dB2,
              dC_dB1 = dC_dB1))
}


#' Setup data
Dat <- Dat2
A_0 <- t(Dat[,c("x1", "x2")])
Y <- matrix(Dat$y, nrow = 1)

#' Define and Initialize network
n = c(2, 5, 5, 1) # Nodes per layer (and input)
W <- list(W_1 = matrix(rnorm(n = n[2] * n[1]), nrow = n[2], ncol = n[1]),
          W_2 = matrix(rnorm(n = n[3] * n[2]), nrow = n[3], ncol = n[2]),
          W_3 = matrix(rnorm(n = n[4] * n[3]), nrow = n[4], ncol = n[3]))
B <- list(B_1 = matrix(rnorm(n = n[2]), nrow = n[2], ncol = 1),
          B_2 = matrix(rnorm(n = n[3]), nrow = n[3], ncol = 1),
          B_3 = matrix(rnorm(n = n[4]), nrow = n[4], ncol = 1))

#' Setup training
epochs <- 20000 # training for 1000 iterations
lr <- 0.5 # set learning rate to 0.1
Cost <- NULL

for(epoch in 1:epochs){
  # Feed forward
  Y_hat <- ff_nnet(A_0 = A_0, W = W, B = B, g = g_logistic)
  
  # Cost (save cost history)
  cost <- cross_entropy_binom_cost(y = Y, y_hat = Y_hat$Y_hat)
  Cost <- c(Cost, cost)
  
  
  Grad <- backpropagation(Y_hat = Y_hat$Y_hat, 
                          Y = Y, 
                          m = m, 
                          A_2 = Y_hat$hidden$A_2,
                          A_1 = Y_hat$hidden$A_2,
                          A_0 = A_0,
                          W_3 = W$W_3,
                          W_2 = W$W_2,
                          W_1 = W$W_1)
  
  
  # Update weights
  W$W_3 <- W$W_3 - (lr * Grad$dC_dW3)
  W$W_2 <- W$W_2 - (lr * Grad$dC_dW2)
  W$W_1 <- W$W_1 - (lr * Grad$dC_dW1)
  
  # Update biases
  B$B_3 <- B$B_3 - (lr * Grad$dC_dB3)
  B$B_2 <- B$B_2 - (lr * Grad$dC_dB2)
  B$B_1 <- B$B_1 - (lr * Grad$dC_dB1)
  
  if(epoch %% 100 == 0){
    cat(paste0("Cost in epoch ", epoch, " is: ", cost, "\n"))
  }
  
}

plot(1:length(Cost), Cost)
Y_hat <- ff_nnet(A_0 = A_0, W = W, B = B, g = g_logistic)
summary(as.vector(Y_hat$Y_hat))
table(obs = as.vector(Y), pred = as.vector(round(Y_hat$Y_hat)))

Dat %>%
  mutate(y_hat = as.vector(Y_hat$Y_hat)) %>%
  mutate(y_pred = round(y_hat)) %>%
  mutate(res = "correct_red") %>%
  mutate(res = replace(res, y == 1 & y_pred == 1, "correct_blue")) %>%
  mutate(res = replace(res, y == 0 & y_pred == 0, "correct_red")) %>%
  mutate(res = replace(res, y == 0 & y_pred == 1, "incorrect_red")) %>%
  mutate(res = replace(res, y == 1 & y_pred == 0, "incorrect_blue")) %>%
  ggplot(aes(x = x1, y = x2)) +
  # geom_point(aes(col=factor(y)))
  geom_point(aes(col = res)) +
  scale_color_manual(values = c("blue", "red", "lightblue", "lightpink"))
