# setwd("/Users/sur/lab/exp/2025/today/")
set.seed(89034)
N <- 10

for(i in 1:N){
  
  n <- rpois(n = 1, lambda = 7)
  adj <- matrix(rpois(n*n, lambda = 2), nrow = n, ncol = n) * matrix(sample(c(1,-1), size = n*n, replace = TRUE), nrow = n, ncol = n)
  targets <- matrix(rbinom(n=n, size = 1, prob = 0.2), ncol = 1)
  
  outfile <- file.path("dummy_data", paste0("graph", i, "_adj.csv"))
  write.table(adj, file = outfile, sep = ",", col.names = FALSE, row.names = FALSE)
  outfile <- file.path("dummy_data", paste0("graph", i, "_targets.csv"))
  write.table(targets, file = outfile, sep = ",", col.names = FALSE, row.names = FALSE)
}

