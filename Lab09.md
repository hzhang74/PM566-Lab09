---
title: "Lab09"
author: "Haoran Zhang"
date: "2021/10/29"
output:
  github_document:
    html_preview: false
  html_document: default
always_allow_html: true
---
## Question 1: Think
Parallel computing may be used in weather forecast, partial differential equations and astrophysics.

## Question 2: Before you
The following functions can be written to be more efficient without using parallel:
1. This function generates a n x k dataset with all its entries distributed poission with mean lambda.

```r
fun1 <- function(n = 100, k = 4, lambda = 4) {
  x <- NULL
  for (i in 1:n)
    x <- rbind(x, rpois(k, lambda))
  return(x)
}
fun1(5,10)
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
## [1,]    4    3    5    7    3    3    2    2    5     4
## [2,]    4    5    4   10    2    4    5    3    2     2
## [3,]    3    4    6    2    7    4    1    3    6     3
## [4,]    2    5    4    3    7    5    2    3    4     7
## [5,]    5    3    6    6    2    1    3    3    6     5
```


```r
fun1alt <- function(n = 100, k = 4, lambda = 4) {
  matrix(rpois(n*k, lambda),nrow = n, ncol = k)
}


# Benchmarking
microbenchmark::microbenchmark(
  fun1(n=1000),
  fun1alt(n=1000), unit="relative"
)
```

```
## Unit: relative
##               expr      min       lq    mean   median       uq      max neval
##     fun1(n = 1000) 33.41145 33.34858 33.2662 33.16009 36.31396 5.307938   100
##  fun1alt(n = 1000)  1.00000  1.00000  1.0000  1.00000  1.00000 1.000000   100
```

## Question 3: Find the column max (hint: Checkout the function max.col()).

```r
# Data Generating Process (10 x 10,000 matrix)
set.seed(1234)
x <- matrix(rnorm(1e4), nrow=10)

# Find each column's max value
fun2 <- function(x) {
  apply(x, 2, max)
}

fun2alt <- function(x) {
  # Position of the max value per row of x.
  idx<-max.col(t(x))
  # Do something to get the actual max value
  # x[cbindc(1,15)]-x[1,15]
  # Want to access x[1,16],x[4,1]
  # x[rbind(c(1,16),c(4,1))]
  # Want to access x[4,16],x[4,1]
  # x[rbind(4,c(16,1))]
  x[cbind(idx, 1:ncol(x))]
}

all(fun2(x)==fun2alt(x))
```

```
## [1] TRUE
```

```r
# Benchmarking
microbenchmark::microbenchmark(
  fun2(x),
  fun2alt(x),unit = "relative"
)
```

```
## Unit: relative
##        expr      min       lq     mean   median       uq       max neval
##     fun2(x) 9.645161 9.498227 7.667172 9.217502 8.655172 0.9431093   100
##  fun2alt(x) 1.000000 1.000000 1.000000 1.000000 1.000000 1.0000000   100
```

Example of the max.col function... what just happened?


```r
set.seed(42343)
M <- matrix(runif(12), ncol = 4)
M # How does it look?
```

```
##           [,1]      [,2]       [,3]      [,4]
## [1,] 0.5193214 0.7539021 0.01253299 0.7891065
## [2,] 0.4644698 0.4484567 0.06041682 0.1218734
## [3,] 0.7760090 0.1781403 0.76544068 0.6561172
```

```r
fun2(M)
```

```
## [1] 0.7760090 0.7539021 0.7654407 0.7891065
```

```r
t(M) # Transpose M...
```

```
##            [,1]       [,2]      [,3]
## [1,] 0.51932135 0.46446984 0.7760090
## [2,] 0.75390212 0.44845674 0.1781403
## [3,] 0.01253299 0.06041682 0.7654407
## [4,] 0.78910647 0.12187338 0.6561172
```

```r
idx <- max.col(t(M)) # Then the max.col
# c(3, 1, 3, 1)
idx
```

```
## [1] 3 1 3 1
```

```r
# How do the coordinates look like?
cbind(idx, 1:ncol(M))
```

```
##      idx  
## [1,]   3 1
## [2,]   1 2
## [3,]   3 3
## [4,]   1 4
```

```r
# The final result
M[cbind(idx, 1:ncol(M))]
```

```
## [1] 0.7760090 0.7539021 0.7654407 0.7891065
```

## Question 4: Bootstrap


```r
library(parallel)
my_boot <- function(dat, stat, R, ncpus = 1L) {
  
  # Getting the random indices
  n <- nrow(dat)
  idx <- matrix(sample.int(n, n*R, TRUE), nrow=n, ncol=R)
 
  # Making the cluster using `ncpus`
  # STEP 1: GOES HERE
  cl <- makePSOCKcluster(ncpus)
  
  # STEP 2: GOES HERE
  clusterSetRNGStream(cl, 123) # Equivalent to `set.seed(123)`
  clusterExport(cl, c("stat", "dat", "idx"), envir = environment())
  
  # STEP 3: THIS FUNCTION NEEDS TO BE REPLACES WITH parLapply
  ans <- parLapply(cl = cl, seq_len(R), function(i) {
    stat(dat[idx[,i], , drop=FALSE])
  })
  
  # Coercing the list into a matrix
  ans <- do.call(rbind, ans)
  
  # STEP 4: GOES HERE
  stopCluster(cl)
  
  ans
  
}
# Bootstrap of an OLS
my_stat <- function(d) coef(lm(y ~ x, data=d))
# DATA SIM
set.seed(1)
n <- 500; R <- 5e3
x <- cbind(rnorm(n)); y <- x*5 + rnorm(n)
# Checking if we get something similar as lm
ans0 <- confint(lm(y~x))
ans1 <- my_boot(dat = data.frame(x, y), my_stat, R = R, ncpus = 2L)
# You should get something like this
t(apply(ans1, 2, quantile, c(.025,.975)))
```

```
##                   2.5%      97.5%
## (Intercept) -0.1395732 0.05291612
## x            4.8686527 5.04503468
```

Is it faster?


```r
system.time(my_boot(dat = data.frame(x, y), my_stat, R = 4000, ncpus = 1L))
```

```
##    user  system elapsed 
##    0.09    0.22    3.64
```

```r
system.time(my_boot(dat = data.frame(x, y), my_stat, R = 4000, ncpus = 2L))
```

```
##    user  system elapsed 
##    0.13    0.39    2.41
```
