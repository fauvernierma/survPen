

 context("tensor")

# basic dimension test
X1 <- matrix(rnorm(10*3),nrow=10,ncol=3)
X2 <- matrix(rnorm(10*2),nrow=10,ncol=2)
X3 <- matrix(rnorm(10*4),nrow=10,ncol=4)

X <- tensor.prod.X(list(X1,X2,X3))
	
test_that("tensor dimensions ok", {
  expect_equal(dim(X),c(10,24))
})

# tensor penalties
S1 <- matrix(rnorm(3*3),nrow=3,ncol=3)
S2 <- matrix(rnorm(2*2),nrow=2,ncol=2)
 
S1 <- 0.5*(S1 + t(S1) ) ; S2 <- 0.5*(S2 + t(S2) )

S <- tensor.prod.S(list(S1,S2))

test_that("tensor penalty 1 ok", {
  expect_equal(dim(S[[1]]),c(6,6))
})

test_that("tensor penalty 2 ok", {
  expect_equal(dim(S[[2]]),c(6,6))
})


# use existing data for test
 data(datCancer) 
 
# tensor product
mod.tensor <- survPen(~tensor(fu,age),data=datCancer,t1=fu,event=dead)

test_that("tensor works", {
  expect_true(inherits(mod.tensor,"survPen"))
})


test_that("number of smoothing parameters with tensor is ok", {
  expect_equal(length(mod.tensor$lambda),2)
})


# tensor product interaction

mod.tint <- survPen(~tint(fu)+tint(age)+tint(fu,age),data=datCancer,t1=fu,event=dead)

test_that("tint works", {
  expect_true(inherits(mod.tint,"survPen"))
})

test_that("number of smoothing parameters with tint is ok", {
  expect_equal(length(mod.tint$lambda),4)
})
















