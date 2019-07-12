
context("smf")

library(splines)

 # use existing data for test
 data(datCancer) 
 
 # unpenalized model using ns function
 f.ns <- ~ns(fu,knots=c(0.25, 0.5, 1, 2, 4),Boundary.knots=c(0,5))

 mod.ns <- survPen(f.ns,data=datCancer,t1=fu,event=dead)
 
 test_that("unpenalized model works", {
  expect_true(inherits(mod.ns,"survPen"))
})

 # unpenalized model smf function
 f.smf <- ~ smf(fu,knots=c(0,0.25, 0.5, 1, 2, 4,5)) # careful here: the boundary knots are included

 mod.smf <- survPen(f.smf,data=datCancer,t1=fu,event=dead,lambda=0)

 test_that("penalized model works", {
  expect_true(inherits(mod.smf,"survPen"))
})
 
 # predictions

 new.time <- seq(0,5,length=50)
 pred.ns <- predict(mod.ns,data.frame(fu=new.time))$haz
 pred.smf <- predict(mod.smf,data.frame(fu=new.time))$haz

 
test_that("basic spline prediction works", {
  expect_true(max(pred.ns - pred.smf) < 1e-10)
})


# penalized spline with LCV smoothing parameter estimation
mod.pen <- survPen(f.smf,data=datCancer,t1=fu,event=dead,method="LCV")

 test_that("LCV smoohting parameter estimation works", {
  expect_true(inherits(mod.pen,"survPen"))
})

mod.pen.excess <- survPen(f.smf,data=datCancer,t1=fu,event=dead,expected=rate,method="LCV")

test_that("LCV smoohting parameter estimation works with expected mortality rates", {
  expect_true(inherits(mod.pen.excess,"survPen"))
})

# summary of the model

test_that("summary of the model works", {
  expect_equal(summary(mod.pen)$penalized.likelihood , mod.pen$ll.pen)
})

test_that("summary of the model with expected rate works", {
  expect_equal(summary(mod.pen.excess)$penalized.likelihood , mod.pen.excess$ll.pen)
})














