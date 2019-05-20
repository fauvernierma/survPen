

 context("tensor")

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
















