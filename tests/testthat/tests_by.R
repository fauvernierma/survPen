

 context("by")

 # use existing data for test
 data(datCancer) 
 

 
# continuous by variable
mod.by <- survPen(~smf(fu,by=age),data=datCancer,t1=fu,event=dead)

test_that("continuous by variables work", {
  expect_true(inherits(mod.by,"survPen"))
})

mod.by2 <- survPen(~age + tint(fu,by=age,df=10),data=datCancer,t1=fu,event=dead)

test_that("alternative 1 for continuous by variables work", {
  expect_true(inherits(mod.by2,"survPen"))
})

test_that("alternative 1 leads to the same penalized likelihood", {
  expect_true(abs(mod.by$ll.pen-mod.by2$ll.pen) < 1e-8)
})


mod.by3 <- survPen(~tensor(fu,by=age,df=10),data=datCancer,t1=fu,event=dead)

test_that("alternative 2 for continuous by variables work", {
  expect_true(inherits(mod.by3,"survPen"))
})

test_that("alternative 2 leads to the same penalized likelihood", {
  expect_true(abs(mod.by$ll.pen-mod.by3$ll.pen) < 1e-8)
})


# factor by variable

# creating sex variable for test purpose only
set.seed(1)
datCancer$sex <-  factor(sample(rep(1:2,each=dim(datCancer)[1]/2)))


mod.by.factor <- survPen(~sex + smf(fu,by=sex),data=datCancer,t1=fu,event=dead)

test_that("factor by variables work", {
  expect_true(inherits(mod.by.factor,"survPen"))
})

test_that("number of smoothing parameters for factor by variables work", {
  expect_equal(length(mod.by.factor$lambda),2)
})


mod.by.factor2 <- survPen(~sex + smf(fu,by=sex,same.rho=TRUE),data=datCancer,t1=fu,event=dead)

test_that("factor by variables 2 work", {
  expect_true(inherits(mod.by.factor2,"survPen"))
})

test_that("number of smoothing parameters for factor by variables 2 work", {
  expect_equal(length(mod.by.factor2$lambda),1)
})


# ordered factor for difference smooth
datCancer$sex2 <- factor(datCancer$sex,ordered=TRUE)


mod.by.factor3 <- survPen(~sex2 + smf(fu) + smf(fu,by=sex2),data=datCancer,t1=fu,event=dead)


test_that("ordered factor by variables work", {
  expect_true(inherits(mod.by.factor3,"survPen"))
})

test_that("number of smoothing parameters for ordered factor by variables work", {
  expect_equal(length(mod.by.factor3$lambda),2)
})





















