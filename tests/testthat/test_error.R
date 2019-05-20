
context("error")


 # use existing data for test
 data(datCancer) 
 


 test_that("wrong method gives error", {
  expect_error(survPen(~smf(fu),data=datCancer,t1=fu,event=dead,method="blabla"),
  "method should be LAML or LCV")
})
 

 test_that("missing event gives error", {
  expect_error(survPen(~smf(fu),data=datCancer,t1=fu),
  "Must have at least a formula, data, t1 and event arguments")
})
 

test_that("wrong number of initial smoothing parameters gives error", {
  expect_error(survPen(~smf(fu),data=datCancer,t1=fu,event=dead,rho.ini=c(-1,-1)),
  "number of initial log smoothing parameters incorrect")
})



test_that("wrong number of smf covariates gives error", {
  expect_error(survPen(~smf(fu,age),data=datCancer,t1=fu,event=dead),
  "smf calls must contain only one covariate")
})








