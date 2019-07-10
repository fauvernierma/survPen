
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



x <- seq(1,10,length=100)
test_that("wrong df gives error", {
expect_error(crs(x,df=2),
  "Number of knots should be at least 3, 1 interior plus 2 boundaries")
})

test_that("wrong number of knots gives error", {
expect_error(crs(x,knots=c(0,10)),
  "Please specify at least 3 knots, 1 interior plus 2 boundaries")
})

test_that("wrong number of individuals gives error", {
expect_error(crs(1,df=10),
  "Please specify at least 2 values or specify at least 3 knots via knots=...")
})


  mod1 <- try(survPen(~tensor(fu,age),t1=fu,event=dead,lambda=exp(c(-15,15)),data=datCancer))
  
test_that("limit case is ok", {
expect_true(inherits(mod1,"survPen"))
})


 mod2 <- try(survPen(~tensor(fu,age),t1=fu,event=dead,lambda=exp(c(-20,20)),data=datCancer))
  
  mod3 <- try(survPen(~tensor(fu,age),t1=fu,event=dead,lambda=exp(c(55,45)),data=datCancer,max.it.beta=50))
  
  mod4 <- try(survPen(~tensor(fu,age),t1=fu,event=dead,data=datCancer[1:6,]))

test_that("extreme cases give errors", {
expect_true(class(mod2)=="try-error")
expect_true(class(mod3)=="try-error")
expect_true(class(mod4)=="try-error")
})

 
  don <- datCancer
  don$event <- 0
  don[don$fu < 0.03,]$event <- 1
  mod5 <- try(survPen(~tensor(fu,age),t1=fu,event=event,data=don))
	
 test_that("Hessian perturbation at convergence", {
expect_true(mod5$Hess.rho.modif)

})

 
  
  
 
    




