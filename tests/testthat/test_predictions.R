context("predictions")


 # use existing data for test
 data(datCancer) 
 
 
  mod1 <- survPen(~smf(fu),t1=fu,event=dead,data=datCancer)
  mod1.LCV <- survPen(~smf(fu),t1=fu,event=dead,data=datCancer,method="LCV")
  
  
  test_that("predictions smf ok", {
  expect_equal(round(predict(mod1,data.frame(fu=5))$surv,4),0.6415)
  expect_equal(round(predict(mod1.LCV,data.frame(fu=5))$surv,4),0.6415)
})

  mod2 <- survPen(~tensor(fu,age),t1=fu,event=dead,data=datCancer)
  mod2.LCV <- survPen(~tensor(fu,age),t1=fu,event=dead,data=datCancer,method="LCV")
  
  test_that("predictions tensor ok", {
  expect_equal(round(predict(mod2,data.frame(fu=5,age=60))$surv,4),0.6177)
  expect_equal(round(predict(mod2.LCV,data.frame(fu=5,age=60))$surv,4),0.613)
})

  
  mod3 <- survPen(~smf(fu) + smf(fu,by=age),t1=fu,event=dead,data=datCancer)
  mod3.LCV <- survPen(~smf(fu) + smf(fu,by=age),t1=fu,event=dead,data=datCancer,method="LCV")
  
  
  test_that("predictions by ok", {
  expect_equal(round(predict(mod3,data.frame(fu=5,age=60))$surv,4),0.6102)
  expect_equal(round(predict(mod3.LCV,data.frame(fu=5,age=60))$surv,4),0.6117)
})

  
  mod4 <- survPen(~smf(fu) + age + tint(fu,by=age,df=10),t1=fu,event=dead,data=datCancer)
  mod4.LCV <- survPen(~smf(fu) + age + tint(fu,by=age,df=10),t1=fu,event=dead,data=datCancer,method="LCV")
  
   test_that("predictions by tint ok", {
  expect_equal(round(predict(mod4,data.frame(fu=5,age=60))$surv,4),0.6102)
  expect_equal(round(predict(mod4.LCV,data.frame(fu=5,age=60))$surv,4),0.6117)
})

  
 
 
 
