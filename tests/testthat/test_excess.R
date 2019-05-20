

 context("excess")

 # use existing data for test
 data(datCancer) 
 

# excess hazard models
mod.total <- survPen(~smf(fu),data=datCancer,t1=fu,event=dead)


 test_that("hazard models have type=overall", {
  expect_true(mod.total$type=="overall")
})

# excess hazard models
mod.excess <- survPen(~smf(fu),data=datCancer,t1=fu,event=dead,expected=rate)

 test_that("excess hazard models works", {
  expect_true(inherits(mod.excess,"survPen"))
})


 test_that("excess hazard models have type=net", {
  expect_true(mod.excess$type=="net")
})

# predictions : excess hazard should be < total hazard

ntime <- seq(0,5,length=50)

test_that("excess hazard < total hazard", {
  expect_true(all(predict(mod.total,data.frame(fu=ntime))$haz > 
  predict(mod.excess,data.frame(fu=ntime))$haz))
})









