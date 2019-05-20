

 context("random_effects")

 # use existing data for test
 data(datCancer) 
 

# creating cluster variable for test purpose only
set.seed(1)
NCluster <- 10
datCancer$cluster <-  factor(sample(rep(1:NCluster,each=dim(datCancer)[1]/NCluster)))

mod.rd <- survPen(~rd(cluster),data=datCancer,t1=fu,event=dead)

test_that("random effects models work", {
  expect_true(inherits(mod.rd,"survPen"))
})


test_that("estimated variance is ok", {
  expect_true(abs(exp(summary(mod.rd)$random.effects)[1]
 - exp(-0.5*log(mod.rd$lambda)-0.5*log(mod.rd$S.scale))[1]) < 1e-08)
})


test_that("penalized likelihoods from model and summary matches", {
  expect_equal(summary(mod.rd)$penalized.likelihood , mod.rd$ll)
})


test_that("summary has right dimensions", {
  expect_true(all(dim(summary(mod.rd)$random.effects)==c(1,2)))
})



