

 context("random_effects")


# creating cluster variable for test purpose only
set.seed(1)
  
  # Weibull parameters
  shape <- 0.9
  scale <- 2
  
  # number of simulated datasets
  NFile <- 1
  
  # number of individuals per dataset
  n <- 2000
  
  # number of clusters
  NCluster <- 20
 
  # data frame
  data.rd <- data.frame(cluster=seq(1:NCluster))
  cluster <- sample(rep(1:NCluster,each=n/NCluster))
  don <- data.frame(num=1:n, cluster=factor(cluster)) # be careful, cluster needs to be a factor !
  don  <- merge(don, data.rd, by="cluster")[, union(names(don), names(data.rd))]
  don <- don[order(don$num),]
  rownames(don) <- NULL
  
  # theoretical standard deviation
  sd1 <- 0.1
  
  
  # maximum follow-up time
  max.time <- 5
  

	  wj <- rnorm(NCluster,mean=0,sd=sd1) 
	  
	  don$wj <- wj[don$cluster]
	  
	  # simulated times
	  u <- runif(n)
	  don$fu <- exp( 1/shape*(log(-log(1-u)) - don$wj) + log(scale))
	  
	  # censoring
	  don$dead <- ifelse(don$fu <= max.time,1,0)
	  don$fu <- pmin(don$fu,max.time)
	  
	  # fitting
	  mod.rd <- survPen(~smf(fu)+rd(cluster),data=don,t1=fu,event=dead,detail.beta=TRUE,detail.rho=TRUE)
	  
	 
test_that("random effects models work", {
  expect_true(inherits(mod.rd,"survPen"))
})


test_that("estimated variance is ok", {
  expect_true(abs(exp(summary(mod.rd)$random.effects)[1]
 - exp(-0.5*log(mod.rd$lambda)-0.5*log(mod.rd$S.scale))[2]) < 1e-08)
})


test_that("prediction ok", {
  expect_false(predict(mod.rd,data.frame(fu=5,cluster=1))$surv == predict(mod.rd,data.frame(fu=5,cluster=1),exclude.random=TRUE)$surv) 
})
	  


test_that("penalized likelihoods from model and summary matches", {
  expect_equal(summary(mod.rd)$penalized.likelihood , mod.rd$ll.pen)
})


test_that("summary has right dimensions", {
  expect_true(all(dim(summary(mod.rd)$random.effects)==c(1,2)))
})



	  

	

