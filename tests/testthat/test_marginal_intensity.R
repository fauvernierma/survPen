 
 context("MarginalIntensity")

data(HeartFailure)

# We are going to fit an unpenalized spline on the log-marginal intensity in order to use the robust variance with 
# a proper theoretical background. Indeed, with penalized splines it is advised to use
# bootstrap for confidence intervals (Coz et al. submitted to Biostatistics).

# Number of knots for an unpenalized natural cubic spline
df.t <- 4
k.t <-  c(quantile(HeartFailure$t1[HeartFailure[,"event"] == 1],seq(0,1,le=df.t)))

# We consider a non-proportional effect of treatment
mod_MI <- survPen(~ smf(t1, knots = k.t), 
                        t0 = t0, 
                        t1 = t1, data = HeartFailure, event = event, cluster=id, lambda=0)

# predictions
nt <- c(0,3)

# marginal intensity
pred.MI <- predict(mod_MI, newdata = data.frame(t1=nt))

test_that("Marginal intensity prediction ok", {
  expect_true(abs(pred.MI$haz[1] -  1.1266532434447120714) < 1e-10)
})

test_that("Marginal intensity standard error ok", {
  expect_true(abs(summary(mod_MI)$coefficients[1,2]
 -  0.032280101353462826996) < 1e-10)
})
   
  


