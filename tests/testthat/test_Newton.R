 
 context("Newton")

# Newton-Raphson algorithm for regression parameters

data <- data.frame(time=seq(0,5,length=100),event=1,t0=0)

form <- ~ smf(time,knots=c(0,1,3,5))

t1 <- eval(substitute(time), data)
t0 <- eval(substitute(t0), data)
event <- eval(substitute(event), data)
	
model.c <- model.cons(form,lambda=0,data.spec=data,t1=t1,t1.name="time",
t0=rep(0,100),t0.name="t0",event=event,event.name="event",
expected=rep(0,100),expected.name=NULL,type="overall",n.legendre=20,cl="survPen(form,data,t1=time,event=event)",beta.ini=NULL)

  
mod <- survPen.fit(model.c,data,form)

Newton1 <- NR.beta(model.c,beta.ini=rep(0,4),detail.beta=FALSE)
 
test_that("NR.beta ok", {
  expect_true(max(abs(mod$coef - Newton1$beta)) < 1e-10)
})


# Newton-Raphson algorithm for smoothing and regression parameters

mod2 <- survPen(form,data,t1=time,event=event)

# we need to reparameterize the model before fitting
Newton2 <- NR.rho(repam(model.c)$build,rho.ini=-1,data,form,nb.smooth=1,detail.rho=FALSE)

# we then return to the initial parameterization
coef.Newton2 <- as.vector(Newton2$U%*%Newton2$coef )
 
test_that("NR.rho coef ok", {
  expect_true(max(abs(mod2$coef - coef.Newton2)) < 1e-10)
})

test_that("NR.rho lambda ok", {
  expect_true(abs(mod2$lambda - Newton2$lambda) < 1e-10)
})
   
  


