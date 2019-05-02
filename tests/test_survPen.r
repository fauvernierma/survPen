#--------------------------------------------------------------------------------------------------------------------------
# test code for survPen package
#--------------------------------------------------------------------------------------------------------------------------


 library(survPen)
 data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer
 
 #-------------------------------------------------------- example 0
 # Comparison between restricted cubic splines and penalized restricted cubic splines

 library(splines)

 # unpenalized
 f <- ~ns(fu,knots=c(0.25, 0.5, 1, 2, 4),Boundary.knots=c(0,5))

 mod <- survPen(f,data=datCancer,t1=fu,event=dead)

 # penalized
 f.pen <- ~ smf(fu,knots=c(0,0.25, 0.5, 1, 2, 4,5)) # careful here: the boundary knots are included

 mod.pen <- survPen(f.pen,data=datCancer,t1=fu,event=dead)

 # predictions

 new.time <- seq(0,5,length=50)
 pred <- predict(mod,data.frame(fu=new.time))
 pred.pen <- predict(mod.pen,data.frame(fu=new.time))

 par(mfrow=c(1,1))
 plot(new.time,pred$haz,type="l",ylim=c(0,0.2),main="hazard vs time",
 xlab="time since diagnosis (years)",ylab="hazard",col="red")
 lines(new.time,pred.pen$haz,col="blue3")
 legend("topright",legend=c("unpenalized","penalized"),
 col=c("red","blue3"),lty=rep(1,2))

 

 #-------------------------------------------------------- example 1
 # hazard models with unpenalized formulas compared to a penalized tensor product smooth

 library(survPen)
 data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer

 # constant hazard model
 f.cst <- ~1
 mod.cst <- survPen(f.cst,data=datCancer,t1=fu,event=dead)

 # piecewise constant hazard model
 f.pwcst <- ~cut(fu,breaks=seq(0,5,by=0.5),include.lowest=TRUE)
 mod.pwcst <- survPen(f.pwcst,data=datCancer,t1=fu,event=dead,n.legendre=200)
 # we increase the number of points for Gauss-Legendre quadrature to make sure that the cumulative
 # hazard is properly approximated

 # linear effect of time
 f.lin <- ~fu
 mod.lin <- survPen(f.lin,data=datCancer,t1=fu,event=dead)

 # linear effect of time and age with proportional effect of age
 f.lin.age <- ~fu+age
 mod.lin.age <- survPen(f.lin.age,data=datCancer,t1=fu,event=dead)

 # linear effect of time and age with time-dependent effect of age (linear)
 f.lin.inter.age <- ~fu*age
 mod.lin.inter.age <- survPen(f.lin.inter.age,data=datCancer,t1=fu,event=dead)

 # cubic B-spline of time with a knot at 1 year, linear effect of age and time-dependent effect
 # of age with a quadratic B-spline of time with a knot at 1 year
 library(splines)
 f.spline.inter.age <- ~bs(fu,knots=c(1),Boundary.knots=c(0,5))+age+
 age:bs(fu,knots=c(1),Boundary.knots=c(0,5),degree=2)
 # here, bs indicates an unpenalized cubic spline

 mod.spline.inter.age <- survPen(f.spline.inter.age,data=datCancer,t1=fu,event=dead)


 # tensor of time and age
 f.tensor <- ~tensor(fu,age)
 mod.tensor <- survPen(f.tensor,data=datCancer,t1=fu,event=dead)


 # predictions of the models at age 60

 new.time <- seq(0,5,length=50)
 pred.cst <- predict(mod.cst,data.frame(fu=new.time))
 pred.pwcst <- predict(mod.pwcst,data.frame(fu=new.time))
 pred.lin <- predict(mod.lin,data.frame(fu=new.time))
 pred.lin.age <- predict(mod.lin.age,data.frame(fu=new.time,age=60))
 pred.lin.inter.age <- predict(mod.lin.inter.age,data.frame(fu=new.time,age=60))
 pred.spline.inter.age <- predict(mod.spline.inter.age,data.frame(fu=new.time,age=60))
 pred.tensor <- predict(mod.tensor,data.frame(fu=new.time,age=60))
 
 lwd1 <- 2
 
 par(mfrow=c(1,1))
 plot(new.time,pred.cst$haz,type="l",ylim=c(0,0.2),main="hazard vs time",
 xlab="years since diagnosis",ylab="hazard",col="blue3",lwd=lwd1)
 segments(x0=new.time[1:49],x1=new.time[2:50],y0=pred.pwcst$haz[1:49],col="lightblue2",lwd=lwd1)
 lines(new.time,pred.lin$haz,col="green3",lwd=lwd1)
 lines(new.time,pred.lin.age$haz,col="yellow",lwd=lwd1)
 lines(new.time,pred.lin.inter.age$haz,col="orange",lwd=lwd1)
 lines(new.time,pred.spline.inter.age$haz,col="red",lwd=lwd1)
 lines(new.time,pred.tensor$haz,col="black",lwd=lwd1)
 legend("topright",
 legend=c("cst","pwcst","lin","lin.age","lin.inter.age","spline.inter.age","tensor"),
 col=c("blue3","lightblue2","green3","yellow","orange","red","black"),
 lty=rep(1,7),lwd=rep(lwd1,7))
 

 # you can also calculate the hazard yourself with the lpmatrix option.
 # For example, compare the following predictions:
 haz.tensor <- pred.tensor$haz

 X.tensor <- predict(mod.tensor,data.frame(fu=new.time,age=60),type="lpmatrix")
 haz.tensor.lpmatrix <- exp(X.tensor%*%mod.tensor$coefficients)

 summary(haz.tensor.lpmatrix - haz.tensor)

 #---------------- The 95% confidence intervals can be calculated like this:
 
 # standard errors from the Bayesian covariance matrix Vp
 std <- sqrt(rowSums((X.tensor%*%mod.tensor$Vp)*X.tensor))
 
 qt.norm <- stats::qnorm(1-(1-0.95)/2)
 haz.inf <- as.vector(exp(X.tensor%*%mod.tensor$coefficients-qt.norm*std))
 haz.sup <- as.vector(exp(X.tensor%*%mod.tensor$coefficients+qt.norm*std))
 
 # checking that they are similar to the ones given by the predict function
 summary(haz.inf - pred.tensor$haz.inf)
 summary(haz.sup - pred.tensor$haz.sup)
 

 #-------------------------------------------------------- example 2
 
 library(survPen)
 data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer

 # model : unidimensional penalized spline for time since diagnosis with 5 knots
 f1 <- ~smf(fu,df=5)
 # when knots are not specified, quantiles are used. For example, for the term "smf(x,df=df1)",
 # the vector of knots will be: quantile(unique(x),seq(0,1,length=df1)) 

 # you can specify your own knots if you want
 # f1 <- ~smf(fu,knots=c(0,1,3,6,8))
 
 # hazard model
 mod1 <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LAML")
 summary(mod1)
 
 # to see where the knots were placed
 mod1$list.smf
 
 # with LCV instead of LAML
 mod1bis <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LCV")
 summary(mod1bis)
 
 # hazard model taking into account left truncation (not representative of cancer data, 
 # the begin variable was simulated for illustration purposes only)
 mod2 <- survPen(f1,data=datCancer,t0=begin,t1=fu,event=dead,expected=NULL,method="LAML")
 summary(mod2)
 
 # excess hazard model
 mod3 <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=rate,method="LAML")
 summary(mod3)
 
 # compare the predictions of the models
 new.time <- seq(0,5,length=50)
 pred1 <- predict(mod1,data.frame(fu=new.time))
 pred1bis <- predict(mod1bis,data.frame(fu=new.time))
 pred2 <- predict(mod2,data.frame(fu=new.time))
 pred3 <- predict(mod3,data.frame(fu=new.time))
 
 # LAML vs LCV
 par(mfrow=c(1,2))
 plot(new.time,pred1$haz,type="l",ylim=c(0,0.2),main="LCV vs LAML",
 xlab="years since diagnosis",ylab="hazard")
 lines(new.time,pred1bis$haz,col="blue3")
 legend("topright",legend=c("LAML","LCV"),col=c("black","blue3"),lty=c(1,1))
 
 plot(new.time,pred1$surv,type="l",ylim=c(0,1),main="LCV vs LAML",
 xlab="years since diagnosis",ylab="survival")
 lines(new.time,pred1bis$surv,col="blue3")
 
 
 # hazard vs excess hazard
 par(mfrow=c(1,2))
 plot(new.time,pred1$haz,type="l",ylim=c(0,0.2),main="hazard vs excess hazard",
 xlab="years since diagnosis",ylab="hazard")
 lines(new.time,pred3$haz,col="green3")
 legend("topright",legend=c("overall","excess"),col=c("black","green3"),lty=c(1,1))
 
 plot(new.time,pred1$surv,type="l",ylim=c(0,1),main="survival vs net survival",
 xlab="time",ylab="survival")
 lines(new.time,pred3$surv,col="green3")
 legend("topright",legend=c("overall survival","net survival"), col=c("black","green3"), lty=c(1,1)) 

 # hazard vs excess hazard with 95% Bayesian confidence intervals (based on Vp matrix, 
 # see predict.survPen)
 par(mfrow=c(1,1))
 plot(new.time,pred1$haz,type="l",ylim=c(0,0.2),main="hazard vs excess hazard",
 xlab="years since diagnosis",ylab="hazard")
 lines(new.time,pred3$haz,col="green3")
 legend("topright",legend=c("overall","excess"),col=c("black","green3"),lty=c(1,1))
 
 lines(new.time,pred1$haz.inf,lty=2)
 lines(new.time,pred1$haz.sup,lty=2)
 
 lines(new.time,pred3$haz.inf,lty=2,col="green3")
 lines(new.time,pred3$haz.sup,lty=2,col="green3")
 


 #-------------------------------------------------------- example 3

 library(survPen)
 data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer

 # models: tensor product smooth vs tensor product interaction of time since diagnosis and 
 # age at diagnosis. Smoothing parameters are estimated via LAML maximization
 f2 <- ~tensor(fu,age,df=c(5,5))
 
 f3 <- ~tint(fu,df=5)+tint(age,df=5)+tint(fu,age,df=c(5,5))
 
 # hazard model
 mod4 <- survPen(f2,data=datCancer,t1=fu,event=dead)
 summary(mod4)
 
 mod5 <- survPen(f3,data=datCancer,t1=fu,event=dead)
 summary(mod5)
 
 # predictions
 new.age <- seq(50,90,length=50)
 new.time <- seq(0,7,length=50)
 
 Z4 <- outer(new.time,new.age,function(t,a) predict(mod4,data.frame(fu=t,age=a))$haz)
 Z5 <- outer(new.time,new.age,function(t,a) predict(mod5,data.frame(fu=t,age=a))$haz)
 
 # color settings
 col.pal <- colorRampPalette(c("white", "red"))
 colors <- col.pal(100)
 
 facet <- function(z){
 
 	facet.center <- (z[-1, -1] + z[-1, -ncol(z)] + z[-nrow(z), -1] + z[-nrow(z), -ncol(z)])/4
 	cut(facet.center, 100)
 	
 }
 
 # plot the hazard surfaces for both models
 par(mfrow=c(1,2))
 persp(new.time,new.age,Z4,col=colors[facet(Z4)],main="tensor",theta=30,
 xlab="years since diagnosis",ylab="age at diagnosis",zlab="excess hazard",ticktype="detailed")
 persp(new.time,new.age,Z5,col=colors[facet(Z5)],main="tint",theta=30,
 xlab="years since diagnosis",ylab="age at diagnosis",zlab="excess hazard",ticktype="detailed")
 
 

#################################################################################################################

