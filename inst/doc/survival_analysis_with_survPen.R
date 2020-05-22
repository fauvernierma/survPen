## ----setup, include = FALSE---------------------------------------------------
library(survPen)
library(splines) # for ns

knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 8, 
  fig.height = 4.5
)

## ---- echo=TRUE, results='asis'-----------------------------------------------
data(datCancer)
knitr::kable(head(datCancer,10))

## ---- fig.show='hold'---------------------------------------------------------
f.cst <- ~1
mod.cst <- survPen(f.cst,data=datCancer,t1=fu,event=dead)

## ---- fig.show='hold'---------------------------------------------------------
f.pwcst <- ~cut(fu,breaks=seq(0,5,by=0.5),include.lowest=TRUE)
mod.pwcst <- survPen(f.pwcst,data=datCancer,t1=fu,event=dead,n.legendre=200)

## ---- fig.show='hold'---------------------------------------------------------
f.lin <- ~fu
mod.lin <- survPen(f.lin,data=datCancer,t1=fu,event=dead)

## ---- fig.show='hold'---------------------------------------------------------
library(splines)

f.rcs <- ~ns(fu,knots=c(0.25, 0.5, 1, 2, 4),Boundary.knots=c(0,5))

mod.rcs <- survPen(f.rcs,data=datCancer,t1=fu,event=dead)

## ---- fig.show='hold'---------------------------------------------------------
f.pen <- ~ smf(fu,knots=c(0,0.25, 0.5, 1, 2, 4,5)) # careful here: the boundary knots are included

mod.pen <- survPen(f.pen,data=datCancer,t1=fu,event=dead)

## ---- fig.show='hold'---------------------------------------------------------
mod.unpen <- survPen(f.pen,data=datCancer,t1=fu,event=dead,lambda=0)

## ---- fig.show='hold'---------------------------------------------------------
new.time <- seq(0,5,length=100)
pred.cst <- predict(mod.cst,data.frame(fu=new.time))
pred.pwcst <- predict(mod.pwcst,data.frame(fu=new.time))
pred.lin <- predict(mod.lin,data.frame(fu=new.time))
pred.rcs <- predict(mod.rcs,data.frame(fu=new.time))
pred.pen <- predict(mod.pen,data.frame(fu=new.time))


lwd1 <- 2

par(mfrow=c(1,1))
plot(new.time,pred.cst$haz,type="l",ylim=c(0,0.2),main="hazard vs time",
xlab="time since diagnosis (years)",ylab="hazard",col="black",lwd=lwd1)
segments(x0=new.time[1:99],x1=new.time[2:100],y0=pred.pwcst$haz[1:99],col="blue3",lwd=lwd1)
lines(new.time,pred.lin$haz,col="green3",lwd=lwd1)
lines(new.time,pred.rcs$haz,col="orange",lwd=lwd1)
lines(new.time,pred.pen$haz,col="red",lwd=lwd1)

legend("topright",
legend=c("constant","piecewise constant","log-linear","cubic spline","penalized cubic spline"),
col=c("black","blue3","green3","orange","red"),
lty=rep(1,5),lwd=rep(lwd1,5))


## ---- fig.show='hold'---------------------------------------------------------

par(mfrow=c(1,2))
plot(new.time,pred.pen$haz,type="l",ylim=c(0,0.2),main="Hazard from mod.pen with CIs",
xlab="time since diagnosis (years)",ylab="hazard",col="red",lwd=lwd1)
lines(new.time,pred.pen$haz.inf,lty=2)
lines(new.time,pred.pen$haz.sup,lty=2)

plot(new.time,pred.pen$surv,type="l",ylim=c(0,1),main="Survival from mod.pen with CIs",
xlab="time since diagnosis (years)",ylab="survival",col="red",lwd=lwd1)
lines(new.time,pred.pen$surv.inf,lty=2)
lines(new.time,pred.pen$surv.sup,lty=2)

## ---- fig.show='hold'---------------------------------------------------------
f.pen.age <- ~tensor(fu,age,df=c(5,5)) # see below for explanations about tensor models
mod.pen.age <- survPen(f.pen.age,data=datCancer,t1=fu,event=dead)
pred.pen.HR <- predict(mod.pen.age,data.frame(fu=new.time,age=70),newdata.ref=data.frame(fu=new.time,age=30),type="HR")

par(mfrow=c(1,1))
plot(new.time,pred.pen.HR$HR,type="l",ylim=c(0,15),main="Hazard ratio with CIs",
xlab="time since diagnosis (years)",ylab="hazard ratio",col="red",lwd=lwd1)
lines(new.time,pred.pen.HR$HR.inf,lty=2)
lines(new.time,pred.pen.HR$HR.sup,lty=2)


## ---- fig.show='hold'---------------------------------------------------------
# you can also calculate the hazard yourself with the lpmatrix option.
# For example, compare the following predictions:
haz.pen <- pred.pen$haz

X.pen <- predict(mod.pen,data.frame(fu=new.time),type="lpmatrix")
haz.pen.lpmatrix <- as.numeric(exp(X.pen%*%mod.pen$coefficients))

summary(haz.pen.lpmatrix - haz.pen)

## ---- fig.show='hold'---------------------------------------------------------
# standard errors from the Bayesian covariance matrix Vp
std <- sqrt(rowSums((X.pen%*%mod.pen$Vp)*X.pen))

qt.norm <- stats::qnorm(1-(1-0.95)/2)
haz.inf <- as.vector(exp(X.pen%*%mod.pen$coefficients-qt.norm*std))
haz.sup <- as.vector(exp(X.pen%*%mod.pen$coefficients+qt.norm*std))

# checking that they are similar to the ones given by the predict function
summary(haz.inf - pred.pen$haz.inf)
summary(haz.sup - pred.pen$haz.sup)


## ---- fig.show='hold'---------------------------------------------------------
summary(mod.pen)

## ---- fig.show='hold'---------------------------------------------------------
mod.pen$ll.unpen
mod.pen$ll.pen
mod.pen$p
sum(mod.pen$edf)
mod.pen$LAML
mod.pen$lambda
summary(mod.pen)$edf.per.smooth

## ---- fig.show='hold'---------------------------------------------------------
mod.pen$aic

## ---- fig.show='hold'---------------------------------------------------------
mod.pen$edf

## ---- fig.show='hold'---------------------------------------------------------
mod.pen$aic2

## ---- fig.show='hold'---------------------------------------------------------
mod.pen$edf2

## ---- fig.show='hold'---------------------------------------------------------
f1 <- ~smf(fu)

mod.LCV <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LCV")
mod.LCV$lambda

mod.LAML <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LAML")
mod.LAML$lambda

new.time <- seq(0,5,length=100)
pred.LCV <- predict(mod.LCV,data.frame(fu=new.time))
pred.LAML <- predict(mod.LAML,data.frame(fu=new.time))


par(mfrow=c(1,1))
plot(new.time,pred.LCV$haz,type="l",ylim=c(0,0.2),main="LCV vs LAML",
xlab="time since diagnosis (years)",ylab="hazard",col="black",lwd=lwd1)
lines(new.time,pred.LAML$haz,col="red",lwd=lwd1,lty=2)
legend("topright",legend=c("LCV","LAML"),col=c("black","red"),lty=c(1,2),lwd=rep(lwd1,2))


## ---- fig.show='hold'---------------------------------------------------------
rho.vec <- seq(-1,15,length=50)
LCV <- rep(0,50)
LAML <- rep(0,50)

for (i in 1:50){
	mod <- survPen(f1,data=datCancer,t1=fu,event=dead,lambda=exp(rho.vec[i]))
	LCV[i] <- mod$LCV
	LAML[i] <- mod$LAML
}	
	
	
par(mfrow=c(1,2),mar=c(3,3,1.5,0.5),mgp=c(1.5,0.5,0))
plot(rho.vec,LCV,type="l",main="LCV vs log(lambda)",ylab="LCV",xlab="log(lambda)",lwd=lwd1)	
	
plot(rho.vec,LAML,type="l",main="LAML vs log(lambda)",ylab="-LAML",xlab="log(lambda)",lwd=lwd1)	

## ---- fig.show='hold'---------------------------------------------------------
f1 <- ~smf(fu,df=5)

## ---- fig.show='hold'---------------------------------------------------------
df1 <- 5
quantile(unique(datCancer$fu),seq(0,1,length=df1)) 


## ---- fig.show='hold'---------------------------------------------------------
mod1 <- survPen(f1,data=datCancer,t1=fu,event=dead)

mod1$list.smf

## ---- fig.show='hold'---------------------------------------------------------
# f1 <- ~smf(fu,knots=c(0,1,3,6,8))

## ---- fig.show='hold'---------------------------------------------------------
mod.total <- survPen(f1,data=datCancer,t1=fu,event=dead,method="LAML")

mod.excess <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=rate,method="LAML")

# compare the predictions of the models
new.time <- seq(0,5,length=100)
pred.total <- predict(mod.total,data.frame(fu=new.time))
pred.excess <- predict(mod.excess,data.frame(fu=new.time))

# hazard vs excess hazard
par(mfrow=c(1,2))
plot(new.time,pred.total$haz,type="l",ylim=c(0,0.2),main="hazard vs excess hazard",
xlab="time since diagnosis (years)",ylab="hazard",lwd=lwd1)
lines(new.time,pred.excess$haz,col="red",lwd=lwd1,lty=2)
legend("topright",legend=c("total","excess"),col=c("black","red"),lty=c(1,2), lwd=rep(lwd1,2))

plot(new.time,pred.total$surv,type="l",ylim=c(0,1),main="survival vs net survival",
xlab="time",ylab="survival",lwd=lwd1)
lines(new.time,pred.excess$surv,col="red",lwd=lwd1,lty=2)
legend("bottomleft",legend=c("overall survival","net survival"), col=c("black","red"), lty=c(1,2), lwd=rep(lwd1,2)) 



## ---- fig.show='hold'---------------------------------------------------------
f.tensor <- ~tensor(fu,age,df=c(5,5))

f.tint <- ~tint(fu,df=5)+tint(age,df=5)+tint(fu,age,df=c(5,5))

# hazard model
mod.tensor <- survPen(f.tensor,data=datCancer,t1=fu,event=dead)
summary(mod.tensor)

mod.tint <- survPen(f.tint,data=datCancer,t1=fu,event=dead)
summary(mod.tint)

# predictions
new.age <- seq(50,90,length=50)
new.time <- seq(0,7,length=50)

Z.tensor <- outer(new.time,new.age,function(t,a) predict(mod.tensor,data.frame(fu=t,age=a))$haz)
Z.tint <- outer(new.time,new.age,function(t,a) predict(mod.tint,data.frame(fu=t,age=a))$haz)

# color settings
col.pal <- colorRampPalette(c("white", "red"))
colors <- col.pal(100)

facet <- function(z){

	facet.center <- (z[-1, -1] + z[-1, -ncol(z)] + z[-nrow(z), -1] + z[-nrow(z), -ncol(z)])/4
	cut(facet.center, 100)
	
}


theta1 = 30
zmax=1.1

# plot the hazard surfaces for both models
par(mfrow=c(1,2),mar=c(3,3,1.5,0.5),mgp=c(1.5,0.5,0))
persp(new.time,new.age,Z.tensor,col=colors[facet(Z.tensor)],main="tensor",theta=theta1,
xlab="\n time since diagnosis",ylab="\n age",zlab="\n excess hazard",
ticktype="detailed",zlim=c(0,zmax))
persp(new.time,new.age,Z.tint,col=colors[facet(Z.tint)],main="tint",theta=theta1,
xlab="\n time since diagnosis",ylab="\n age",zlab="\n excess hazard",
ticktype="detailed",zlim=c(0,zmax))


## ---- fig.show='hold'---------------------------------------------------------
set.seed(18)
subdata <- datCancer[sample(1:2000,50),]

## ---- fig.show='hold'---------------------------------------------------------
mod.tensor.sub <- survPen(f.tensor,data=subdata,t1=fu,event=dead)

mod.tint.sub <- survPen(f.tint,data=subdata,t1=fu,event=dead)

## ---- fig.show='hold'---------------------------------------------------------
# tensor
mod.tensor.sub$lambda
summary(mod.tensor.sub)$edf.per.smooth

# tint
mod.tint.sub$lambda
summary(mod.tint.sub)$edf.per.smooth

## ---- fig.show='hold'---------------------------------------------------------
new.age <- seq(quantile(subdata$age,0.10),quantile(subdata$age,0.90),length=50)
new.time <- seq(0,max(subdata$fu),length=50)

Z.tensor.sub <- outer(new.time,new.age,function(t,a) predict(mod.tensor.sub,data.frame(fu=t,age=a))$haz)
Z.tint.sub <- outer(new.time,new.age,function(t,a) predict(mod.tint.sub,data.frame(fu=t,age=a))$haz)

theta1 = 30
zmax=0.7

# plot the hazard surfaces for both models
par(mfrow=c(1,2),mar=c(3,3,1.5,0.5),mgp=c(1.5,0.5,0))
persp(new.time,new.age,Z.tensor.sub,col=colors[facet(Z.tensor.sub)],main="tensor",theta=theta1,
xlab="\n time since diagnosis",ylab="\n age",zlab="\n excess hazard",
ticktype="detailed",zlim=c(0,zmax))
persp(new.time,new.age,Z.tint.sub,col=colors[facet(Z.tint.sub)],main="tint",theta=theta1,
xlab="\n time since diagnosis",ylab="\n age",zlab="\n excess hazard",
ticktype="detailed",zlim=c(0,zmax))

## ---- fig.show='hold'---------------------------------------------------------
data2D <- expand.grid(fu=new.time,age=c(50,60,70,80))

data2D$haz.tensor <- predict(mod.tensor.sub,data2D)$haz
data2D$haz.tint <- predict(mod.tint.sub,data2D)$haz


par(mfrow=c(2,2),mar=c(3,3,1.5,0.5),mgp=c(1.5,0.5,0))

plot(new.time,data2D[data2D$age==50,]$haz.tensor,type="l",ylim=c(0,0.7),
main="age 50",xlab="time since diagnosis",ylab="excess hazard",lwd=lwd1)
lines(new.time,data2D[data2D$age==50,]$haz.tint,col="red",lty=2,lwd=lwd1)
legend("topright",c("tensor","tint"),lty=c(1,2),col=c("black","red"),lwd=rep(lwd1,2))

for (i in c(60,70,80)){
plot(new.time,data2D[data2D$age==i,]$haz.tensor,type="l",ylim=c(0,0.7),
main=paste("age", i),xlab="time since diagnosis",ylab="excess hazard",lwd=lwd1)
lines(new.time,data2D[data2D$age==i,]$haz.tint,col="red",lty=2,lwd=lwd1)
}

## ---- fig.show='hold'---------------------------------------------------------
mod.tensor.sub$aic2

mod.tint.sub$aic2

## ---- fig.show='hold'---------------------------------------------------------
f4 <- ~tensor(fu,age,yod,df=c(5,5,5))

# excess hazard model
mod6 <- survPen(f4,data=datCancer,t1=fu,event=dead,expected=rate)
summary(mod6)


# predictions of surfaces for years 1990, 1997, 2003 and 2010
new.age <- seq(50,90,length=50)
new.time <- seq(0,5,length=50)

Z_1990 <- outer(new.time,new.age,function(t,a) predict(mod6,data.frame(fu=t,yod=1990,age=a))$haz)
Z_1997 <- outer(new.time,new.age,function(t,a) predict(mod6,data.frame(fu=t,yod=1997,age=a))$haz)
Z_2003 <- outer(new.time,new.age,function(t,a) predict(mod6,data.frame(fu=t,yod=2003,age=a))$haz)
Z_2010 <- outer(new.time,new.age,function(t,a) predict(mod6,data.frame(fu=t,yod=2010,age=a))$haz)


par(mfrow=c(1,2),mar=c(3,3,1.5,0.5),mgp=c(1.5,0.5,0))
persp(new.time,new.age,Z_1990,col=colors[facet(Z_1990)],main="1990",theta=20,
xlab="\n time since diagnosis",ylab="\n age",zlab="\n excess hazard",
ticktype="detailed",zlim=c(0,1))
persp(new.time,new.age,Z_1997,col=colors[facet(Z_1997)],main="1997",theta=20,
xlab="\n time since diagnosis",ylab="\n age",zlab="\n excess hazard",
ticktype="detailed",zlim=c(0,1))

par(mfrow=c(1,2),mar=c(3,3,1.5,0.5),mgp=c(1.5,0.5,0))
persp(new.time,new.age,Z_2003,col=colors[facet(Z_2003)],main="2003",theta=20,
xlab="\n time since diagnosis",ylab="\n age",zlab="\n excess hazard",
ticktype="detailed",zlim=c(0,1))
persp(new.time,new.age,Z_2010,col=colors[facet(Z_2010)],main="2010",theta=20,
xlab="\n time since diagnosis",ylab="\n age",zlab="\n excess hazard",
ticktype="detailed",zlim=c(0,1))

## ---- fig.show='hold'---------------------------------------------------------
n <- 10000
don <- data.frame(num=1:n)

shape_men <- 0.90 # shape for men (first weibull parameter)
shape_women <- 0.90 # shape for women

scale_men <- 0.6 # second weibull parameter
scale_women <- 0.7

prop_men <- 0.5 # proportion of men

set.seed(50)
don$sex <- factor(sample(c("men","women"),n,replace=TRUE,prob=c(prop_men,1-prop_men)))
don$sex.order <- factor(don$sex,levels=c("women","men"),ordered=TRUE)

don$shape <- ifelse(don$sex=="men",shape_men,shape_women)
don$scale <- ifelse(don$sex=="men",scale_men,scale_women)

don$fu <- rweibull(n,shape=don$shape,scale=don$scale)
don$dead <- 1 # no censoring

## ---- fig.show='hold'---------------------------------------------------------
hazard <- function(x,shape,scale){
exp(dweibull(x,shape=shape,scale=scale,log=TRUE) - pweibull(x,shape=shape,scale=scale,log.p=TRUE,lower.tail=FALSE))
}

nt <- seq(0.01,5,by=0.1)

mar1 <- c(3,3,1.5,0.5)
mgp1 <- c(1.5,0.5,0)

par(mfrow=c(1,2),mar=mar1,mgp=mgp1)
plot(nt,hazard(nt,shape_women,scale_women),type="l",
xlab="time",ylab="hazard",lwd=lwd1,main="Theoretical hazards",
ylim=c(0,max(hazard(nt,shape_women,scale_women),hazard(nt,shape_men,scale_men))))
lines(nt,hazard(nt,shape_men,scale_men),col="red",lwd=lwd1,lty=2)
legend("bottomleft",c("women","men"),lty=c(1,2),lwd=rep(lwd1,2),col=c("black","red"))

plot(nt,hazard(nt,shape_men,scale_men)/hazard(nt,shape_women,scale_women),type="l",
xlab="time",ylab="hazard ratio",lwd=lwd1,
ylim=c(0,2),
main="Theoretical HR men / women")

## ---- fig.show='hold'---------------------------------------------------------
# knots for time
knots.t <- quantile(don$fu,seq(0,1,length=10))

# stratified analysis via the two models
m.men <- survPen(~smf(fu,knots=knots.t),t1=fu,event=dead,data=don[don$sex=="men",])
m.women <- survPen(~smf(fu,knots=knots.t),t1=fu,event=dead,data=don[don$sex=="women",])

# by variable with same.rho = FALSE
m.FALSE <- survPen(~sex + smf(fu,by=sex,same.rho=FALSE,knots=knots.t),t1=fu,event=dead,data=don)

# by variable with same.rho = TRUE
m.TRUE <- survPen(~sex + smf(fu,by=sex,same.rho=TRUE,knots=knots.t),t1=fu,event=dead,data=don)

# difference smooth via ordered factor by variable
m.difference <- survPen(~sex.order + smf(fu,knots=knots.t) +smf(fu,by=sex.order,same.rho=FALSE,knots=knots.t),t1=fu,event=dead,data=don)

## ---- fig.show='hold'---------------------------------------------------------
newt <- seq(0,5,by=0.1)
data.pred <- expand.grid(fu=newt,sex=c("women","men"))
data.pred$men <- ifelse(data.pred$sex=="men",1,0)
data.pred$women <- ifelse(data.pred$sex=="women",1,0)
data.pred$sex.order <- data.pred$sex # no need to reorder here as the model keeps track of the factor's structure

data.pred$haz.men <- predict(m.men,data.pred)$haz
data.pred$haz.women <- predict(m.women,data.pred)$haz
data.pred$haz.FALSE <- predict(m.FALSE,data.pred)$haz
data.pred$haz.TRUE <- predict(m.TRUE,data.pred)$haz
data.pred$haz.difference <- predict(m.difference,data.pred)$haz

# predicting hazard
ylim1 <- c(0,max(data.pred$haz.men,data.pred$haz.women,
data.pred$haz.FALSE,data.pred$haz.TRUE,data.pred$haz.difference))

par(mfrow=c(1,2),mar=mar1,mgp=mgp1)
plot(newt,data.pred[data.pred$sex=="men",]$haz.men,type="l",main="Men",lwd=lwd1,
ylim=ylim1,xlab="time since diagnosis",ylab="hazard")
lines(newt,data.pred[data.pred$sex=="men",]$haz.FALSE,col="red",lwd=lwd1,lty=2)
lines(newt,data.pred[data.pred$sex=="men",]$haz.TRUE,col="green3",lwd=lwd1,lty=4)
lines(newt,data.pred[data.pred$sex=="men",]$haz.difference,col="orange",lwd=lwd1,lty=5)
lines(nt,hazard(nt,shape_men,scale_men),col="blue3",lty=3)
legend("bottomleft",c("stratified","same.rho=FALSE","same.rho=TRUE","difference smooth","true"),lty=c(1,2,4,5,3),
col=c("black","red","green3","orange","blue3"),lwd=c(rep(lwd1,4),1))

plot(newt,data.pred[data.pred$sex=="women",]$haz.women,type="l",main="Women",lwd=lwd1,
ylim=ylim1,xlab="time since diagnosis",ylab="hazard")
lines(newt,data.pred[data.pred$sex=="women",]$haz.FALSE,col="red",lwd=lwd1,lty=2)
lines(newt,data.pred[data.pred$sex=="women",]$haz.TRUE,col="green3",lwd=lwd1,lty=4)
lines(newt,data.pred[data.pred$sex=="women",]$haz.difference,col="orange",lwd=lwd1,lty=5)
lines(nt,hazard(nt,shape_women,scale_women),col="blue3",lty=3)

## ---- fig.show='hold'---------------------------------------------------------
# predicting hazard ratio men / women

HR.stratified <- data.pred[data.pred$sex=="men",]$haz.men / data.pred[data.pred$sex=="women",]$haz.women
HR.FALSE <- data.pred[data.pred$sex=="men",]$haz.FALSE / data.pred[data.pred$sex=="women",]$haz.FALSE
HR.TRUE <- data.pred[data.pred$sex=="men",]$haz.TRUE / data.pred[data.pred$sex=="women",]$haz.TRUE
HR.difference <- data.pred[data.pred$sex=="men",]$haz.difference / data.pred[data.pred$sex=="women",]$haz.difference

par(mfrow=c(1,1))
plot(newt,HR.stratified,type="l",main="Hazard ratio, Men/Women",lwd=lwd1,
ylim=c(0,2),xlab="time since diagnosis",ylab="hazard ratio")
lines(newt,HR.FALSE,col="red",lwd=lwd1,lty=2)
lines(newt,HR.TRUE,col="green3",lwd=lwd1,lty=4)
lines(newt,HR.difference,col="orange",lwd=lwd1,lty=5)
abline(h=hazard(nt,shape_men,scale_men)/hazard(nt,shape_women,scale_women),lty=3,col="blue3")
legend("bottomright",c("stratified","same.rho=FALSE","same.rho=TRUE","difference smooth","true"),lty=c(1,2,4,5,3),
col=c("black","red","green3","orange","blue3"),lwd=c(rep(lwd1,4),1))

## ---- fig.show='hold'---------------------------------------------------------
datCancer$agec <- datCancer$age - 50

## ---- fig.show='hold'---------------------------------------------------------
m <- survPen(~smf(fu) + smf(fu,by=agec),data=datCancer,t1=fu,event=dead)
m$ll.pen

## ---- fig.show='hold'---------------------------------------------------------
m.bis <- survPen(~smf(fu) + agec + tint(fu,by=agec,df=10),data=datCancer,t1=fu,event=dead)
m.bis$ll.pen # same penalized log-likelihood as m

## ---- fig.show='hold'---------------------------------------------------------
m2 <- survPen(~tint(fu,df=10) + tint(agec,df=10) + tint(fu,by=agec,df=10),data=datCancer,t1=fu,event=dead)
m2$ll.pen 

## ---- fig.show='hold'---------------------------------------------------------
  set.seed(1)
  
  # Weibull parameters
  shape <- 0.9
  scale <- 2
  
  # number of simulated datasets
  NFile <- 50
  
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
  
  # vector of estimated log standard deviations
  log.sd.vec <- rep(as.numeric(NA),NFile)
  
  # maximum follow-up time
  max.time <- 5
  

## ---- fig.show='hold'---------------------------------------------------------
  for (file in 1:NFile){
  
	  wj <- rnorm(NCluster,mean=0,sd=sd1) 
	  
	  don$wj <- wj[don$cluster]
	  
	  # simulated times
	  u <- runif(n)
	  don$fu <- exp( 1/shape*(log(-log(1-u)) - don$wj) + log(scale))
	  
	  # censoring
	  don$dead <- ifelse(don$fu <= max.time,1,0)
	  don$fu <- pmin(don$fu,max.time)
	  
	  # fitting
	  mod.frailty <- survPen(~smf(fu)+rd(cluster),data=don,t1=fu,event=dead)
	  
	  # estimated log standard deviation
	  log.sd.vec[file] <- summary(mod.frailty)$random.effects[,"Estimate"]
	  
   }

# Relative Bias in percentage for sd1
100*(mean(exp(log.sd.vec)) - sd1)/sd1

## ---- fig.show='hold'---------------------------------------------------------
summary(mod.frailty)

## ---- fig.show='hold'---------------------------------------------------------
exp(summary(mod.frailty)$random.effects)[1]

## ---- fig.show='hold'---------------------------------------------------------
exp(-0.5*log(mod.frailty$lambda)-0.5*log(mod.frailty$S.scale))[2]

## ---- fig.show='hold'---------------------------------------------------------
# 1-year survival for a patient in cluster 6
predict(mod.frailty,data.frame(fu=1,cluster=6))$surv

# 1-year survival for a patient in cluster 10
predict(mod.frailty,data.frame(fu=1,cluster=10))$surv

## ---- fig.show='hold'---------------------------------------------------------
# 1-year survival for a patient when random effect is set to zero 
predict(mod.frailty,data.frame(fu=1,cluster=10),exclude.random=TRUE)$surv

## ---- fig.show='hold'---------------------------------------------------------
# fitting
f1 <- ~smf(fu)

mod.trunc <- survPen(f1,data=datCancer,t0=begin,t1=fu,event=dead,expected=NULL,method="LAML")

# predictions
new.time <- seq(0,5,length=100)

pred.trunc <- predict(mod.trunc,data.frame(fu=new.time))

par(mfrow=c(1,2))
plot(new.time,pred.trunc$haz,type="l",ylim=c(0,0.2),main="Hazard",
xlab="time since diagnosis (years)",ylab="hazard",lwd=lwd1)

plot(new.time,pred.trunc$surv,type="l",ylim=c(0,1),main="Survival",
xlab="time since diagnosis (years)",ylab="survival",lwd=lwd1)


## ---- fig.show='hold'---------------------------------------------------------
f.pen <- ~ smf(fu) 

vec.lambda <- c(0,1000,10^6)
new.time <- seq(0,5,length=100)


par(mfrow=c(1,3),mar=c(3,3,1.5,0.5),mgp=c(1.5,0.5,0))

for (i in (1:3)){

	mod.pen <- survPen(f.pen,data=datCancer,t1=fu,event=dead,lambda=vec.lambda[i])
	pred.pen <- predict(mod.pen,data.frame(fu=new.time))

	plot(new.time,pred.pen$haz,type="l",ylim=c(0,0.2),main=paste0("hazard vs time, lambda = ",vec.lambda[i]),
	xlab="time since diagnosis (years)",ylab="hazard",col="black",lwd=lwd1)

}

## ---- fig.show='hold'---------------------------------------------------------
mod.pen <- survPen(f.pen,data=datCancer,t1=fu,event=dead,rho.ini=5)

mod.excess.pen <- survPen(f.pen,data=datCancer,t1=fu,event=dead,expected=rate,rho.ini=5,beta.ini=mod.pen$coef)

## ---- fig.show='hold'---------------------------------------------------------
mod.pen <- survPen(f.pen,data=datCancer,t1=fu,event=dead,detail.rho=TRUE)

## ---- fig.show='hold'---------------------------------------------------------
mod.pen <- survPen(f.pen,data=datCancer,t1=fu,event=dead,detail.rho=TRUE,detail.beta=TRUE)

