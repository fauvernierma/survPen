#--------------------------------------------------------------------------------------------------------------------------
# Comparison between survPen, rstpm2 and mgcv
#--------------------------------------------------------------------------------------------------------------------------

library(rstpm2)
library(mgcv)
library(survPen)


#_____________________________________________________________________
# Preparing data

# functions to split the data in order to use mgcv

library(RCurl)
	

script1 <- getURL("https://github.com/fauvernierma/survPen/blob/master/compar/fun.split.data.R")
script2 <- getURL("https://github.com/fauvernierma/survPen/blob/master/compar/Lexis.R")

eval(parse(text = script1))	
eval(parse(text = script2))	


# dataframe for survpen and rstpm2
data(datCancer)
dat <- datCancer

# identifying all individuals
dat$Ident <- 1:dim(dat)[1]

# corresponding split dataframe for mgcv (we need to split enough so that the 
# "point milieu" gives a good approximation of the integral of the hazard = cumulative hazard)
split_dat <- fun.split.data(dat, bands=c(seq(0,1,by=0.05),seq(1.1,5,by=0.1)))
	

	
#_____________________________________________________________________
# Fitting models 

# We are going to fit a tensor product spline of time and age 
# using cross-validation GCV/LCV for smoohting parameter estimation

# The linear predictor is the log-hazard for survpen and mgcv and the log-cumulative hazard
# for rstpm2 (the log-hazard scale is not available) 
	
	
# Number of knots for each covariate
df2 <- c(5,5) 

# so we get 5*5 = 25 regression parameters associated with two smoothing parameters



#-------------- survPen
cat("\n","\n","_____________________________________________________________________", "\n")
cat("\n","survPen ","\n") 

time.survPen <- system.time(survPen.tensor <- survPen(~tensor(fu,age,df=df2),t1=fu,event=dead,
data=dat,method="LCV"))

print(summary(survPen.tensor))

cat("\n","survPen, execution time ", time.survPen[3], "\n")


#-------------- rstpm2
cat("\n","\n","_____________________________________________________________________", "\n")
cat("\n","rstpm2 ","\n") 

time.rstpm2 <- system.time(rstpm2.tensor <- pstpm2(Surv(fu,dead)~1, data=dat, 
smooth.formula = ~te(log(fu),age,bs="cr",k=df2),
tvc = NULL,
bhazard = NULL,
sp=NULL,
criterion="GCV",
link.type="PH"))

print(summary(rstpm2.tensor))

cat("\n","rstpm2, execution time ", time.rstpm2[3], "\n")


#-------------- mgcv
cat("\n","\n","_____________________________________________________________________", "\n")
cat("\n","mgcv ","\n") 


time.mgcv <- system.time(mgcv.tensor <- gam(dead~te(fu,age,bs="cr",k=df2),offset=log(tik),
		family=poisson,data=split_dat,scale=1,method="GCV.Cp" 
		))	  
			
		
print(summary(mgcv.tensor))

cat("\n","mgcv, execution time ", time.mgcv[3], "\n")


#_____________________________________________________________________
# Checking predictions	

ntime <- seq(0.001,5,length=100)
nage <- quantile(dat$age,c(0,0.1,0.25,0.4,0.5,0.6,0.75,0.9,1))

ndata<-expand.grid(fu=ntime,age=nage)
ndata$haz.survPen <- predict(survPen.tensor,ndata)$haz
ndata$haz.rstpm2 <- predict(rstpm2.tensor,ndata,type="hazard")
ndata$haz.mgcv <- predict(mgcv.tensor,ndata,type="response")


par(mfrow=c(3,3))
for (age in nage){

condition <- ndata$age == age

plot(ntime,ndata[condition,]$haz.survPen,type="l",col="blue3",
main=paste("age",round(age,0)),ylab="hazard",xlab="time",lwd=2)
lines(ntime,ndata[condition,]$haz.rstpm2,col="red",lwd=2)
lines(ntime,ndata[condition,]$haz.mgcv,col="green3",lwd=2,lty=2)

}
legend("topright",c("survPen","rstpm2","mgcv"),lty=c(1,1,2),
col=c("blue3","red","green3"),lwd=rep(2,3))


# survPen and mgcv show the same predictions (as they should) while rstpm2 differs and shows some
# irregularities in early follow-up (this can be explained by the facts that : i) rstpm2 has its linear predictor
# on the log-cumulative hazard scale, ii) rstpm2 relies on numerical derivatives in its optimization scheme).
# Besides, survPen provides the fastest implementation.





