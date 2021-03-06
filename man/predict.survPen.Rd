% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/survPenV1_51.r
\name{predict.survPen}
\alias{predict.survPen}
\title{Hazard and Survival prediction from fitted \code{survPen} model}
\usage{
\method{predict}{survPen}(
  object,
  newdata,
  newdata.ref = NULL,
  n.legendre = 50,
  conf.int = 0.95,
  do.surv = TRUE,
  type = "standard",
  exclude.random = FALSE,
  get.deriv.H = FALSE,
  ...
)
}
\arguments{
\item{object}{a fitted \code{survPen} object as produced by \code{\link{survPen.fit}}}

\item{newdata}{data frame giving the new covariates value}

\item{newdata.ref}{data frame giving the new covariates value for the reference population (used only when type="HR")}

\item{n.legendre}{number of nodes to approximate the cumulative hazard by Gauss-Legendre quadrature; default is 50}

\item{conf.int}{numeric value giving the precision of the confidence intervals; default is 0.95}

\item{do.surv}{If TRUE, the survival and its lower and upper confidence values are computed. Survival computation requires numerical integration and can be time-consuming so if you only want the hazard use do.surv=FALSE; default is TRUE}

\item{type, }{if type="lpmatrix" returns the design matrix (or linear predictor matrix) corresponding to the new values of the covariates; if equals "HR", returns the predicted HR and CIs between newdata and newdata.ref; default is "standard" for classical hazard and survival estimation}

\item{exclude.random}{if TRUE all random effects are set to zero; default is FALSE}

\item{get.deriv.H}{if TRUE, the derivatives wrt to the regression parameters of the cumulative hazard are returned; default is FALSE}

\item{...}{other arguments}
}
\value{
List of objects:
\item{haz}{hazard predicted by the model}
\item{haz.inf}{lower value for the confidence interval on the hazard based on the Bayesian covariance matrix Vp (Wood et al. 2016)}
\item{haz.sup}{Upper value for the confidence interval on the hazard based on the Bayesian covariance matrix Vp}
\item{surv}{survival predicted by the model}
\item{surv.inf}{lower value for the confidence interval on the survival based on the Bayesian covariance matrix Vp}
\item{surv.sup}{Upper value for the confidence interval on the survival based on the Bayesian covariance matrix Vp}
\item{deriv.H}{derivatives wrt to the regression parameters of the cumulative hazard. Useful to calculate standardized survival}
\item{HR}{predicted hazard ratio ; only when type = "HR"}
\item{HR.inf}{lower value for the confidence interval on the hazard ratio based on the Bayesian covariance matrix Vp  ; only when type = "HR"}
\item{HR.sup}{Upper value for the confidence interval on the hazard ratio based on the Bayesian covariance matrix Vp  ; only when type = "HR"}
}
\description{
Takes a fitted \code{survPen} object and produces hazard and survival predictions given a new set of values for the model covariates.
}
\details{
The confidence intervals noted CI.U are built on the log cumulative hazard scale U=log(H) (efficient scale in terms of respect towards the normality assumption)
using Delta method. The confidence intervals on the survival scale are then \code{CI.surv = exp(-exp(CI.U))}
}
\examples{

library(survPen)

data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer

# model : unidimensional penalized spline for time since diagnosis with 5 knots
f1 <- ~smf(fu,df=5)

# hazard model
mod1 <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LAML")

# predicting hazard and survival at time 1
pred <- predict(mod1,data.frame(fu=1))
pred$haz
pred$surv

# predicting hazard ratio between age 70 and age 30
pred.HR <- predict(mod1,data.frame(fu=1,age=70),newdata.ref=data.frame(fu=1,age=30),type="HR")
pred.HR$HR
pred.HR$HR.inf
pred.HR$HR.sup

}
\references{
Wood, S.N., Pya, N. and Saefken, B. (2016), Smoothing parameter and model selection for general smooth models (with discussion). Journal of the American Statistical Association 111, 1548-1575
}
