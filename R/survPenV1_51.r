#--------------------------------------------------------------------------------------------------------------------------
# Penalized (excess) hazard model for time-to-event data
#--------------------------------------------------------------------------------------------------------------------------


#' @useDynLib survPen
#' @importFrom Rcpp sourceCpp
NULL


#----------------------------------------------------------------------------------------------------------------
# datCancer : simulated cancer dataset
#----------------------------------------------------------------------------------------------------------------

#' Patients diagnosed with cervical cancer
#'
#' A simulated dataset containing the follow-up times of 2000 patients diagnosed with cervical cancer between 
#' 1990 and 2010. End of follow-up is June 30th 2013. The variables are as follows:
#' \itemize{
#'   \item begin. beginning of follow-up. For illustration purposes about left truncation only (0--1)
#'   \item fu. follow-up time in years (0--5)
#'   \item age. age at diagnosis in years, from 21.39 to 99.33
#'   \item yod. decimal year of diagnosis, from 1990.023 to 2010.999
#'   \item dead. censoring indicator (1 for dead, 0 for censored)
#'   \item rate. expected mortality rate (from overall mortality of the general population) (0--0.38)
#' }
#' @docType data
#' @keywords datasets
#' @name datCancer
#' @usage data(datCancer)
#' @format A data frame with 2000 rows and 6 variables
NULL


#----------------------------------------------------------------------------------------------------------------
# END of code : datCancer
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# survPenObject : description of the object returned by function survPen
#----------------------------------------------------------------------------------------------------------------

#' Fitted survPen object
#'
#' A fitted survPen object returned by function \code{\link{survPen}} and of class "survPen". 
#' Method functions predict and summary are available for this class.
#'
#' @return A \code{survPen} object has the following elements:
#' \item{call}{original \code{survPen} call}
#' \item{formula}{formula object specifying the model}
#' \item{t0.name}{name of the vector of origin times}
#' \item{t1.name}{name of the vector of follow-up times}
#' \item{event.name}{name of the vector of right-censoring indicators}
#' \item{expected.name}{name of the vector of expected hazard}
#' \item{haz}{fitted hazard}
#' \item{coefficients}{estimated regression parameters. Unpenalized parameters are first, followed by the penalized ones}
#' \item{type}{"net" for net survival estimation with penalized excess hazard model or "overall" for overall survival with penalized hazard model}
#' \item{df.para}{degrees of freedom associated with fully parametric terms (unpenalized)}
#' \item{df.smooth}{degrees of freedom associated with penalized terms}
#' \item{p}{number of regression parameters}
#' \item{edf}{effective degrees of freedom}
#' \item{edf1}{alternative effective degrees of freedom ; used as an upper bound for edf2}
#' \item{edf2}{effective degrees of freedom corrected for smoothing parameter uncertainty}
#' \item{aic}{Akaike information criterion with number of parameters replaced by edf when there are penalized terms. Corresponds to 2*edf - 2*ll.unpen}
#' \item{aic2}{Akaike information criterion corrected for smoothing parameter uncertainty. Be careful though, this is still a work in progress, especially when one of the smoothing parameters tends to infinity.}
#' \item{iter.beta}{vector of numbers of iterations needed to estimate the regression parameters for each smoothing parameters trial. It thus contains \code{iter.rho+1} elements.}
#' \item{X}{design matrix of the model}
#' \item{S}{penalty matrix of the model}
#' \item{S.scale}{vector of rescaling factors for the penalty matrices}
#' \item{S.list}{Equivalent to pen but with every element multiplied by its associated smoothing parameter}
#' \item{S.smf}{List of penalty matrices associated with all "smf" calls}
#' \item{S.tensor}{List of penalty matrices associated with all "tensor" calls}
#' \item{S.tint}{List of penalty matrices associated with all "tint" calls}
#' \item{S.rd}{List of penalty matrices associated with all "rd" calls}
#' \item{smooth.name.smf}{List of names for the "smf" calls associated with S.smf}
#' \item{smooth.name.tensor}{List of names for the "tensor" calls associated with S.tensor}
#' \item{smooth.name.tint}{List of names for the "tint" calls associated with S.tint}
#' \item{smooth.name.rd}{List of names for the "rd" calls associated with S.rd}
#' \item{S.pen}{List of all the rescaled penalty matrices redimensioned to df.tot size. Every element of \code{S.pen} noted \code{S.pen[[i]]} is made from a penalty matrix \code{pen[[i]]} returned by
#' \code{\link{smooth.cons}} and is multiplied by S.scale}
#' \item{grad.unpen.beta}{gradient vector of the log-likelihood with respect to the regression parameters}
#' \item{grad.beta}{gradient vector of the penalized log-likelihood with respect to the regression parameters}
#' \item{Hess.unpen.beta}{hessian of the log-likelihood with respect to the regression parameters}
#' \item{Hess.beta}{hessian of the penalized log-likelihood with respect to the regression parameters}
#' \item{Hess.beta.modif}{if TRUE, the hessian of the penalized log-likelihood has been perturbed at convergence}
#' \item{ll.unpen}{log-likelihood at convergence}
#' \item{ll.pen}{penalized log-likelihood at convergence}
#' \item{deriv.rho.beta}{transpose of the Jacobian of beta with respect to the log smoothing parameters}
#' \item{deriv.rho.inv.Hess.beta}{list containing the derivatives of the inverse of \code{Hess} with respect to the log smoothing parameters}
#' \item{deriv.rho.Hess.unpen.beta}{list containing the derivatives of \code{Hess.unpen} with respect to the log smoothing parameters}
#' \item{lambda}{estimated or given smoothing parameters}
#' \item{nb.smooth}{number of smoothing parameters}
#' \item{iter.rho}{number of iterations needed to estimate the smoothing parameters}
#' \item{optim.rho}{identify whether the smoothing parameters were estimated or not; 1 when exiting the function \code{\link{NR.rho}}; default is NULL}
#' \item{method}{criterion used for smoothing parameter estimation}
#' \item{criterion.val}{value of the criterion used for smoothing parameter estimation at convergence}
#' \item{LCV}{Likelihood cross-validation criterion at convergence}
#' \item{LAML}{negative Laplace approximate marginal likelihood at convergence}
#' \item{grad.rho}{gradient vector of criterion with respect to the log smoothing parameters}
#' \item{Hess.rho}{hessian matrix of criterion with respect to the log smoothing parameters}
#' \item{inv.Hess.rho}{inverse of \code{Hess.rho}}
#' \item{Hess.rho.modif}{if TRUE, the hessian of LCV or LAML has been perturbed at convergence}
#' \item{Ve}{Frequentist covariance matrix}
#' \item{Vp}{Bayesian covariance matrix}
#' \item{Vc}{Bayesian covariance matrix corrected for smoothing parameter uncertainty (see Wood et al. 2016)}
#' \item{Vc.approx}{Kass and Steffey approximation of \code{Vc} (see Wood et al. 2016)}
#' \item{Z.smf}{List of matrices that represents the sum-to-zero constraint to apply for \code{\link{smf}} splines}
#' \item{Z.tensor}{List of matrices that represents the sum-to-zero constraint to apply for \code{\link{tensor}} splines}
#' \item{Z.tint}{List of matrices that represents the sum-to-zero constraint to apply for \code{\link{tint}} splines}
#' \item{list.smf}{List of all \code{smf.smooth.spec} objects contained in the model}
#' \item{list.tensor}{List of all \code{tensor.smooth.spec} objects contained in the model}
#' \item{list.tint}{List of all \code{tint.smooth.spec} objects contained in the model}
#' \item{list.rd}{List of all \code{rd.smooth.spec} objects contained in the model}
#' \item{U.F}{Eigen vectors of S.F, useful for the initial reparameterization to separate penalized ad unpenalized subvectors. Allows stable evaluation of the log determinant of S and its derivatives}
#' \item{factor.structure}{List containing the levels and classes of all factor variables present in the data frame used for fitting}
#' \item{converged}{convergence indicator, TRUE or FALSE. TRUE if Hess.beta.modif=FALSE and Hess.rho.modif=FALSE (or NULL)}
#'
#' @references
#' Wood, S.N., Pya, N. and Saefken, B. (2016), Smoothing parameter and model selection for general smooth models (with discussion). Journal of the American Statistical Association 111, 1548-1575
#'
#' @name survPenObject
#'
NULL

#----------------------------------------------------------------------------------------------------------------
# END of code : survPenObject
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# tensor.in : constructs the design matrix for a tensor product from two marginals design matrices
#----------------------------------------------------------------------------------------------------------------

#' tensor model matrix for two marginal bases
#'
#' Function called recursively inside \code{\link{tensor.prod.X}}.
#'
#' @param X1 first marginal design matrix with n rows and p1 columns
#' @param X2 first marginal design matrix with n rows and p2 columns
#' @return Matrix of dimensions n*(p1*p2) representing the row tensor product of the matrices X1 and X2
#' @export
#'
#' @examples
#'
#' library(survPen)
#' 
#' # row-wise tensor product between two design matrices
#' set.seed(15)
#'
#' X1 <- matrix(rnorm(10*3),nrow=10,ncol=3)
#' X2 <- matrix(rnorm(10*2),nrow=10,ncol=2)
#' tensor.in(X1,X2)
#' 
tensor.in <- function(X1,X2){

	# each column of X1 is multiplied by all the columns of X2
	l <- lapply(1:ncol(X1),function(i) {X1[,i]*X2})
	do.call(cbind, l)
	
}

#----------------------------------------------------------------------------------------------------------------
# END of code : tensor.in
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# tensor.prod.X : constructs the design matrix for a tensor product from the marginals design matrices
#----------------------------------------------------------------------------------------------------------------

#' tensor model matrix
#'
#' Computes the model matrix of tensor product smooth from the marginal bases.
#'
#' @param X list of m design matrices with n rows and p1, p2, ... pm columns respectively
#' @return
#' \item{T}{Matrix of dimensions n*(p1*p2*...*pm) representing the row tensor product of the matrices in X}
#' @export
#'
#' @examples
#'
#' library(survPen)
#'
#' # row-wise tensor product between three design matrices
#' set.seed(15)
#'
#' X1 <- matrix(rnorm(10*3),nrow=10,ncol=3)
#' X2 <- matrix(rnorm(10*2),nrow=10,ncol=2)
#' X3 <- matrix(rnorm(10*2),nrow=10,ncol=2)
#' tensor.prod.X(list(X1,X2,X3))
#' 
tensor.prod.X <- function (X) 
{
    m <- length(X) # number of matrices
	
	if(m>1){
	
		# starts with the first two matrices
		T <- tensor.in(X[[1]],X[[2]])

		if(m>2){ # repeats the function tensor.in from the previous result, one matrix at a time
			for (j in 3:m){
			
				T <- tensor.in(T,X[[m]])
			
			}
		}
		
	}else{
		# if there is only one matrix in the list X, we return that matrix
		T <- X[[1]]
	
	}

    T
}

#----------------------------------------------------------------------------------------------------------------
# END of code : tensor.prod.X
#----------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------
# tensor.prod.S : equivalent to function tensor.prod.penalties from mgcv package
#----------------------------------------------------------------------------------------------------------------

#' Tensor product for penalty matrices
#'
#' Computes the penalty matrices of a tensor product smooth from the marginal penalty matrices. The code is from
#' function \code{tensor.prod.penalties} in \code{mgcv} package.
#'
#' @param S list of m marginal penalty matrices
#' @return
#' \item{TS}{List of the penalty matrices associated with the tensor product smooth}
#' @export
#'
#' @examples
#'
#' library(survPen)
#'
#' # tensor product between three penalty matrices
#' set.seed(15)
#'
#' S1 <- matrix(rnorm(3*3),nrow=3,ncol=3)
#' S2 <- matrix(rnorm(2*2),nrow=2,ncol=2)
#' 
#' S1 <- 0.5*(S1 + t(S1) ) ; S2 <- 0.5*(S2 + t(S2) )
#'
#' tensor.prod.S(list(S1,S2))
#' 
tensor.prod.S <- function (S) 
{
    m <- length(S)
    I <- vector("list", m)
    for (i in 1:m) {
        n <- ncol(S[[i]])
        I[[i]] <- diag(n)
    }
    TS <- vector("list", m)
    if (m == 1) 
        TS[[1]] <- S[[1]]
    else for (i in 1:m) {
        if (i == 1) 
            M0 <- S[[1]]
        else M0 <- I[[1]]
        for (j in 2:m) {
            if (i == j) 
                M1 <- S[[i]]
            else M1 <- I[[j]]
            M0 <- M0 %x% M1
        }
        TS[[i]] <- if (ncol(M0) == nrow(M0)) 
            (M0 + t(M0))/2
        else M0
    }
    TS
}

#----------------------------------------------------------------------------------------------------------------
# END of code : tensor.prod.S
#----------------------------------------------------------------------------------------------------------------




#----------------------------------------------------------------------------------------------------------------
# crs : bases for cubic regression splines (equivalent to the "cr" in mgcv)
#----------------------------------------------------------------------------------------------------------------

#' Bases for cubic regression splines (equivalent to "cr" in \code{mgcv})
#'
#' Builds the design matrix and the penalty matrix for cubic regression splines.
#'
#' @param x Numeric vector
#' @param knots Numeric vectors that specifies the knots of the splines (including boundaries); default is NULL
#' @param df numeric value that indicates the number of knots desired (or degrees of freedom) if knots=NULL; default is 10
#' @param intercept if FALSE, the intercept is excluded from the basis; default is TRUE
#' @details
#' See package \code{mgcv} and section 4.1.2 of Wood (2006) for more details about this basis
#' @return List of three elements
#' \item{bs}{design matrix}
#' \item{pen}{penalty matrix}
#' \item{knots}{vector of knots (specified or calculated from \code{df})}
#' @export
#'
#' @references
#' Wood, S. N. (2006), Generalized additive models: an introduction with R. London: Chapman & Hall/CRC.
#'
#' @examples
#' x <- seq(1,10,length=100)
#' # natural cubic spline with 3 knots
#' crs(x,knots=c(1,5,10))
#'
crs <- function(x, knots=NULL,df=10, intercept=TRUE) {

  n <- length(x)

  if (is.null(knots)) # spacing knots through data using quantiles if knots are unspecified
  {
    if (is.null(df)) {df <- 10}
    if(df<3) {stop("Number of knots should be at least 3, 1 interior plus 2 boundaries")}

    if(n<2) {stop("Please specify at least 2 values or specify at least 3 knots via knots=...")}
    knots <- stats::quantile(unique(x),seq(0,1,length=df))
    # if you don't use unique(x), the penalization matrix can be non positive semi definite
  }

  k <- length(knots)

  if (k<3) {stop("Please specify at least 3 knots, 1 interior plus 2 boundaries")}

  knots <- sort(knots)

  h <- diff(knots)

  F.P <- crs.FP(knots,h)

  # matrix mapping beta to delta (the values at the knots of the second derivative of the splines, see section 4.1.2
  # of )
  F.mat <- F.P$F.mat
  F.mat1 <- rbind(rep(0,k),F.mat)
  F.mat2 <- rbind(F.mat,rep(0,k))

  # penalty matrix
  P.mat <- F.P$P.mat 

  # to project beyond the boundaries
  condition.min <- rep(0,n)
  condition.min[x<min(knots)] <- 1

  condition.max <- rep(0,n)
  condition.max[x>max(knots)] <- 1

  x.min <- x[condition.min==1]

  x.max <- x[condition.max==1]

  len.min <- length(x.min)
  
  len.max <- length(x.max)
  
  x[condition.min==1] <- min(knots)

  x[condition.max==1] <- max(knots)

  # interval condition matrix (to which interval belongs each x value)
  condition <- matrix(0,nrow=n,ncol=k-1)
  
  for (l in 1:(k-1)){

    # Careful to the knots that belong to two intervals : we define the spline function on
    # right-open intervals
    condition[(x >= knots[l]) & (x < knots[l+1]),l] <- 1
	
  }
  
  # position bases
  a.minus <- sapply(1:(k-1),function(l) (knots[l+1]-x)/h[l])
  a.plus <- sapply(1:(k-1),function(l) (x-knots[l])/h[l])
  
  # curvature bases
  c.minus <- 1/6*sapply(1:(k-1),function(l) (knots[l+1]-x)^3/h[l]  -  h[l]*(knots[l+1]-x))
  c.plus <- 1/6*sapply(1:(k-1),function(l) (x-knots[l])^3/h[l]  -  h[l]*(x-knots[l]))
  
  # we multiply by every interval condition
  a.minus <- a.minus*condition
  a.plus <- a.plus*condition
  
  c.minus <- c.minus*condition
  c.plus <- c.plus*condition
  
  # position bases conditions
  Ident <- diag(k-1)
  Mat_j <- cbind(Ident,rep(0,k-1))
  Mat_j_1 <- cbind(rep(0,k-1),Ident)
  
  # bases
  b <- c.minus%mult%F.mat1 + c.plus%mult%F.mat2 + a.minus%mult%Mat_j + a.plus%mult%Mat_j_1
  
  # Since we defined the spline function on right-open intervals, the last knots is not taken into account.
  # That's why we add 1 manually into the design matrix
  if(any(x == max(knots))) b[x == max(knots), k] <- 1
  
  # to project beyond the boundaries
  if (sum(condition.min)>0){
  
	v1 <- (x.min-min(knots))
	v2 <- -h[1]/6*F.mat[1,] - 1/h[1]*c(1,rep(0,k-1)) + 1/h[1]*c(0,1,rep(0,k-2))
	
	b[condition.min==1,] <- b[condition.min==1,] +
	matrix(v1*rep(v2,each=len.min),nrow=len.min,ncol=k)
	
  }
  
  if (sum(condition.max)>0){
  
	v1 <- (x.max-max(knots))
	v2 <- h[k-1]/6*F.mat[k-2,] - 1/h[k-1]*c(rep(0,k-2),1,0) + 1/h[k-1]*c(rep(0,k-1),1)
	
	b[condition.max==1,] <- b[condition.max==1,] +
	matrix(v1*rep(v2,each=len.max),nrow=len.max,ncol=k)
	
  }
  
  if(intercept == FALSE) {
    return(list(bs=b[,-1],pen=P.mat[-1,-1],knots=knots))
  } else {
    return(list(bs=b,pen=P.mat,knots=knots))
  }

}

#----------------------------------------------------------------------------------------------------------------
# END of code : crs
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# crs.FP : called inside crs to get the penalty matrix
#----------------------------------------------------------------------------------------------------------------

#' Penalty matrix constructor for cubic regression splines
#'
#' constructs the penalty matrix associated with cubic regression splines basis. This function is called inside
#' \code{\link{crs}}.
#'
#' @param knots Numeric vectors that specifies the knots of the splines (including boundaries)
#' @param h vector of knots differences (corresponds to \code{diff(sort(knots))})
#' @return List of two elements:
#' \item{F.mat}{matrix used in function \code{\link{crs}} for basis construction}
#' \item{P.mat}{penalty matrix}
#' @export
#'
#' @examples
#'
#' library(survPen)
#'
#' # construction of the penalty matrix using a sequence of knots
#' knots <- c(0,0.25,0.5,0.75,1)
#' diff.knots <- diff(knots)
#'
#' crs.FP(knots,diff.knots)
#' 
crs.FP <- function(knots,h){
  # constraints of second derivatives continuity and nullity beyond the boundaries

  k <- length(knots)

  if (k<3) {stop("Please specify at least 3 knots, 1 interior plus 2 boundaries")}
  
  B <- matrix(0,nrow=k-2,ncol=k-2)

  D <- matrix(0,nrow=k-2,ncol=k)

  for (i in 1:(k-2)){

    D[i,i] <- 1/h[i]

    D[i,i+1] <- -1/h[i]-1/h[i+1]

    D[i,i+2] <- 1/h[i+1]

    B[i,i] <- (h[i]+h[i+1])/3

	if(i<(k-2)){
	
		B[i,i+1] <- B[i+1,i] <- h[i+1]/6
		
	}
	  
  }

  F.mat <- chol2inv(chol(B))%mult%D
  
  P.mat <- D%cross%F.mat # penalty matrix

  P.mat <- (P.mat+t(P.mat))*0.5 # to make sure the penalty matrix is symmetric

  return(list(F.mat=F.mat,P.mat=P.mat))

}

#----------------------------------------------------------------------------------------------------------------
# END of code : crs.FP
#----------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------
# smf, tensor and tint : key words to put inside a formula object to define penalized splines
# Multiple "smf", "tensor" or "tint" calls are accepted
#----------------------------------------------------------------------------------------------------------------

#' Defining smooths in survPen formulae
#'
#' Used inside a formula object to define a smooth, a tensor product smooth or a tensor product interaction. 
#' Natural cubic regression splines (linear beyond the knots, equivalent to \code{ns} from package \code{splines}) are used as marginal bases. While \code{tensor} builds a tensor product of marginal bases including 
#' the intercepts, \code{tint} applies a tensor product of the marginal bases without their intercepts.
#' Unlike \code{tensor}, the marginal effects of the covariates should also be present in the formula when using \code{tint}.
#' For a conceptual difference between tensor products and tensor product interactions see Section 5.6.3 from Wood (2017) 
#'
#' @param ... Any number of covariates separated by ","
#' @param knots numeric vector that specifies the knots of the splines (including boundaries); default is NULL, in which case the knots are spread through the covariate values using quantiles. Precisely, for the term "smf(x,df=df1)", the vector of knots will be: quantile(unique(x),seq(0,1,length=df1))
#' @param df numeric value that indicates the number of knots (or degrees of freedom) desired; default is NULL. If knots and df are NULL, df will be set to 10
#' @param by numeric or factor variable in order to define a varying coefficient smooth
#' @param same.rho if the specified by variable is a factor, specifies whether the smoothing parameters should be the same for all levels; default is FALSE.
#' @return object of class \code{smf.smooth.spec}, \code{tensor.smooth.spec} or \code{tint.smooth.spec}  (see \code{\link{smooth.spec}} for details)
#'
#' @export
#'
#' @references
#' Wood, S. N. (2017), Generalized additive models: an introduction with R. Second Edition. London: Chapman & Hall/CRC.
#'
#' @examples
#' # penalized cubic regression spline of time with 5 unspecified knots
#' formula.test <- ~smf(time,df=5)
#'
#' # suppose that we want to fit a model from formula.test
#' library(survPen)
#' data(datCancer)
#'
#' mod.test <- survPen(~smf(fu,df=5) ,data=datCancer,t1=fu,event=dead)
#'
#' # then the knots can be retrieved like this:
#' mod.test$list.smf[[1]]$knots
#
#' # or calculated like this
#' quantile(unique(datCancer$fu),seq(0,1,length=5))
#'
#'
#' # penalized cubic regression splines of time and age with respectively 5 and 7 unspecified knots
#' formula.test2 <- ~smf(time,df=5)+smf(age,df=7)
#'
#' # penalized cubic regression splines of time and age with respectively 3 and 4 specified knots
#' formula.test3 <- ~smf(time,knots=c(0,3,5))+smf(age,knots=c(30,50,70,90))
#'
#' # penalized tensor product for time and age with respectively 5 and 4 unspecified knots leading
#' # to 5*4 = 20 regression parameters
#' formula.test <- ~tensor(time,age,df=c(5,4))
#'
#' # penalized tensor product for time and age with respectively 3 and 4 specified knots
#' formula.test3 <- ~tensor(time,agec,knots=list(c(0,3,5),c(30,50,70,90)))
#'
#' # penalized tensor product for time, age and year with respectively 6, 5 and 4 unspecified knots
#' formula.test <- ~tensor(time,age,year,df=c(6,5,4))
#'
#' # penalized tensor product interaction for time and age with respectively 5 and 4 unspecified knots
#' # main effects are specified as penalized cubic regression splines
#' formula.test <- ~smf(time,df=5)+smf(age,df=4)+tint(time,age,df=c(5,4))
#'
smf <- function(..., knots=NULL,df=NULL,by=NULL,same.rho=FALSE){

	by <- substitute(by)
	
	if(!is.character(by)) by <- deparse(by)

	smooth.spec(..., knots=knots,df=df,by=by,option="smf",same.rho=same.rho)

}

#' @rdname smf
tensor <- function(..., knots=NULL,df=NULL,by=NULL,same.rho=FALSE){

	by <- substitute(by)
	
	if(!is.character(by)) by <- deparse(by)

	smooth.spec(..., knots=knots,df=df,by=by,option="tensor",same.rho=same.rho)

}

#' @rdname smf
tint <- function(..., knots=NULL,df=NULL,by=NULL,same.rho=FALSE){

	by <- substitute(by)
	
	if(!is.character(by)) by <- deparse(by)

	smooth.spec(..., knots=knots,df=df,by=by,option="tint",same.rho=same.rho)

}


#----------------------------------------------------------------------------------------------------------------
# End of code : smf, tensor and tint
#----------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------
# rd : key word to specify multinormal iid random effects
#----------------------------------------------------------------------------------------------------------------

#' Defining random effects in survPen formulae
#'
#' Used inside a formula object to define a random effect.
#'
#' @param ... Any number of covariates separated by ","
#' @return object of class \code{rd.smooth.spec}
#'
#' @export
#'
#' @examples
#' # cubic regression spline of time with 10 unspecified knots + random effect at the cluster level
#' formula.test <- ~smf(time,df=10) + rd(cluster)
#'
#'
rd <- function(...){

	smooth.spec(...,option="rd")

}

#----------------------------------------------------------------------------------------------------------------
# End of code : rd
#----------------------------------------------------------------------------------------------------------------





#----------------------------------------------------------------------------------------------------------------
# smooth.spec : function called by the wrappers smf, tensor and tint
# The function does not construct any bases or penalty matrices, it just specifies the covariates
# that will be dealt as penalized splines, the dimensions of those splines, plus all the knots and
# degrees of freedom
#----------------------------------------------------------------------------------------------------------------

#' Covariates specified as penalized splines
#'
#' Specifies the covariates to be considered as penalized splines.
#'
#' @param ... Numeric vectors specified in \code{\link{smf}}, \code{\link{tensor}} or \code{\link{tint}}
#' @param knots List of numeric vectors that specifies the knots of the splines (including boundaries); default is NULL
#' @param df Degrees of freedom: numeric vector that indicates the number of knots desired for each covariate; default is NULL
#' @param by numeric or factor variable in order to define a varying coefficient smooth; default is NULL
#' @param option "smf", "tensor" or "tint". Depends on the wrapper function; default is "smf"
#' @param same.rho if there is a factor by variable, should the smoothing parameters be the same for all levels; default is FALSE.
#' @return object of class smooth.spec
#' \item{term}{Vector of strings giving the names of each covariate specified in ...}
#' \item{dim}{Numeric value giving the number of covariates associated with this spline}
#' \item{knots}{list of numeric vectors that specifies the knots for each covariate}
#' \item{df}{Numeric vector giving the number of knots associated with each covariate}
#' \item{by}{numeric or factor variable in order to define a varying coefficient smooth}
#' \item{same.rho}{if there is a factor by variable, should the smoothing parameters be the same for all levels; default is FALSE}
#' \item{name}{simplified name of the call to function smooth.spec}
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 10 unspecified knots
#' smooth.spec(time)
#'
#' # tensor of time and age with 5*5 specified knots
#' smooth.s <- smooth.spec(time,age,knots=list(time=seq(0,5,length=5),age=seq(20,80,length=5)),
#' option="tensor")
#'
smooth.spec <- function(..., knots=NULL,df=NULL,by=NULL,option=NULL,same.rho=FALSE){

  if (is.null(option)) {
    option <- "smf"
  }else{
    if (!option %in% c("tensor","smf","tint","rd")) stop("option must be : smf, tensor, tint or rd")
  }

  # By default, variables specified as "smf" splines get 10 degrees of freedom,
  # varibales specified inside "tensor" calls get 5
  if (option!="smf") {
    df.def <- 5

  }else{

    df.def <- 10
  }

  # We get information about the number of covariates and their names
  vars <- as.list(substitute(list(...)))[-1]
  dim <- length(vars)
  name <- paste0(option,"(",paste(vars,collapse=","),")") # simplified name of the call
   
  term <- deparse(vars[[1]], backtick = TRUE)
  if (dim > 1) {
    for (i in 2:dim) term[i] <- deparse(vars[[i]], backtick = TRUE)
  }
  for (i in 1:dim) term[i] <- attr(stats::terms(stats::reformulate(term[i])), "term.labels")

  #if (length(unique(term)) != dim) {
   # stop("Repeated variables as arguments of a smooth are not permitted")
  #}


  if (option=="rd"){

		spec <- list(term=term,dim=dim,knots=NULL,df=NULL,by=NULL,same.rho=NULL,name=name)
		class(spec) <- paste(option,".smooth.spec",sep="")
		return(spec)

  }
  
  
  if (is.null(knots)) {

	if (is.null(df)) {
		
		df <- rep(df.def,dim)
			
	}else{
		
		if (length(df)!=dim){
			
			df <- rep(df.def,dim)
			warning("wrong df length, df put to ",df.def," for each covariate")
				
		}
		
	}
		
  }else{

    if(!is.list(knots)) {

        if(dim>1) stop("knots must be a list argument")

        knots <- list(knots)

    }

    if (length(knots)!=dim){
	
        df <- rep(df.def,dim)
        knots <- NULL
        warning("wrong list of knots, df put to ",df.def," for each covariate and quantiles used")
		
    }else{

		df.temp	<- sapply(knots,FUN=length)
        
		if (is.null(df)) {
		
			df <- df.temp
		
		}else{
		
			if (any(df!=df.temp)){
		
				df <- df.temp
				if (all(df>2)) warning("wrong df, df put to ",df.temp)
		
			}
		}
		
    }

  }

  spec <- list(term=term,dim=dim,knots=knots,df=df,by=by,same.rho=same.rho,name=name)
  class(spec) <- paste(option,".smooth.spec",sep="")
  spec
}

#----------------------------------------------------------------------------------------------------------------
# END of code : smooth.spec
#----------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------
# smooth.cons : for each penalized spline, the function builds the design and penalty matrices
#----------------------------------------------------------------------------------------------------------------

#' Design and penalty matrices of penalized splines in a smooth.spec object
#'
#' Builds the design and penalty matrices from the result of \code{\link{smooth.spec}}.
#' @param term Vector of strings that generally comes from the value "term" of a \code{smooth.spec} object.
#' @param knots List of numeric vectors that specifies the knots of the splines (including boundaries).
#' @param df Degrees of freedom: numeric vector that indicates the number of knots desired for each covariate.
#' @param by numeric or factor variable in order to define a varying coefficient smooth; default is NULL.
#' @param option "smf", "tensor" or "tint".
#' @param data.spec data frame that represents the environment from which the covariate values and knots are to be calculated; default is NULL.
#' @param same.rho if there is a factor by variable, should the smoothing parameters be the same for all levels; default is FALSE.
#' @param name simplified name of the smooth.spec call.
#' @return List of objects with the following items:
#' \item{X}{Design matrix}
#' \item{pen}{List of penalty matrices}
#' \item{term}{Vector of strings giving the names of each covariate}
#' \item{knots}{list of numeric vectors that specifies the knots for each covariate}
#' \item{dim}{Number of covariates}
#' \item{all.df}{Numeric vector giving the number of knots associated with each covariate}
#' \item{sum.df}{Sum of all.df}
#' \item{Z.smf}{List of matrices that represents the sum-to-zero constraint to apply for "smf" splines}
#' \item{Z.tensor}{List of matrices that represents the sum-to-zero constraint to apply for "tensor" splines}
#' \item{Z.tint}{List of matrices that represents the sum-to-zero constraint to apply for "tint" splines}
#' \item{lambda.name}{name of the smoothing parameters}
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 4 knots (so we get a design matrix with 3 columns 
#' # because of centering constraint)
#'
#' data <- data.frame(time=seq(0,5,length=100))
#' smooth.c <- smooth.cons("time",knots=list(c(0,1,3,5)),df=4,option="smf",
#' data.spec=data,name="smf(time)")
#'
smooth.cons <- function(term, knots, df, by=NULL, option, data.spec, same.rho=FALSE, name){

  if (option=="rd"){
  
	dim <- 1
	
	pen <- vector("list", dim)
	
	X <- stats::model.matrix(stats::as.formula(paste0("~",paste(term,collapse=":"),"-1")),data=data.spec)
	
	n.col <- dim(X)[2]
	pen[[1]] <- diag(n.col)
	
	colnames(X) <- rep(paste(name,".",1:n.col,sep=""))
		
	lambda.name <- rep(name)
		
	
	res <- list(X=X,pen=pen,term=term,dim=dim,lambda.name=lambda.name)
	
	return(res)
	
  }
  
  centering <- TRUE

  dim <- length(term)

  # for by variables
  if(!is.character(by)) by <- deparse(by)
  by.var <- eval(parse(text=by),envir=as.environment(data.spec))
  
  if (!is.null(by.var)){ 
	# for a numeric by variable, no centering constraint is applied
	if(!is.factor(by.var)) centering <- FALSE
  }
  
  all.df <- if(!is.null(knots)){sapply(knots,length)}else{df}
  # for "smf" splines, we must remove the intercept for all covariates but one (except when centering is FALSE)
  all.df <- all.df+rep(if(option=="smf" & centering){-1}else{0},dim)
  sum.df <- sum(all.df)

  Base <- vector("list", dim) # list containing all design and penalty matrices for each covariate
  bs <- vector("list", dim) # design matrices
  pen <- vector("list", dim) # penalty matrices

  Z.smf <- vector("list", dim) # matrices of sum-to-zero constraint
  Z.tint <- vector("list", dim) # matrices of sum-to-zero constraint

  if (option=="smf") sum.temp <- 1

  # if knots do not have names we give them the names of all the terms
  if (!is.null(knots) & is.null(names(knots))) names(knots) <- term
  if (is.null(names(df))) names(df) <- term
  
  knots2 <- vector("list", dim)
  names(knots2) <- term
  
  for (i in 1:dim){

    Base[[i]] <- crs(eval(parse(text=term[i]),envir=as.environment(data.spec)),knots=knots[[term[i]]],df=df[[term[i]]],intercept=TRUE)

    bs[[i]] <- Base[[i]]$bs

	knots2[[i]] <- Base[[i]]$knots

    # For each "smf" spline, we apply the sum-to-zero constraint
    if (option=="smf")  {

	  if (centering) { # standard case
	  
		contr.smf <- constraint(bs[[i]],Base[[i]]$pen)

		bs[[i]] <- contr.smf$X

		pen[[i]] <- matrix(0,nrow=sum.df,ncol=sum.df)
		
		pen[[i]][(sum.temp:(sum.temp+all.df[i]-1)),(sum.temp:(sum.temp+all.df[i]-1))] <- contr.smf$S

		sum.temp <- sum.temp+all.df[i]

		Z.smf[[i]] <- contr.smf$Z
		
	  }else{ # continuous by variable case
	  
		pen[[i]] <- matrix(0,nrow=sum.df,ncol=sum.df)
		
		pen[[i]][(sum.temp:(sum.temp+all.df[i]-1)),(sum.temp:(sum.temp+all.df[i]-1))] <- Base[[i]]$pen
	  
		sum.temp <- sum.temp+all.df[i]

		Z.smf[[i]] <- NULL
	  
	  }

    }else{

		if (option=="tint")  {

			contr.tint <- constraint(bs[[i]],Base[[i]]$pen)

			bs[[i]] <- contr.tint$X

			pen[[i]] <- contr.tint$S

			Z.tint[[i]] <- contr.tint$Z

		}else{

			pen[[i]] <- Base[[i]]$pen

			Z.smf <- NULL

		}

    }

  }

	if (option=="smf") {

		X <- bs[[1]]
		Z.tensor <- NULL
		Z.tint <- NULL
  
	}
  
	# For a tensor product spline, the sum-to-zero constraint is not applied on the marginal matrices but after
	# the creation of the multidimensional basis
	if (option=="tensor") {
	    
		if (centering) { # standard case
		
			contr.tensor <- constraint(tensor.prod.X(bs),tensor.prod.S(pen))
			X <- contr.tensor$X
			pen <- contr.tensor$S

			Z.tensor <- contr.tensor$Z
			Z.smf <- NULL
			Z.tint <- NULL
		
		}else{ # continuous by variable case
		
			X <- tensor.prod.X(bs)
			pen <- tensor.prod.S(pen)
			
			Z.tensor <- NULL
			Z.smf <- NULL
			Z.tint <- NULL
		
		}
    }

	# For a tensor product spline tint, the sum-to-zero constraint is applied on the marginal matrices before
	# the creation of the multidimensional basis
	if (option=="tint") {

		X <- tensor.prod.X(bs)
		pen <- tensor.prod.S(pen)
		Z.smf <- NULL
		Z.tensor <- NULL
	}

	n.col <- NCOL(X) # number of regression parameters
		
	if (!is.null(by.var)){ 
	
		if (is.factor(by.var)){

			lev <- levels(by.var)
			n.lev <- length(lev)
	
			dum = stats::model.matrix(~by.var-1) # create dummy variables for factor
			colnames(dum) <- gsub("by.var",by,colnames(dum))
			
			# In the case of an ordered factor, we generate no smooth for the first level of the factor
			if (is.ordered(by.var)) {
				
				ind.level <- 1 # indice of the column in dum corresponding to the first level of by.var
				position.factor <- (1:n.lev)[-ind.level]
				n.lev <- n.lev - 1
			
			}else{
			
				position.factor <- 1:n.lev
		
			}
			
			dim.pen2 <- n.col*n.lev # dimension of the new penalty matrix
			
			# we get a new design matrix for each level of the factor variable
			X <- do.call(cbind,lapply(position.factor,function(i) X*dum[,i]))
			
			colnames(X) <- sapply(position.factor,function(i) rep(paste(name,":",colnames(dum)[i],".",1:n.col,sep="")))
		
			# if we consider that the smoothing parameters are the same accross the
			# levels of a factor by-variable then we do not need several penalty matrices but 
			# a bigger penalty matrix
			dim2 <- dim*n.lev
			
			pen2 <- vector("list", dim2) # list of penalty matrices
			pen.same <- vector("list", dim)
			
			for (i in 1:dim){
					
				pen.same[[i]] <- matrix(0,nrow=dim.pen2,ncol=dim.pen2)
					
				for (j in (n.lev*(i-1)+1):(n.lev*i)){
				
					pen2[[j]] <- matrix(0,nrow=dim.pen2,ncol=dim.pen2)
					
					k <- j - n.lev*(i-1)
					position <- (1+n.col*(k-1)):(n.col*k)
					
					pen2[[j]][position,position] <- pen[[i]]
					
					pen.same[[i]] <- pen.same[[i]] + pen2[[j]]
				
				}
			}
			
			if (same.rho) { # are the smoothing parameters the same accross all levels of the factor 
				
				pen <- pen.same
				
				if (dim==1){
				
					lambda.name <- paste(name,":",by,sep="")
				
				}else{
				
					lambda.name <- rep(paste(name,":",by,".",1:dim,sep=""))
				
				}
				
			}else{
			
				pen <- pen2
				
				if (dim==1){
				
					lambda.name <- sapply(position.factor,function(i) paste(name,":",colnames(dum)[i],sep=""))
	
				}else{
					
					lambda.name <- c(sapply(1:dim,function(j) sapply(position.factor,function(i) paste(name,":",colnames(dum)[i],".",j,sep=""))))
					
				}
				
			}
			
		}else{
		
			X <- X*by.var
			
			colnames(X) <- paste(name,":",by,".",1:n.col,sep="")
		
			if (dim==1){
				
				lambda.name <- paste(name,":",by,sep="")
			
			}else{
			
				lambda.name <- paste(name,":",by,".",1:dim,sep="")
			
			}
		}

		
	}else{
	
		colnames(X) <- paste(name,".",1:n.col,sep="")
		
		if (dim==1){
			
			lambda.name <- name
			
		}else{
		
			lambda.name <- paste(name,".",1:dim,sep="")
		
		}
		
	}
	
	list(X=X,pen=pen,term=term,knots=knots2,dim=dim,all.df=all.df,sum.df=sum.df,Z.tensor=Z.tensor,Z.smf=Z.smf,Z.tint=Z.tint,lambda.name=lambda.name)

}

#----------------------------------------------------------------------------------------------------------------
# END of code : smooth.cons
#----------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------
# constraint : applies the sum-to-zero constraint
#----------------------------------------------------------------------------------------------------------------

#' Sum-to-zero constraint
#'
#' Applies the sum-to-zero constraints to design and penalty matrices.
#'
#' @param X A design matrix
#' @param S A penalty matrix or a list of penalty matrices
#' @param Z A list of sum-to-zero constraint matrices; default is NULL
#' @return List of objects with the following items:
#' \item{X}{Design matrix}
#' \item{S}{Penalty matrix or list of penalty matrices}
#' \item{Z}{List of sum-to-zero constraint matrices}
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' set.seed(15)
#'
#' X <- matrix(rnorm(10*3),nrow=10,ncol=3)
#' S <- matrix(rnorm(3*3),nrow=3,ncol=3) ; S <- 0.5*( S + t(S))
#'
#' # applying sum-to-zero constraint to a desgin matrix and a penalty matrix
#' constr <- constraint(X,S) 
#'
constraint <- function(X,S,Z=NULL){

  if (is.null(Z)){

	C <- colSums2(X)

	qrc <- qr(C)

	Z <- qr.Q(qrc,complete=TRUE)[,2:length(C)]

  }

  # Reparameterized design matrix
  XZ <- X%mult%Z

  # Reparameterized penalty matrix (or matrices)
  if(is.list(S)){

	length.S <- length(S)
	
	SZ <- lapply(1:length.S,function(i) Z%cross%S[[i]]%mult%Z)

  }else{

    SZ <- Z%cross%S%mult%Z

  }

  list(X=XZ,S=SZ,Z=Z)

}

#----------------------------------------------------------------------------------------------------------------
# END of code : constraint
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# smooth.cons.integral : almost identical to smooth.cons. This version is called inside the Gauss-Legendre
# quadrature. Here, the sum-to-zero constraints must be specified so that they correspond to the ones that
# were calculated with the initial dataset
#----------------------------------------------------------------------------------------------------------------

#' Design matrix of penalized splines in a smooth.spec object for Gauss-Legendre quadrature
#'
#' Almost identical to \code{\link{smooth.cons}}. This version is dedicated to Gauss-Legendre
#' quadrature. Here, the sum-to-zero constraints must be specified so that they correspond to the ones that
#' were calculated with the initial dataset.
#'
#' @param term Vector of strings that generally comes from the value "term" of a smooth.spec object
#' @param knots List of numeric vectors that specifies the knots of the splines (including boundaries).
#' @param df Degrees of freedom : numeric vector that indicates the number of knots desired for each covariate.
#' @param by numeric or factor variable in order to define a varying coefficient smooth; default is NULL.
#' @param option "smf", "tensor" or "tint".
#' @param data.spec data frame that represents the environment from which the covariate values and knots are to be calculated; default is NULL.
#' @param Z.smf List of matrices that represents the sum-to-zero constraint to apply for \code{\link{smf}} splines.
#' @param Z.tensor List of matrices that represents the sum-to-zero constraint to apply for \code{\link{tensor}} splines.
#' @param Z.tint List of matrices that represents the sum-to-zero constraint to apply for \code{\link{tint}} splines.
#' @param name simplified name of the smooth.spec call.
#' @return design matrix
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 4 knots (so we get a design matrix with 3 columns 
#' # because of centering constraint)
#'
#' data <- data.frame(time=seq(0,5,length=100))
#'
#' # retrieving sum-to-zero constraint matrices
#' Z.smf <- smooth.cons("time",knots=list(c(0,1,3,5)),df=4,option="smf",
#' data.spec=data,name="smf(time)")$Z.smf
#'
#' # constructing the design matrices for Gauss-Legendre quadrature
#' smooth.c.int <- smooth.cons.integral("time",knots=list(c(0,1,3,5)),df=4,option="smf",data.spec=data,
#' name="smf(time)",Z.smf=Z.smf,Z.tensor=NULL,Z.tint=NULL)
#'
smooth.cons.integral <- function(term, knots, df, by=NULL, option, data.spec, Z.smf, Z.tensor, Z.tint, name){

  
  if (option=="rd"){
  
	dim <- length(term)
	
	X <- stats::model.matrix(stats::as.formula(paste0("~",paste(term,collapse=":"),"-1")),data=data.spec)
	
	n.col <- dim(X)[2]
	
	colnames(X) <- rep(paste(name,".",1:n.col,sep=""))
	return(X)
	
  }
  
  centering <- TRUE

  dim <- length(term)

  # for by variables
  if(!is.character(by)) by <- deparse(by)
  by.var <- eval(parse(text=by),envir=as.environment(data.spec))
  
  if (!is.null(by.var)){ 
	# for a numeric by variable, no centering constraint is applied
	if(!is.factor(by.var)) centering <- FALSE
		
  }
  
  # if knots do not have names we give them the names of all the terms
  if (!is.null(knots) & is.null(names(knots))) names(knots) <- term
  if (is.null(names(df))) names(df) <- term
  
  Base <- vector("list",dim) # list containing all design and penalty matrices for each covariate
  bs <- vector("list", dim) # design matrices

  #if (option=="smf") sum.temp=1
	
  for (i in 1:dim){

    Base[[i]] <- crs(eval(parse(text=term[i]),envir=as.environment(data.spec)),knots=knots[[term[i]]],df=df[[term[i]]],intercept=TRUE)

    bs[[i]] <- Base[[i]]$bs

    # For each "smf" spline, we apply the sum-to-zero constraint
    if (option=="smf")  {

      if (centering) bs[[i]] <- bs[[i]]%mult%Z.smf[[i]]

    }

	if (option=="tint")  {

      bs[[i]] <- bs[[i]]%mult%Z.tint[[i]]

    }

  }

	if (option=="smf") {
	
		#X <- do.call(cbind,bs)
		X <- bs[[1]]
	
	}

	if (option=="tensor") {

		X <- tensor.prod.X(bs)
		if (centering) X <- X%mult%Z.tensor
		
    }

	if (option=="tint") {

		X <- tensor.prod.X(bs)
		
    }

	n.col <- NCOL(X) # number of regression parameters
	
	if (!is.null(by.var)){ 
	
		if (is.factor(by.var)){

			lev <- levels(by.var)
			n.lev <- length(lev)
	
			dum = stats::model.matrix(~by.var-1) # create dummy variables for factor
			colnames(dum) <- gsub("by.var",by,colnames(dum))
			
			# In the case of an ordered factor, we generate no smooth for the first level of the factor
			if (is.ordered(by.var)) {
				
				ind.level <- 1 # indice of the column in dum corresponding to the first level of by.var
				position.factor <- (1:n.lev)[-ind.level]
				n.lev <- n.lev - 1
			
			}else{
			
				position.factor <- 1:n.lev
		
			}
			
			# we get a new design matrix for each level of the factor variable
			X <- do.call(cbind,lapply(position.factor,function(i) X*dum[,i]))
			
			colnames(X) <- sapply(position.factor,function(i) rep(paste(name,":",colnames(dum)[i],".",1:n.col,sep="")))
			
		}else{
		
			X <- X*by.var
			
			colnames(X) <- rep(paste(name,":",by,".",1:n.col,sep=""))
		
			lambda.name <- rep(paste(name,":",by,".",1:dim,sep=""))
		
		}

	}else{
	
		colnames(X) <- rep(paste(name,".",1:n.col,sep=""))
		
		lambda.name <- rep(paste(name,".",1:dim,sep=""))
	
	}
	
	return(X)

}

#----------------------------------------------------------------------------------------------------------------
# END of code : smooth.cons.integral
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# instr : returns the position of the nth occurrence of a string in another one
#----------------------------------------------------------------------------------------------------------------

#' Position of the nth occurrence of a string in another one
#'
#' Returns the position of the nth occurrence of str2 in str1. Returns 0 if str2 is not found
#'
#' @param str1 main string in which str2 is to be found
#' @param str2 substring contained in str1
#' @param startpos starting position in str1; default is 1
#' @param n which occurrence is to be found; default is 1
#' @return number representing the nth position of str2 in str1 
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' instr("character test to find the position of the third letter r","r",n=3)
#'
instr <- function(str1,str2,startpos=1,n=1){
			aa=unlist(strsplit(substring(str1,startpos),str2))
			if(length(aa) < n+1 ) return(0);
			return(sum(nchar(aa[1:n])) + startpos+(n-1)*nchar(str2) )
}

#----------------------------------------------------------------------------------------------------------------
# END of code : instr
#----------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------
# model.cons : based on the model formula, builds the global design matrix and penalty matrices
#----------------------------------------------------------------------------------------------------------------

#' Design and penalty matrices for the model
#'
#' Sets up the model before optimization. Builds the design matrix, the penalty matrix and all the design matrices needed for Gauss-Legendre quadrature.
#'
#' @param formula formula object identifying the model
#' @param lambda vector of smoothing parameters
#' @param data.spec data frame that represents the environment from which the covariate values and knots are to be calculated
#' @param t1 vector of follow-up times
#' @param t1.name name of \code{t1} in \code{data.spec}
#' @param t0 vector of origin times (usually filled with zeros)
#' @param t0.name name of \code{t0} in \code{data.spec}
#' @param event vector of censoring indicators
#' @param event.name name of event in \code{data.spec}
#' @param expected vector of expected hazard
#' @param expected.name name of expected in \code{data.spec}
#' @param type "net" or "overall"
#' @param n.legendre number of nodes for Gauss-Legendre quadrature
#' @param cl original \code{survPen} call
#' @param beta.ini initial set of regression parameters
#' @return List of objects with the following items:
#' \item{cl}{original \code{survPen} call}
#' \item{type}{"net" or "overall"}
#' \item{n.legendre}{number of nodes for Gauss-Legendre quadrature}
#' \item{n}{number of individuals}
#' \item{p}{number of parameters}
#' \item{X.para}{design matrix associated with fully parametric parameters (unpenalized)}
#' \item{X.smooth}{design matrix associated with the penalized parameters}
#' \item{X}{design matrix for the model}
#' \item{leg}{list of nodes and weights for Gauss-Legendre integration on [-1;1] as returned by \code{\link[statmod]{gauss.quad}}}
#' \item{X.GL}{list of matrices (\code{length(X.GL)=n.legendre}) for Gauss-Legendre quadrature}
#' \item{S}{penalty matrix for the model. Sum of the elements of \code{S.list}}
#' \item{S.scale}{vector of rescaling factors for the penalty matrices}
#' \item{rank.S}{rank of the penalty matrix}
#' \item{S.F}{balanced penalty matrix as described in section 3.1.2 of (Wood,2016). Sum of the elements of \code{S.F.list}}
#' \item{U.F}{Eigen vectors of S.F, useful for the initial reparameterization to separate penalized ad unpenalized subvectors. Allows stable evaluation of the log determinant of S and its derivatives}
#' \item{S.smf}{List of penalty matrices associated with all "smf" calls}
#' \item{S.tensor}{List of penalty matrices associated with all "tensor" calls}
#' \item{S.tint}{List of penalty matrices associated with all "tint" calls}
#' \item{S.rd}{List of penalty matrices associated with all "rd" calls}
#' \item{smooth.name.smf}{List of names for the "smf" calls associated with S.smf}
#' \item{smooth.name.tensor}{List of names for the "tensor" calls associated with S.tensor}
#' \item{smooth.name.tint}{List of names for the "tint" calls associated with S.tint}
#' \item{smooth.name.rd}{List of names for the "rd" calls associated with S.rd}
#' \item{S.pen}{List of all the rescaled penalty matrices redimensioned to df.tot size. Every element of \code{pen} noted \code{pen[[i]]} is made from a penalty matrix returned by
#' \code{\link{smooth.cons}} and is multiplied by the factor 
#' S.scale=norm(X,type="I")^2/norm(pen[[i]],type="I")}
#' \item{S.list}{Equivalent to S.pen but with every element multiplied by its associated smoothing parameter}
#' \item{S.F.list}{Equivalent to S.pen but with every element divided by its Frobenius norm}
#' \item{lambda}{vector of smoothing parameters}
#' \item{df.para}{degrees of freedom associated with fully parametric terms (unpenalized)}
#' \item{df.smooth}{degrees of freedom associated with penalized terms}
#' \item{df.tot}{\code{df.para + df.smooth}}
#' \item{list.smf}{List of all \code{smf.smooth.spec} objects contained in the model}
#' \item{list.tensor}{List of all \code{tensor.smooth.spec} objects contained in the model}
#' \item{list.tint}{List of all \code{tint.smooth.spec} objects contained in the model}
#' \item{nb.smooth}{number of smoothing parameters}
#' \item{Z.smf}{List of matrices that represents the sum-to-zero constraints to apply for \code{\link{smf}} splines}
#' \item{Z.tensor}{List of matrices that represents the sum-to-zero constraints to apply for \code{\link{tensor}} splines}
#' \item{Z.tint}{List of matrices that represents the sum-to-zero constraints to apply for \code{\link{tint}} splines}
#' \item{beta.ini}{initial set of regression parameters}
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 4 knots
#'
#' data <- data.frame(time=seq(0,5,length=100),event=1,t0=0)
#'
#' form <- ~ smf(time,knots=c(0,1,3,5))
#'
#' t1 <- eval(substitute(time), data)
#' t0 <- eval(substitute(t0), data)
#' event <- eval(substitute(event), data)
#'	
#' # The following code sets up everything we need in order to fit the model
#' model.c <- model.cons(form,lambda=0,data.spec=data,t1=t1,t1.name="time",
#' t0=rep(0,100),t0.name="t0",event=event,event.name="event",
#' expected=NULL,expected.name=NULL,type="overall",n.legendre=20,
#' cl="survPen(form,data,t1=time,event=event)",beta.ini=NULL)
#'
model.cons <- function(formula,lambda,data.spec,t1,t1.name,t0,t0.name,event,event.name,expected,expected.name,type,n.legendre,cl,beta.ini){

  #--------------------------------------------
  # extracting information from formula

  formula <- stats::as.formula(formula)
  
  Terms <- stats::terms(formula)
  tmp <- attr(Terms, "term.labels")

  if(attr(Terms, "intercept")==0){

	intercept <- "-1"

  }else{

	intercept <- ""

  }

  # indices of smooth terms
  ind.smf <- grep("^smf\\(", tmp)

  ind.tensor <- grep("^tensor\\(", tmp)
  
  ind.tint <- grep("^tint\\(", tmp)
  
  ind.rd <- grep("^rd\\(", tmp)

  # names of smooth terms
  Ad <- tmp[ind.smf]
  Tens <- tmp[ind.tensor]
  Tint <- tmp[ind.tint]
  Rd <- tmp[ind.rd]

  smooth.smf <- FALSE
  smooth.tensor <- FALSE
  smooth.tint <- FALSE
  smooth.rd <- FALSE

  # Are there smooth terms ?
  length.Ad <- length(Ad)
  length.Tens <- length(Tens)
  length.Tint <- length(Tint)
  length.Rd <- length(Rd)

  if (length.Ad!=0) smooth.smf <- TRUE
  if (length.Tens!=0) smooth.tensor <- TRUE
  if (length.Tint!=0) smooth.tint <- TRUE
  if (length.Rd!=0) smooth.rd <- TRUE
  
  # full parametric terms
  if (smooth.smf | smooth.tensor | smooth.tint | smooth.rd){
    Para <- tmp[-c(ind.smf,ind.tensor,ind.tint,ind.rd)]
  }else{
    Para <- tmp
  }

  # parametric formula
  if (length(Para)==0){
    formula.para <- stats::as.formula("~1")
  }else{
    formula.para <- stats::as.formula(paste("~",paste(Para,collapse="+"),intercept,sep=""))
  }

  # parametric design matrix
  X.para <- stats::model.matrix(formula.para,data=data.spec)
	
  df.para <- NCOL(X.para)

  # Initialization of smooth matrices
  X.smf <- NULL
  X.tensor <- NULL
  X.tint <- NULL
  X.rd <- NULL

  Z.smf <- NULL
  Z.tensor <- NULL
  Z.tint <- NULL

  S.smf <- NULL
  S.tensor <- NULL
  S.tint <- NULL
  S.rd <- NULL

  # If there are additive smooth terms
  if (smooth.smf){

    X.smf <- vector("list",length.Ad)
    S.smf <- vector("list",length.Ad)

	Z.smf <- vector("list",length.Ad)
	list.smf <- vector("list",length.Ad)
    
	# We get the design and penalty matrices from the smf call(s)
    # (there may be several calls)
	dim.smf <- vector(length=length.Ad)
	df.smf <- vector(length=length.Ad)
	
	smooth.name.smf <- vector(length=length.Ad)
	lambda.name.smf <- vector("list",length.Ad)

    for (i in 1:length.Ad){

	    list.smf[[i]] <- eval(parse(text=Ad[i]))

	    if(list.smf[[i]]$dim>1) stop("smf calls must contain only one covariate")

		temp <- smooth.cons(list.smf[[i]]$term,list.smf[[i]]$knots,list.smf[[i]]$df,list.smf[[i]]$by,option="smf",data.spec,list.smf[[i]]$same.rho,list.smf[[i]]$name)
		
		X.smf[[i]] <- temp$X # design matrix
		S.smf[[i]] <- temp$pen # List of penalty matrices

		Z.smf[[i]] <- temp$Z.smf # list of sum-to-zero constraint matrices
	    list.smf[[i]]$knots <- temp$knots # List of knots

	    dim.smf[i] <- length(temp$pen) # number of smoothing parameters in the ith smf call
	    df.smf[i] <- NCOL(temp$X) # number of regression parameters

		smooth.name.smf[i] <- list.smf[[i]]$name 
		lambda.name.smf[[i]] <- temp$lambda.name
		
    }

    # We join all design matrices for additive smooths
    X.smf <- do.call(cbind, X.smf)

  }

  # If there are tensor smooth terms
  if (smooth.tensor){

    X.tensor <- vector("list",length.Tens)
    S.tensor <- vector("list",length.Tens)

	Z.tensor <- vector("list",length.Tens)
	list.tensor <- vector("list",length.Tens)

    dim.tensor <- vector(length=length.Tens)
    df.tensor <- vector(length=length.Tens)
	
	smooth.name.tensor <- vector(length=length.Tens)
	lambda.name.tensor <- vector("list",length.Tens)

    # We get the design and penalization matrices from the tensor calls
    # (there may be several)
    for (i in 1:length.Tens){

		list.tensor[[i]] <- eval(parse(text=Tens[i]))
		
		temp <- smooth.cons(list.tensor[[i]]$term,list.tensor[[i]]$knots,list.tensor[[i]]$df,list.tensor[[i]]$by,option="tensor",data.spec,list.tensor[[i]]$same.rho,list.tensor[[i]]$name)
       
		dim.tensor[i] <- length(temp$pen) # number of smoothing parameters in the ith tensor call
		df.tensor[i] <- NCOL(temp$X) # number of regression parameters

		X.tensor[[i]] <- temp$X # penalty matrix
		S.tensor[[i]] <- temp$pen # List of penalty matrices

		Z.tensor[[i]] <- temp$Z.tensor # list of sum-to-zero constraint matrices
		list.tensor[[i]]$knots <- temp$knots # List of knots

		smooth.name.tensor[i] <- list.tensor[[i]]$name 
		lambda.name.tensor[[i]] <- temp$lambda.name
	  
    }

    # We join all design matrices for tensor smooths
    X.tensor <- do.call(cbind, X.tensor)
	
  }

  # If there are tint smooth terms
  if (smooth.tint){

    X.tint <- vector("list",length.Tint)
    S.tint <- vector("list",length.Tint)

	Z.tint <- vector("list",length.Tint)
	list.tint <- vector("list",length.Tint)

    dim.tint <- vector(length=length.Tint)
    df.tint <- vector(length=length.Tint)
	
	smooth.name.tint <- vector(length=length.Tint)
	lambda.name.tint <- vector("list",length.Tint)

    # We get the design and penalization matrices from the tint calls
    # (there may be several)

    for (i in 1:length.Tint){

  		list.tint[[i]] <- eval(parse(text=Tint[i]))

  		temp <- smooth.cons(list.tint[[i]]$term,list.tint[[i]]$knots,list.tint[[i]]$df,list.tint[[i]]$by,option="tint",data.spec,list.tint[[i]]$same.rho,list.tint[[i]]$name)

  		dim.tint[i] <- length(temp$pen) # number of smoothing parameters in the ith tint call
  		df.tint[i] <- NCOL(temp$X) # number of regression parameters 

  		X.tint[[i]] <- temp$X # design matrix
  		S.tint[[i]] <- temp$pen # list of penalty matrices

  		Z.tint[[i]] <- temp$Z.tint # list of sum-to-zero constraint matrices
  		list.tint[[i]]$knots <- temp$knots # list of knots

		smooth.name.tint[i] <- list.tint[[i]]$name 
		lambda.name.tint[[i]] <- temp$lambda.name
	  
    }

    # We join all design matrices for tint smooths
    X.tint <- do.call(cbind, X.tint)

  }

  
  # If there are additive smooth terms
  if (smooth.rd){

    X.rd <- vector("list",length.Rd)
    S.rd <- vector("list",length.Rd)

	Z.rd <- vector("list",length.Rd)
	list.rd <- vector("list",length.Rd)
    
	# We get the design and penalty matrices from the smf call(s)
    # (there may be several calls)
	dim.rd <- vector(length=length.Rd)
	df.rd <- vector(length=length.Rd)
	
	smooth.name.rd <- vector(length=length.Rd)
	lambda.name.rd <- vector("list",length.Rd)

    for (i in 1:length.Rd){

	    list.rd[[i]] <- eval(parse(text=Rd[i]))

		temp <- smooth.cons(list.rd[[i]]$term,list.rd[[i]]$knots,list.rd[[i]]$df,list.rd[[i]]$by,option="rd",data.spec,list.rd[[i]]$same.rho,list.rd[[i]]$name)
		
		X.rd[[i]] <- temp$X # design matrix
		S.rd[[i]] <- temp$pen # List of penalty matrices

	    dim.rd[i] <- length(temp$pen) # number of smoothing parameters in the ith smf call
	    df.rd[i] <- NCOL(temp$X) # number of regression parameters

		smooth.name.rd[i] <- list.rd[[i]]$name 
		lambda.name.rd[[i]] <- temp$lambda.name
		
    }

    # We join all design matrices for additive smooths
    X.rd <- do.call(cbind, X.rd)

  }

  X.smooth <- cbind(X.smf,X.tensor,X.tint,X.rd)
  
  X <- cbind(X.para,X.smooth)
	
  # total number of regression parameters including those associated with fully parametric terms
  df.tot <- NCOL(X)

  if (!smooth.smf){
    dim.smf <- 0
    df.smf <- 0
	list.smf <- NULL
	lambda.name.smf <- NULL
	smooth.name.smf <- NULL
  }

  if (!smooth.tensor){
	dim.tensor <- 0
    df.tensor <- 0
	list.tensor <- NULL
	lambda.name.tensor <- NULL
	smooth.name.tensor <- NULL
  }

  if (!smooth.tint){
	dim.tint <- 0
    df.tint <- 0
	list.tint <- NULL
	lambda.name.tint <- NULL
	smooth.name.tint <- NULL
  }

  if (!smooth.rd){
	dim.rd <- 0
    df.rd <- 0
	list.rd <- NULL
	lambda.name.rd <- NULL
	smooth.name.rd <- NULL
  }
  
  # number of regression parameters for each type of smooth
  sum.df.smf <- sum(df.smf)
  sum.df.tensor <- sum(df.tensor)
  sum.df.tint <- sum(df.tint)
  sum.df.rd <- sum(df.rd)
  
  # number of terms i.e number of smoothing parameters for each type of smooth
  sum.dim.smf <- sum(dim.smf)
  sum.dim.tensor <- sum(dim.tensor)
  sum.dim.tint <- sum(dim.tint)
  sum.dim.rd <- sum(dim.rd)
  
  df.smooth <- sum.df.smf+sum.df.tensor+sum.df.tint+sum.df.rd # number of regression parameters associated with smooth terms
  nb.smooth <- sum.dim.smf+sum.dim.tensor+sum.dim.tint+sum.dim.rd # number of smoothing parameters
  name.smooth <- c(unlist(lambda.name.smf),unlist(lambda.name.tensor),unlist(lambda.name.tint),unlist(lambda.name.rd))

  # Define the final penalization matrix
  S <- matrix(0,nrow=df.tot,ncol=df.tot)

  # Define the balanced penalty (see Wood 2016)
  S.F <- matrix(0,nrow=df.tot,ncol=df.tot)

  # List of all the penalization matrices with nrow=df.tot and ncol=df.tot
  S.pen <- lapply(1:nb.smooth, function(i) matrix(0,nrow=df.tot,ncol=df.tot))	
  
  S.list <- lapply(1:nb.smooth, function(i) matrix(0,nrow=df.tot,ncol=df.tot))

  S.F.list <- lapply(1:nb.smooth, function(i) matrix(0,nrow=df.tot,ncol=df.tot))

  # Finally We join all penalization matrices into S if there are any smooths
  if (smooth.smf | smooth.tensor | smooth.tint | smooth.rd){

	  pen <- lapply(1:nb.smooth, function(i) matrix(0,nrow=df.smooth,ncol=df.smooth))

    # additive matrices
    if (smooth.smf){
      for (i in 1:length.Ad){

        df.plus <- if(i==1){0}else{cumsum(df.smf)[i-1]}
        dim.plus <- if(i==1){0}else{cumsum(dim.smf)[i-1]}
		
        position.smf <- (1+df.plus):(df.smf[i]+df.plus)
		
        for (j in 1:dim.smf[i]){
			
          pen[[j+dim.plus]][position.smf,position.smf] <- S.smf[[i]][[j]]

        }

      }
    }

    # tensor matrices
    if (smooth.tensor){
      for (i in 1:length.Tens){

		df.plus <- if(i==1){0}else{cumsum(df.tensor)[i-1]}
		dim.plus <- if(i==1){0}else{cumsum(dim.tensor)[i-1]}

		position.temp <- sum.df.smf+df.plus
		position.tensor <- (position.temp+1):(position.temp+df.tensor[i])

        for (j in 1:dim.tensor[i]){

          pen[[j+sum.dim.smf+dim.plus]][position.tensor,position.tensor] <- S.tensor[[i]][[j]]

        }

      }
    }

	# tint matrices
    if (smooth.tint){
      for (i in 1:length.Tint){

		df.plus <- if(i==1){0}else{cumsum(df.tint)[i-1]}
		dim.plus <- if(i==1){0}else{cumsum(dim.tint)[i-1]}

		position.temp <- sum.df.smf+sum.df.tensor+df.plus
		position.tint <- (position.temp+1):(position.temp+df.tint[i])

        for (j in 1:dim.tint[i]){

          pen[[j+sum.dim.smf+sum.dim.tensor+dim.plus]][position.tint,position.tint] <- S.tint[[i]][[j]]

        }

      }
    }
	
	# rd matrices
    if (smooth.rd){
      for (i in 1:length.Rd){

		df.plus <- if(i==1){0}else{cumsum(df.rd)[i-1]}
		dim.plus <- if(i==1){0}else{cumsum(dim.rd)[i-1]}

		position.temp <- sum.df.smf+sum.df.tensor+sum.df.tint+df.plus
		position.rd <- (position.temp+1):(position.temp+df.rd[i])

        for (j in 1:dim.rd[i]){

          pen[[j+sum.dim.smf+sum.dim.tensor+sum.dim.tint+dim.plus]][position.rd,position.rd] <- S.rd[[i]][[j]]

        }

      }
    }
	
    if (is.null(lambda)) {lambda <- rep(0,nb.smooth)}
	
	if (length(lambda)!=nb.smooth) {
	
		if (length(lambda)==1) {
		
			warning("lambda is of length 1. All smoothing parameters (",nb.smooth,") are set to ",lambda)
			lambda <- rep(lambda,nb.smooth)
		
		}else{
	
			stop("lambda should be of length ",nb.smooth)
		
		}
	}
	
	names(lambda) <- name.smooth
	
    # All penalty matrices into a unique S

	norm.X <- norm(X,type="I")^2

	S.scale <- rep(0,nb.smooth)
	
	for (i in 1:nb.smooth){

		# Rescaling the penalty matrices

		S.scale[i] <- norm.X/norm(pen[[i]],type="I")

		pen[[i]] <- S.scale[i]*pen[[i]]

		S.pen[[i]][(df.para+1):df.tot,(df.para+1):df.tot] <- pen[[i]]
		
	    S.list[[i]] <- lambda[i]*S.pen[[i]]

	    S.F.list[[i]] <- S.pen[[i]]/norm(S.pen[[i]],type="F")

	    S <- S+S.list[[i]]

	    S.F <- S.F+S.F.list[[i]]

    }
	
	#------------------------------------------------------------------------------- 
	# Initial reparameterization to separate penalized and unpenalized subvectors
	
	eigen.F <- eigen(S.F,symmetric=TRUE) 
	U.F <- eigen.F$vectors # eigen vectors of S.F
	vp.F <- eigen.F$values # eigen values of S.F
		
	tol.S.F <- .Machine$double.eps^0.8 * max(vp.F) # see Appendix B of Wood, S. N. (2011) 
	# Fast Stable Restricted Maximum Likelihood and Marginal Likelihood Estimation of 
	# Semiparametric Generalized Linear Models, Journal of the Royal Statistical Society, Series B, 73, 3 36.
		
	pos.S.F.eigen <- vp.F[vp.F >= tol.S.F] # positive eigen values of S.F
	rank.S <- length(pos.S.F.eigen) 

	
  } else {
    
	nb.smooth <- 0
	S.scale <- 0
    S.pen <- NULL
	rank.S <- NULL
	U.F <- NULL
	
  }

  #-------------------------------------------------------------------
  # Design matrices for Gauss-Legendre quadrature

  leg <- statmod::gauss.quad(n=n.legendre,kind="legendre")

  X.func <- function(t1,t1.name,data,formula,Z.smf,Z.tensor,Z.tint,list.smf,list.tensor,list.tint,list.rd){

    data.t <- data
    data.t[,t1.name] <- t1
    design.matrix(formula,data.spec=data.t,Z.smf=Z.smf,Z.tensor=Z.tensor,Z.tint=Z.tint,list.smf=list.smf,list.tensor=list.tensor,list.tint=list.tint,list.rd=list.rd)

  }
	
  tm <- 0.5*(t1-t0)

  # list of n.legendre design matrices for numerical integration
  X.GL <- lapply(1:n.legendre, function(i) X.func(tm*leg$nodes[i]+(t0+t1)/2,t1.name,data.spec,formula,Z.smf,Z.tensor,Z.tint,list.smf,list.tensor,list.tint,list.rd))
  

  return(list(cl=cl,type=type,n.legendre=n.legendre,t0=t0,t0.name=t0.name,t1=t1,t1.name=t1.name,tm=tm,event=event,event.name=event.name,expected=expected,expected.name=expected.name,
  n=dim(X)[1],p=dim(X)[2],X.para=X.para,X.smooth=X.smooth,X=X,leg=leg,X.GL=X.GL,S=S,S.scale=S.scale,rank.S=rank.S,S.F=S.F,U.F=U.F,
  S.smf=S.smf,S.tensor=S.tensor,S.tint=S.tint,S.rd=S.rd,smooth.name.smf=smooth.name.smf,smooth.name.tensor=smooth.name.tensor,smooth.name.tint=smooth.name.tint,smooth.name.rd=smooth.name.rd,
  S.pen=S.pen,S.list=S.list,S.F.list=S.F.list,lambda=lambda,df.para=df.para,df.smooth=df.smooth,df.tot=df.tot,
  list.smf=list.smf,list.tensor=list.tensor,list.tint=list.tint,list.rd=list.rd,nb.smooth=nb.smooth,Z.smf=Z.smf,Z.tensor=Z.tensor,Z.tint=Z.tint,beta.ini=beta.ini))

}

#----------------------------------------------------------------------------------------------------------------
# END of code : model.cons
#----------------------------------------------------------------------------------------------------------------
  
  

#----------------------------------------------------------------------------------------------------------------
# design.matrix : builds the design matrices for numerical integration. The sum-to-zero constraints applied must
# be the ones that were derived in model.cons
#----------------------------------------------------------------------------------------------------------------

#' Design matrix for the model needed in Gauss-Legendre quadrature
#'
#' Builds the design matrix for the whole model when the sum-to-zero constraints are specified. The function is called inside \code{\link{model.cons}}
#' for Gauss-Legendre quadrature.
#'
#' @param formula formula object identifying the model
#' @param data.spec data frame that represents the environment from which the covariate values and knots are to be calculated
#' @param Z.smf List of matrices that represents the sum-to-zero constraint to apply for \code{\link{smf}} splines
#' @param Z.tensor List of matrices that represents the sum-to-zero constraint to apply for \code{\link{tensor}} splines
#' @param Z.tint List of matrices that represents the sum-to-zero constraint to apply for \code{\link{tint}} splines
#' @param list.smf List of all smf.smooth.spec objects contained in the model
#' @param list.tensor List of all tensor.smooth.spec objects contained in the model
#' @param list.tint List of all tint.smooth.spec objects contained in the model
#' @param list.rd List of all rd.smooth.spec objects contained in the model
#' @return design matrix for the model
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 4 knots
#'
#' data <- data.frame(time=seq(0,5,length=100),event=1,t0=0)
#' 
#' form <- ~ smf(time,knots=c(0,1,3,5))
#' 
#' t1 <- eval(substitute(time), data)
#' t0 <- eval(substitute(t0), data)
#' event <- eval(substitute(event), data)
#' 	
#' # Setting up the model
#' model.c <- model.cons(form,lambda=0,data.spec=data,t1=t1,t1.name="time",
#' t0=rep(0,100),t0.name="t0",event=event,event.name="event",
#' expected=NULL,expected.name=NULL,type="overall",n.legendre=20,
#' cl="survPen(form,data,t1=time,event=event)",beta.ini=NULL)
#'  
#' # Retrieving the sum-to-zero constraint matrices and the list of knots
#' Z.smf <- model.c$Z.smf ; list.smf <- model.c$list.smf
#' 
#' # Calculating the design matrix
#' design.M <- design.matrix(form,data.spec=data,Z.smf=Z.smf,list.smf=list.smf,
#' Z.tensor=NULL,Z.tint=NULL,list.tensor=NULL,list.tint=NULL,list.rd=NULL)
#'
design.matrix <- function(formula,data.spec,Z.smf,Z.tensor,Z.tint,list.smf,list.tensor,list.tint,list.rd){

  formula <- stats::as.formula(formula)

  Terms <- stats::terms(formula)
  tmp <- attr(Terms, "term.labels")

  if(attr(Terms, "intercept")==0){

	intercept <- "-1"

  }else{

	intercept <- ""

  }

  # indices of smooth terms
  ind.smf <- grep("^smf\\(", tmp)

  ind.tensor <- grep("^tensor\\(", tmp)
  
  ind.tint <- grep("^tint\\(", tmp)
  
  ind.rd <- grep("^rd\\(", tmp)

  # names of smooth terms
  Ad <- tmp[ind.smf]
  Tens <- tmp[ind.tensor]
  Tint <- tmp[ind.tint]
  Rd <- tmp[ind.rd]

  smooth.smf <- FALSE
  smooth.tensor <- FALSE
  smooth.tint <- FALSE
  smooth.rd <- FALSE

  # Are there smooth terms ?
  length.Ad <- length(Ad)
  length.Tens <- length(Tens)
  length.Tint <- length(Tint)
  length.Rd <- length(Rd)

  if (length.Ad!=0) smooth.smf <- TRUE
  if (length.Tens!=0) smooth.tensor <- TRUE
  if (length.Tint!=0) smooth.tint <- TRUE
  if (length.Rd!=0) smooth.rd <- TRUE
  
  # fully parametric terms
  if (smooth.smf | smooth.tensor | smooth.tint | smooth.rd){
    Para <- tmp[-c(ind.smf,ind.tensor,ind.tint,ind.rd)]
  }else{
    Para <- tmp
  }

  # parametric formula
  if (length(Para)==0){
    formula.para <- stats::as.formula("~1")
  }else{
    formula.para <- stats::as.formula(paste("~",paste(Para,collapse="+"),intercept,sep=""))
  }

  # parametric design matrix
  X.para <- stats::model.matrix(formula.para,data=data.spec)
	
  df.para <- NCOL(X.para)

  # Initialization of smooth matrices
  X.smf <- NULL
  X.tensor <- NULL
  X.tint <- NULL
  X.rd <- NULL

  ##################### If there are additive smooth terms
  if (smooth.smf){

    X.smf <- vector("list",length.Ad)

    # We get the design matrix from the smf call
    # (there should be only one call)

    for (i in 1:length.Ad){ # here we must have length.Ad=1

      X.smf[[i]] <- smooth.cons.integral(list.smf[[i]]$term,list.smf[[i]]$knots,list.smf[[i]]$df,list.smf[[i]]$by,option="smf",data.spec,Z.smf=Z.smf[[i]],Z.tensor=NULL,Z.tint=NULL,list.smf[[i]]$name)

    }

    # We join all design matrices for additive smooths
    X.smf <- do.call(cbind, X.smf)

  }

  ##################### If there are tensor smooth terms
  if (smooth.tensor){

    X.tensor <- vector("list",length.Tens)

    # We get the design from the tensor calls
    # (there may be several)

    for (i in 1:length.Tens){
	
      X.tensor[[i]] <- smooth.cons.integral(list.tensor[[i]]$term,list.tensor[[i]]$knots,list.tensor[[i]]$df,list.tensor[[i]]$by,option="tensor",data.spec,Z.smf=NULL,Z.tensor=Z.tensor[[i]],Z.tint=NULL,list.tensor[[i]]$name)
		
    }
	
    # We join all design matrices for tensor smooths
    X.tensor <- do.call(cbind, X.tensor)

  }

  ##################### If there are tint smooth terms
  if (smooth.tint){

    X.tint <- vector("list",length.Tint)

    # We get the design from the tint calls
    # (there may be several)

    for (i in 1:length.Tint){

      X.tint[[i]] <- smooth.cons.integral(list.tint[[i]]$term,list.tint[[i]]$knots,list.tint[[i]]$df,list.tint[[i]]$by,option="tint",data.spec,Z.smf=NULL,Z.tensor=NULL,Z.tint=Z.tint[[i]],list.tint[[i]]$name)

    }

    # We join all design matrices for tint smooths
    X.tint <- do.call(cbind, X.tint)

  }
  
  ##################### If there are random effects
  if (smooth.rd){

    X.rd <- vector("list",length.Rd)

    # We get the design from the rd calls
    # (there should be only one)

    for (i in 1:length.Rd){

      X.rd[[i]] <- smooth.cons.integral(list.rd[[i]]$term,list.rd[[i]]$knots,list.rd[[i]]$df,list.rd[[i]]$by,option="rd",data.spec,Z.smf=NULL,Z.tensor=NULL,Z.tint=NULL,list.rd[[i]]$name)

    }

    # We join all design matrices for rd smooths
    X.rd <- do.call(cbind, X.rd)

  }
  
  X.smooth <- cbind(X.smf,X.tensor,X.tint,X.rd)

  X <- cbind(X.para,X.smooth)

  return(X)

}
#----------------------------------------------------------------------------------------------------------------
# END of code : design.matrix
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# repam : Initial reparameterization
#----------------------------------------------------------------------------------------------------------------

#' Applies initial reparameterization for stable evaluation of the log determinant of the penalty matrix
#'
#' Transforms the object from \code{\link{model.cons}} by applying the matrix reparameterization (matrix U.F). The reparameterization
#' is reversed at convergence by \code{\link{inv.repam}}.
#'
#' @param build object as returned by \code{\link{model.cons}}
#' @return 
#' \item{build}{an object as returned by \code{\link{model.cons}}}
#' \item{X.ini}{initial design matrix (before reparameterization)}
#' \item{S.pen.ini}{initial penalty matrices}
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 4 knots
#'
#' data <- data.frame(time=seq(0,5,length=100),event=1,t0=0)
#' 
#' form <- ~ smf(time,knots=c(0,1,3,5))
#' 
#' t1 <- eval(substitute(time), data)
#' t0 <- eval(substitute(t0), data)
#' event <- eval(substitute(event), data)
#' 	
#' # Setting up the model before fitting
#' model.c <- model.cons(form,lambda=0,data.spec=data,t1=t1,t1.name="time",
#' t0=rep(0,100),t0.name="t0",event=event,event.name="event",
#' expected=NULL,expected.name=NULL,type="overall",n.legendre=20,
#' cl="survPen(form,data,t1=time,event=event)",beta.ini=NULL)
#'  
#' # Reparameterization allows separating the parameters into unpenalized and 
#' # penalized ones for maximum numerical stability
#' re.model.c <- repam(model.c)
#'
repam <- function(build){
	
	coef.name <- colnames(build$X)
	
	# We store X and S to given them back at convergence
	X.ini <- build$X
	S.pen.ini <- build$S.pen
	#--------
	
	build$X <- build$X%mult%build$U.F
	
	colnames(build$X) <- coef.name
	
	
	# We need to reparameterize the initial regression parameters too
	if (!is.null(build$beta.ini)) build$beta.ini <- t(build$U.F)%vec%build$beta.ini
	
	
	# penalty matrices
	build$S.pen <- lapply(1:build$nb.smooth,function(i) build$U.F%cross%build$S.pen[[i]]%mult%build$U.F)
	build$S.pen <- lapply(1:build$nb.smooth,function(i) 0.5*(t(build$S.pen[[i]])+build$S.pen[[i]]))

	build$S.list <- lapply(1:build$nb.smooth,function(i) build$lambda[i]*build$S.pen[[i]])
	build$S <- Reduce("+",build$S.list)
	
	# List of the design matrices for Gauss-Legendre quadrature
	build$X.GL <- lapply(1:build$n.legendre, function(i) build$X.GL[[i]]%mult%build$U.F)
		  
	
	return(list(build=build,X.ini=X.ini,S.pen.ini=S.pen.ini))
						
}
#----------------------------------------------------------------------------------------------------------------
# END of code : repam
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# inv.repam : return to initial parameterization
#----------------------------------------------------------------------------------------------------------------

#' Reverses the initial reparameterization for stable evaluation of the log determinant of the penalty matrix
#'
#' Transforms the final model by reversing the initial reparameterization performed by \code{\link{repam}}. Derives the corrected version of the Bayesian covariance matrix 
#'
#' @param model survPen object, see \code{\link{survPen.fit}} for details
#' @param X.ini initial design matrix (before reparameterization)
#' @param S.pen.ini initial penalty matrices
#' @return survPen object with standard parameterization
#'
inv.repam <- function(model,X.ini,S.pen.ini){
	
	U <- model$U.F
	T.U <- t(U)
	
	coef.name <- colnames(model$X)
	
	model$coefficients <- U%vec%model$coefficients
	model$X <- X.ini
	model$S.pen <- S.pen.ini
	model$S.list <- lapply(1:model$nb.smooth,function(i) model$lambda[i]*model$S.pen[[i]])
	model$S <- Reduce("+",model$S.list)

	model$grad.unpen.beta <- U%vec%model$grad.unpen.beta
	model$grad.beta <- U%vec%model$grad.beta
	
	model$Hess.unpen.beta <- U%mult%model$Hess.unpen.beta%mult%T.U
	model$Hess.unpen.beta <- 0.5*(model$Hess.unpen.beta + t(model$Hess.unpen.beta)) # to be sure Hess.unpen.beta is symmetric
	
	model$Hess.beta <- model$Hess.unpen.beta - model$S
	
	model$Ve <- U%mult%model$Ve%mult%T.U
	model$Ve <- 0.5*(model$Ve + t(model$Ve)) # to be sure Ve is symmetric
	
	model$Vp <- U%mult%model$Vp%mult%T.U
	model$Vp <- 0.5*(model$Vp + t(model$Vp)) # to be sure Vp is symmetric
	
	# very important to recalculate the edf in the initial parameterization if we want to keep the good
	# termwise edf
	model$edf <- rowSums(-model$Vp*(model$Hess.beta + model$S))
	
	model$edf1 <- 2*model$edf - rowSums(t(model$Vp%mult%model$Hess.unpen.beta)*(model$Vp%mult%model$Hess.unpen.beta))
	
	rownames(model$Vp) <- colnames(model$Vp) <- rownames(model$Ve) <- colnames(model$Ve) <-
	rownames(model$Hess.unpen.beta) <- colnames(model$Hess.unpen.beta) <- rownames(model$Hess.beta) <- colnames(model$Hess.beta) <-
	names(model$edf) <- names(model$edf1) <- colnames(model$X) <- names(model$grad.beta) <- names(model$grad.unpen.beta) <- names(model$coefficients) <- coef.name
	
	optim.rho <- !is.null(model$optim.rho)
	
	if (optim.rho){
	
		rownames(model$Hess.rho) <- colnames(model$Hess.rho) <- names(model$grad.rho) <- names(model$lambda)
	
		model$deriv.rho.inv.Hess.beta <- lapply(1:model$nb.smooth,function(i) U%mult%model$deriv.rho.inv.Hess.beta[[i]]%mult%T.U)
		model$deriv.rho.beta <- model$deriv.rho.beta%mult%T.U
		
	}
	
	return(model)

}
#----------------------------------------------------------------------------------------------------------------
# END of code : inv.repam
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
# cor.var : implementation of Vc
#----------------------------------------------------------------------------------------------------------------

#' Implementation of the corrected variance Vc
#'
#' Takes the model at convergence and calculates the variance matrix corrected for smoothing parameter uncertainty
#'
#' @param model survPen object, see \code{\link{survPen.fit}} for details
#' @return survPen object with corrected variance Vc
#'
cor.var <- function(model){

	#---------------------------------------------------------------------------------------------------------
	# for corrected variance, we need the derivative of R wrt smooting parameters with t(R)%mult%R=Vp
	model$deriv.R1 <- deriv_R(model$deriv.rho.inv.Hess.beta,model$p,chol(model$Vp))	
		
	#-------------- regularization of the inverse Hessian when a smoothing parameter tends to infinity
	eigen.cor <- eigen(model$Hess.rho,symmetric=TRUE)
	U.cor <- eigen.cor$vectors
	vp.cor <- eigen.cor$values

	ind <- vp.cor <= 0
	
	vp.cor[ind] <- 0;vp.cor <- 1/sqrt(vp.cor+1/10) # same as mgcv version 1-8.29 
	# function gam.fit3.post.proc, line 930 in https://github.com/cran/mgcv/blob/master/R/gam.fit3.r
    
	rV <- (vp.cor*t(U.cor)) ## root of cov matrix
	
    inv.Hess.rho <- crossprod(rV)
	
	# V second, second part of the corrected covariance matrix
	V.second <- matrix(0,model$p,model$p)

	for (l in 1:model$nb.smooth){

		for (k in 1:model$nb.smooth){

			prod.temp <- (model$deriv.R1[[k]]%cross%model$deriv.R1[[l]])*inv.Hess.rho[k,l]
				
			V.second <- V.second + prod.temp
				
		}

	}
	model$V.second <- V.second
	#---------------------------------------------------------------------------------------------------------

	# corrected variance (for smoothing parameter uncertainty), Kass and Steffey approximation
	Vc.approx <- model$Vp + crossprod(rV%mult%model$deriv.rho.beta)
	
	# full corrected variance
	Vc <- Vc.approx + V.second
		
	coef.name <- colnames(model$X)
	rownames(Vc.approx) <- colnames(Vc.approx) <- rownames(Vc) <- colnames(Vc) <- coef.name
			
	model$Vc.approx <- Vc.approx
	model$Vc <- Vc
			
	#----------------- corrected AIC
	model$edf2 <- rowSums(-model$Hess.unpen*model$Vc)
				
	# edf1 is supposed to be an upper bound for edf2 (see code from function gam.fit5.post.proc from package mgcv)
	if (sum(model$edf2) >= sum(model$edf1)) model$edf2 <- model$edf1		
			
	model$aic2 <- -2*model$ll.unpen + 2*sum(model$edf2)
		
	model
		
}	
#----------------------------------------------------------------------------------------------------------------
# END of code : cor.var
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
# survPen : fits a multidimensionnal penalized survival model on the logarithm of the (excess) hazard
# This function estimates automatically the smoohting parameters via NR.rho and calls survPen.fit for
# estimation of the betas
#----------------------------------------------------------------------------------------------------------------

#' (Excess) hazard model with (multidimensional) penalized splines and integrated smoothness estimation
#'
#' Fits an (excess) hazard model with (multidimensional) penalized splines allowing for
#' time-dependent effects, non-linear effects and interactions between several continuous covariates. The linear predictor is specified on the logarithm of the (excess) hazard. Smooth terms are represented using
#' cubic regression splines with associated quadratic penalties. For multidimensional smooths, tensor product splines or tensor product interactions
#' are available. Smoothness is estimated automatically by optimizing one of two criteria: Laplace approximate marginal likelihood (LAML) or likelihood cross-validation (LCV).
#' When specifying the model's formula, no distinction is made between the part relative to the form of the baseline hazard and the one relative
#' to the effects of the covariates. Thus, time-dependent effects are naturally specified as interactions with some function of time via "*" or ":". See the examples below for more details.
#' The main functions of the survPen package are \code{\link{survPen}}, \code{\link{smf}}, \code{\link{tensor}}, \code{\link{tint}} and \code{\link{rd}}. The first one fits the model while the other four are constructors for penalized splines. \cr \cr
#' The user must be aware that the \code{survPen} package does not depend on \code{mgcv}. Thus, all the functionalities available in \code{mgcv} in terms of types of splines (such as thin plate regression splines or P-splines) are not available in \code{survPen} (yet).
#'
#' @param formula formula object specifying the model. Penalized terms are specified using \code{\link{smf}} (comparable to \code{s(...,bs="cr")} in \code{mgcv}),
#' \code{\link{tensor}} (comparable to \code{te(...,bs="cr")} in \code{mgcv}), \code{\link{tint}} (comparable to \code{ti(...,bs="cr")} in \code{mgcv}),
#' or \code{\link{rd}} (comparable to \code{s(...,bs="re")} in \code{mgcv}).
#' @param data an optional data frame containing the variables in the model
#' @param t1 vector of follow-up times or name of the column in \code{data} containing follow-up times
#' @param t0 vector of origin times or name of the column in \code{data} containing origin times; allows to take into account left truncation; default is NULL, in which case it will be a vector of zeroes
#' @param event vector of right-censoring indicators or name of the column in \code{data} containing right-censoring indicators; 1 if the event occurred and 0 otherwise
#' @param expected (for net survival only) vector of expected hazard or name of the column in \code{data} containing expected hazard; default is NULL, in which case overall survival will be estimated
#' @param lambda vector of smoothing parameters; default is NULL when it is to be estimated by LAML or LCV
#' @param rho.ini vector of initial log smoothing parameters; default is NULL, in which case every initial log lambda will be -1
#' @param max.it.beta maximum number of iterations to reach convergence in the regression parameters; default is 200
#' @param max.it.rho maximum number of iterations to reach convergence in the smoothing parameters; default is 30
#' @param beta.ini vector of initial regression parameters; default is NULL, in which case the first beta will be \code{log(sum(event)/sum(t1))} and the others will be zero (except if there are "by" variables in which case all betas are set to zero)
#' @param detail.rho if TRUE, details concerning the optimization process in the smoothing parameters are displayed; default is FALSE
#' @param detail.beta if TRUE, details concerning the optimization process in the regression parameters are displayed; default is FALSE
#' @param n.legendre number of Gauss-Legendre quadrature nodes to be used to compute the cumulative hazard; default is 20
#' @param method criterion used to select the smoothing parameters. Should be "LAML" or "LCV"; default is "LAML"
#' @param tol.beta convergence tolerance for regression parameters; default is \code{1e-04}. See \code{\link{NR.beta}} for details
#' @param tol.rho convergence tolerance for smoothing parameters; default is \code{1e-04}. See \code{\link{NR.rho}} for details
#' @param step.max maximum absolute value possible for any component of the step vector (on the log smoothing parameter scale) in LCV or LAML optimization; default is 5. If necessary, consider lowering this value to achieve convergence
#' @return Object of class "survPen" (see \code{\link{survPenObject}} for details)
#' @export
#'
#' @details
#' In time-to-event analysis, we may deal with one or several continuous covariates whose functional forms, time-dependent effects and interaction structure are challenging. 
#' One possible way to deal with these effects and interactions is to use the classical approximation of the survival likelihood by a Poisson likelihood. Thus, by artificially splitting 
#' the data, the package \code{mgcv} can then be used to fit penalized hazard models (Remontet et al. 2018). The problem with this option is that the setup is rather complex and the method can fail with huge datasets (before splitting).  
#' Wood et al. (2016) provided a general penalized framework that made available smooth function estimation to a wide variety of models. 
#' They proposed to estimate smoothing parameters by maximizing a Laplace approximate marginal likelihood (LAML) criterion and demonstrate how statistical consistency is maintained by doing so.
#' The \code{\link{survPen}} function implements the framework described by Wood et al. (2016) for modelling time-to-event data without requiring data splitting and Poisson likelihood approximation.
#' The effects of continuous covariates are represented using low rank spline bases with associated quadratic penalties. The \code{\link{survPen}} function allows to account simultaneously for time-dependent effects, non-linear effects and
#' interactions between several continuous covariates without the need to build a possibly demanding model-selection procedure.
#' Besides LAML, a likelihood cross-validation (LCV) criterion (O Sullivan 1988) can be used for smoothing parameter estimation.
#' First and second derivatives of LCV with respect to the smoothing parameters are implemented so that LCV optimization is computationally equivalent to the LAML optimization proposed by Wood et al. (2016).
#' In practice, LAML optimization is generally both a bit faster and a bit more stable so it is used as default. 
#' For \eqn{m} covariates \eqn{(x_1,\ldots,x_m)}, if we note \eqn{h(t,x_1,\ldots,x_m)} the hazard at time \eqn{t}, the hazard model is the following :
#' \deqn{log[h(t,x_1,\ldots,x_m)]=\sum_j g_j(t,x_1,\ldots,x_m)}
#'
#' where each \eqn{g_j} is either the marginal basis of a specific covariate or a tensor product smooth of any number of covariates. The marginal bases of the covariates are represented
#' as natural (or restricted) cubic splines (as in function \code{ns} from library \code{splines}) with associated quadratic penalties. Full parametric (unpenalized) terms for the effects of covariates are also possible (see the examples below).
#' Each \eqn{g_j} is then associated with zero, one or several smoothing parameters. 
#' The estimation procedure is based on outer Newton-Raphson iterations for the smoothing parameters and on inner Newton-Raphson iterations for the regression parameters (see Wood et al. 2016).
#' Estimation of the regression parameters in the inner algorithm is by direct maximization of the penalized likelihood of the survival model, therefore avoiding data augmentation and Poisson likelihood approximation. 
#' The cumulative hazard included in the log-likelihood is approximated by Gauss-Legendre quadrature for numerical stability.
#'
#' @section by variables:
#' The \code{\link{smf}}, \code{\link{tensor}} and \code{\link{tint}} terms used to specify smooths accept an argument \code{by}. This \code{by} argument allows for building varying-coefficient models i.e. for letting
#' smooths interact with factors or parametric terms. If a \code{by} variable is numeric, then its ith element multiples the ith row of the model matrix corresponding to the smooth term concerned.
#' If a \code{by} variable is a factor then it generates an indicator vector for each level of the factor, unless it is an ordered factor. In the non-ordered case, the model matrix for the smooth term is then replicated 
#' for each factor level, and each copy has its rows multiplied by the corresponding rows of its indicator variable. The smoothness penalties are also duplicated for each factor level. In short a different smooth is generated 
#' for each factor level. The main interest of by variables over separated models is the \code{same.rho} argument (for \code{\link{smf}}, \code{\link{tensor}} and \code{\link{tint}}) which allows forcing all smooths to have the same smoothing parameter(s). 
#' Ordered \code{by} variables are handled in the same way, except that no smooth is generated for the first level of the ordered factor. This is useful if you are interested in differences from a reference level.
#'
#' See the survival_analysis_with_survPen vignette for more details. 
#' 
#' @section Random effects:
#' i.i.d random effects can be specified using penalization. Indeed, the ridge penalty is equivalent to an assumption that the regression parameters are i.i.d. normal random effects.
#' Thus, it is easy to fit a frailty hazard model. For example, consider the model term \code{rd(clust)} which will result in a model matrix component corresponding to \code{model.matrix(~clust-1)} being added to the model matrix for the whole model. 
#' The associated regression parameters are assumed i.i.d. normal, with unknown variance (to be estimated). This assumption is equivalent to an identity penalty matrix (i.e. a ridge penalty) on the regression parameters.
#' The unknown smoothing parameter \eqn{\lambda} associated with the term \code{rd(clust)} is directly linked to the unknown variance \eqn{\sigma^2}: \eqn{\sigma^2 = \frac{1}{\lambda * S.scale}}.
#' Then, the estimated log standard deviation is: \eqn{log(\hat{\sigma})=-0.5*log(\hat{\lambda})-0.5*log(S.scale)}. And the estimated variance of the log standard deviation is: \eqn{Var[log(\hat{\sigma})]=0.25*Var[log(\hat{\lambda})]=0.25*inv.Hess.rho}.
#' See the survival_analysis_with_survPen vignette for more details.
#' This approach allows implementing commonly used random effect structures. For example if \code{g} is a factor then \code{rd(g)} produces a random parameter for each level of \code{g}, the random parameters being i.i.d. normal. 
#' If \code{g} is a factor and \code{x} is numeric, then \code{rd(g,x)} produces an i.i.d. normal random slope relating the response to \code{x} for each level of \code{g}.
#' Thus, random effects treated as penalized splines allow specifying frailty (excess) hazard models (Charvat et al. 2016). For each individual i from cluster (usually geographical unit)  j, a possible model would be:
#' \deqn{log[h(t_{ij},x_{ij1},\ldots,x_{ijm})]=\sum_k g_k(t_{ij},x_{ij1},\ldots,x_{ijm}) + w_j}
#'
#' where \code{w_j} follows a normal distribution with mean 0. The random effect associated with the cluster variable is specified with the model term \code{rd(cluster)}. We could also specify a random effect depending on age for example with the model term \code{rd(cluster,age)}.
#' \code{u_j = exp(w_j)} is known as the shared frailty.
#'
#' See the survival_analysis_with_survPen vignette for more details.
#'
#' @section Excess hazard model:
#' When studying the survival of patients who suffer from a common pathology we may be interested in the concept of excess mortality that represents the mortality due to that pathology. 
#' For example, in cancer epidemiology, individuals may die from cancer or from another cause. The problem is that the cause of death is often either unavailable or unreliable. 
#' Supposing that the mortality due to other causes may be obtained from the total mortality of the general population (called expected mortality for cancer patients), we can define the concept of excess mortality. 
#' The excess mortality is directly linked to the concept of net survival, which would be the observed survival if patients could not die from other causes. Therefore, when such competing events are present, 
#' one may choose to fit an excess hazard model instead of a classical hazard model. Flexible excess hazard models have already been proposed (for examples see Remontet et al. 2007, Charvat et al. 2016) but none of them deals with a penalized framework (in a non-fully Bayesian setting).
#' Excess mortality can be estimated supposing that, in patients suffering from a common pathology, mortality due to others causes than the pathology can be obtained from the (all cause) mortality of the general population; the latter is referred to as the expected mortality \eqn{h_P}. 
#' The mortality observed in the patients (\eqn{h_O}) is actually decomposed as the sum of \eqn{h_P} and the excess mortality due to the pathology (\eqn{h_E}). This may be written as:
#' \deqn{h_O(t,x)=h_E(t,x)+h_P(a+t,z)}
#' In that equation, \eqn{t} is the time since cancer diagnosis, \eqn{a} is the age at diagnosis, \eqn{h_P} is the mortality of the general population at age \eqn{a+t} given demographical characteristics \eqn{z} (\eqn{h_P} is considered known and available from national statistics), 
#' and \eqn{x} a vector of variables that may have an effect on \eqn{h_E}. Including the age in the model is necessary in order to deal with the informative censoring due to other causes of death. 
#' Thus, for \eqn{m} covariates \eqn{(x_1,\ldots,x_m)}, if we note \eqn{h_E(t,x_1,\ldots,x_m)} the excess hazard at time \eqn{t}, the excess hazard model is the following:
#' \deqn{log[h_E(t,x_1,\ldots,x_m)]=\sum_j g_j(t,x_1,\ldots,x_m)}
#'
#' @section Convergence:
#' No convergence indicator is given. If the function returns an object of class \code{survPen}, it means that the algorithm has converged. If convergence issues occur, an error message is displayed.
#' If convergence issues occur, do not refrain to use detail.rho and/or detail.beta to see exactly what is going on in the optimization process. To achieve convergence, consider lowering step.max and/or changing rho.ini and beta.ini.
#' If your excess hazard model fails to converge, consider fitting a hazard model and use its estimated parameters as initial values for the excess hazard model. Finally, do not refrain to change the "method" argument (LCV or LAML) if convergence issues occur.
#'
#' @section Other:
#' Be aware that all character variables are transformed to factors before fitting.
#'
#' @references
#' Charvat, H., Remontet, L., Bossard, N., Roche, L., Dejardin, O., Rachet, B., ... and Belot, A. (2016), A multilevel excess hazard model to estimate net survival on hierarchical data allowing for non linear and non proportional effects of covariates. Statistics in medicine, 35(18), 3066-3084. \cr \cr
#' Fauvernier, M., Roche, L., Uhry, Z., Tron, L., Bossard, N., Remontet, L. and the CENSUR Working Survival Group. Multidimensional penalized hazard model with continuous covariates: applications for studying trends and social inequalities in cancer survival, in revision in the Journal of the Royal Statistical Society, series C. \cr \cr
#' O Sullivan, F. (1988), Fast computation of fully automated log-density and log-hazard estimators. SIAM Journal on scientific and statistical computing, 9(2), 363-379. \cr \cr
#' Remontet, L., Bossard, N., Belot, A., & Esteve, J. (2007), An overall strategy based on regression models to estimate relative survival and model the effects of prognostic factors in cancer survival studies. Statistics in medicine, 26(10), 2214-2228. \cr \cr
#' Remontet, L., Uhry, Z., Bossard, N., Iwaz, J., Belot, A., Danieli, C., Charvat, H., Roche, L. and CENSUR Working Survival Group (2018) Flexible and structured survival model for a simultaneous estimation of non-linear and non-proportional effects and complex interactions between continuous variables: Performance of this multidimensional penalized spline approach in net survival trend analysis. Stat Methods Med Res. 2018 Jan 1:962280218779408. doi: 10.1177/0962280218779408. [Epub ahead of print]. \cr \cr
#' Wood, S.N., Pya, N. and Saefken, B. (2016), Smoothing parameter and model selection for general smooth models (with discussion). Journal of the American Statistical Association 111, 1548-1575
#'
#' @examples
#'
#' \donttest{
#'
#' library(survPen)
#' data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer
#' 
#' #-------------------------------------------------------- example 0
#' # Comparison between restricted cubic splines and penalized restricted cubic splines
#'
#' library(splines)
#'
#' # unpenalized
#' f <- ~ns(fu,knots=c(0.25, 0.5, 1, 2, 4),Boundary.knots=c(0,5))
#'
#' mod <- survPen(f,data=datCancer,t1=fu,event=dead)
#'
#' # penalized
#' f.pen <- ~ smf(fu,knots=c(0,0.25, 0.5, 1, 2, 4,5)) # careful here: the boundary knots are included
#'
#' mod.pen <- survPen(f.pen,data=datCancer,t1=fu,event=dead)
#'
#' # predictions
#'
#' new.time <- seq(0,5,length=100)
#' pred <- predict(mod,data.frame(fu=new.time))
#' pred.pen <- predict(mod.pen,data.frame(fu=new.time))
#'
#' par(mfrow=c(1,1))
#' plot(new.time,pred$haz,type="l",ylim=c(0,0.2),main="hazard vs time",
#' xlab="time since diagnosis (years)",ylab="hazard",col="red")
#' lines(new.time,pred.pen$haz,col="blue3")
#' legend("topright",legend=c("unpenalized","penalized"),
#' col=c("red","blue3"),lty=rep(1,2))
#'
#' 
#'
#' #-------------------------------------------------------- example 1
#' # hazard models with unpenalized formulas compared to a penalized tensor product smooth
#'
#' library(survPen)
#' data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer
#'
#' # constant hazard model
#' f.cst <- ~1
#' mod.cst <- survPen(f.cst,data=datCancer,t1=fu,event=dead)
#'
#' # piecewise constant hazard model
#' f.pwcst <- ~cut(fu,breaks=seq(0,5,by=0.5),include.lowest=TRUE)
#' mod.pwcst <- survPen(f.pwcst,data=datCancer,t1=fu,event=dead,n.legendre=200)
#' # we increase the number of points for Gauss-Legendre quadrature to make sure that the cumulative
#' # hazard is properly approximated
#'
#' # linear effect of time
#' f.lin <- ~fu
#' mod.lin <- survPen(f.lin,data=datCancer,t1=fu,event=dead)
#'
#' # linear effect of time and age with proportional effect of age
#' f.lin.age <- ~fu+age
#' mod.lin.age <- survPen(f.lin.age,data=datCancer,t1=fu,event=dead)
#'
#' # linear effect of time and age with time-dependent effect of age (linear)
#' f.lin.inter.age <- ~fu*age
#' mod.lin.inter.age <- survPen(f.lin.inter.age,data=datCancer,t1=fu,event=dead)
#'
#' # cubic B-spline of time with a knot at 1 year, linear effect of age and time-dependent effect
#' # of age with a quadratic B-spline of time with a knot at 1 year
#' library(splines)
#' f.spline.inter.age <- ~bs(fu,knots=c(1),Boundary.knots=c(0,5))+age+
#' age:bs(fu,knots=c(1),Boundary.knots=c(0,5),degree=2)
#' # here, bs indicates an unpenalized cubic spline
#'
#' mod.spline.inter.age <- survPen(f.spline.inter.age,data=datCancer,t1=fu,event=dead)
#'
#'
#' # tensor of time and age
#' f.tensor <- ~tensor(fu,age)
#' mod.tensor <- survPen(f.tensor,data=datCancer,t1=fu,event=dead)
#'
#'
#' # predictions of the models at age 60
#'
#' new.time <- seq(0,5,length=100)
#' pred.cst <- predict(mod.cst,data.frame(fu=new.time))
#' pred.pwcst <- predict(mod.pwcst,data.frame(fu=new.time))
#' pred.lin <- predict(mod.lin,data.frame(fu=new.time))
#' pred.lin.age <- predict(mod.lin.age,data.frame(fu=new.time,age=60))
#' pred.lin.inter.age <- predict(mod.lin.inter.age,data.frame(fu=new.time,age=60))
#' pred.spline.inter.age <- predict(mod.spline.inter.age,data.frame(fu=new.time,age=60))
#' pred.tensor <- predict(mod.tensor,data.frame(fu=new.time,age=60))
#' 
#' lwd1 <- 2
#' 
#' par(mfrow=c(1,1))
#' plot(new.time,pred.cst$haz,type="l",ylim=c(0,0.2),main="hazard vs time",
#' xlab="time since diagnosis (years)",ylab="hazard",col="blue3",lwd=lwd1)
#' segments(x0=new.time[1:99],x1=new.time[2:100],y0=pred.pwcst$haz[1:99],col="lightblue2",lwd=lwd1)
#' lines(new.time,pred.lin$haz,col="green3",lwd=lwd1)
#' lines(new.time,pred.lin.age$haz,col="yellow",lwd=lwd1)
#' lines(new.time,pred.lin.inter.age$haz,col="orange",lwd=lwd1)
#' lines(new.time,pred.spline.inter.age$haz,col="red",lwd=lwd1)
#' lines(new.time,pred.tensor$haz,col="black",lwd=lwd1)
#' legend("topright",
#' legend=c("cst","pwcst","lin","lin.age","lin.inter.age","spline.inter.age","tensor"),
#' col=c("blue3","lightblue2","green3","yellow","orange","red","black"),
#' lty=rep(1,7),lwd=rep(lwd1,7))
#' 
#'
#' # you can also calculate the hazard yourself with the lpmatrix option.
#' # For example, compare the following predictions:
#' haz.tensor <- pred.tensor$haz
#'
#' X.tensor <- predict(mod.tensor,data.frame(fu=new.time,age=60),type="lpmatrix")
#' haz.tensor.lpmatrix <- exp(X.tensor%mult%mod.tensor$coefficients)
#'
#' summary(haz.tensor.lpmatrix - haz.tensor)
#'
#' #---------------- The 95% confidence intervals can be calculated like this:
#' 
#' # standard errors from the Bayesian covariance matrix Vp
#' std <- sqrt(rowSums((X.tensor%mult%mod.tensor$Vp)*X.tensor))
#' 
#' qt.norm <- stats::qnorm(1-(1-0.95)/2)
#' haz.inf <- as.vector(exp(X.tensor%mult%mod.tensor$coefficients-qt.norm*std))
#' haz.sup <- as.vector(exp(X.tensor%mult%mod.tensor$coefficients+qt.norm*std))
#' 
#' # checking that they are similar to the ones given by the predict function
#' summary(haz.inf - pred.tensor$haz.inf)
#' summary(haz.sup - pred.tensor$haz.sup)
#' 
#'
#' #-------------------------------------------------------- example 2
#' 
#' library(survPen)
#' data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer
#'
#' # model : unidimensional penalized spline for time since diagnosis with 5 knots
#' f1 <- ~smf(fu,df=5)
#' # when knots are not specified, quantiles are used. For example, for the term "smf(x,df=df1)",
#' # the vector of knots will be: quantile(unique(x),seq(0,1,length=df1)) 
#'
#' # you can specify your own knots if you want
#' # f1 <- ~smf(fu,knots=c(0,1,3,6,8))
#' 
#' # hazard model
#' mod1 <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LAML")
#' summary(mod1)
#' 
#' # to see where the knots were placed
#' mod1$list.smf
#' 
#' # with LCV instead of LAML
#' mod1bis <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LCV")
#' summary(mod1bis)
#' 
#' # hazard model taking into account left truncation (not representative of cancer data, 
#' # the begin variable was simulated for illustration purposes only)
#' mod2 <- survPen(f1,data=datCancer,t0=begin,t1=fu,event=dead,expected=NULL,method="LAML")
#' summary(mod2)
#' 
#' # excess hazard model
#' mod3 <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=rate,method="LAML")
#' summary(mod3)
#' 
#' # compare the predictions of the models
#' new.time <- seq(0,5,length=50)
#' pred1 <- predict(mod1,data.frame(fu=new.time))
#' pred1bis <- predict(mod1bis,data.frame(fu=new.time))
#' pred2 <- predict(mod2,data.frame(fu=new.time))
#' pred3 <- predict(mod3,data.frame(fu=new.time))
#' 
#' # LAML vs LCV
#' par(mfrow=c(1,2))
#' plot(new.time,pred1$haz,type="l",ylim=c(0,0.2),main="LCV vs LAML",
#' xlab="time since diagnosis (years)",ylab="hazard")
#' lines(new.time,pred1bis$haz,col="blue3")
#' legend("topright",legend=c("LAML","LCV"),col=c("black","blue3"),lty=c(1,1))
#' 
#' plot(new.time,pred1$surv,type="l",ylim=c(0,1),main="LCV vs LAML",
#' xlab="time since diagnosis (years)",ylab="survival")
#' lines(new.time,pred1bis$surv,col="blue3")
#' 
#' 
#' 
#' # hazard vs excess hazard
#' par(mfrow=c(1,2))
#' plot(new.time,pred1$haz,type="l",ylim=c(0,0.2),main="hazard vs excess hazard",
#' xlab="time since diagnosis (years)",ylab="hazard")
#' lines(new.time,pred3$haz,col="green3")
#' legend("topright",legend=c("overall","excess"),col=c("black","green3"),lty=c(1,1))
#' 
#' plot(new.time,pred1$surv,type="l",ylim=c(0,1),main="survival vs net survival",
#' xlab="time",ylab="survival")
#' lines(new.time,pred3$surv,col="green3")
#' legend("topright",legend=c("overall survival","net survival"), col=c("black","green3"), lty=c(1,1)) 
#'
#' # hazard vs excess hazard with 95% Bayesian confidence intervals (based on Vp matrix, 
#' # see predict.survPen)
#' par(mfrow=c(1,1))
#' plot(new.time,pred1$haz,type="l",ylim=c(0,0.2),main="hazard vs excess hazard",
#' xlab="time since diagnosis (years)",ylab="hazard")
#' lines(new.time,pred3$haz,col="green3")
#' legend("topright",legend=c("overall","excess"),col=c("black","green3"),lty=c(1,1))
#' 
#' lines(new.time,pred1$haz.inf,lty=2)
#' lines(new.time,pred1$haz.sup,lty=2)
#' 
#' lines(new.time,pred3$haz.inf,lty=2,col="green3")
#' lines(new.time,pred3$haz.sup,lty=2,col="green3")
#' 
#'
#'
#' #-------------------------------------------------------- example 3
#'
#' library(survPen)
#' data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer
#'
#' # models: tensor product smooth vs tensor product interaction of time since diagnosis and 
#' # age at diagnosis. Smoothing parameters are estimated via LAML maximization
#' f2 <- ~tensor(fu,age,df=c(5,5))
#' 
#' f3 <- ~tint(fu,df=5)+tint(age,df=5)+tint(fu,age,df=c(5,5))
#' 
#' # hazard model
#' mod4 <- survPen(f2,data=datCancer,t1=fu,event=dead)
#' summary(mod4)
#' 
#' mod5 <- survPen(f3,data=datCancer,t1=fu,event=dead)
#' summary(mod5)
#' 
#' # predictions
#' new.age <- seq(50,90,length=50)
#' new.time <- seq(0,7,length=50)
#' 
#' Z4 <- outer(new.time,new.age,function(t,a) predict(mod4,data.frame(fu=t,age=a))$haz)
#' Z5 <- outer(new.time,new.age,function(t,a) predict(mod5,data.frame(fu=t,age=a))$haz)
#' 
#' # color settings
#' col.pal <- colorRampPalette(c("white", "red"))
#' colors <- col.pal(100)
#' 
#' facet <- function(z){
#' 
#' 	facet.center <- (z[-1, -1] + z[-1, -ncol(z)] + z[-nrow(z), -1] + z[-nrow(z), -ncol(z)])/4
#' 	cut(facet.center, 100)
#' 	
#' }
#' 
#' # plot the hazard surfaces for both models
#' par(mfrow=c(1,2))
#' persp(new.time,new.age,Z4,col=colors[facet(Z4)],main="tensor",theta=30,
#' xlab="time since diagnosis",ylab="age at diagnosis",zlab="excess hazard",ticktype="detailed")
#' persp(new.time,new.age,Z5,col=colors[facet(Z5)],main="tint",theta=30,
#' xlab="time since diagnosis",ylab="age at diagnosis",zlab="excess hazard",ticktype="detailed")
#' 
#' #-------------------------------------------------------- example 4
#' 
#' library(survPen)
#' data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer
#'
#' # model : tensor product spline for time, age and yod (year of diagnosis)
#' # yod is not centered here since it does not create unstability but be careful in practice
#' # and consider centering your covariates if you encounter convergence issues
#' f4 <- ~tensor(fu,age,yod,df=c(5,5,5))
#' 
#' # excess hazard model
#' mod6 <- survPen(f4,data=datCancer,t1=fu,event=dead,expected=rate)
#' summary(mod6)
#' 
#' 
#' # predictions of the surfaces for ages 50, 60, 70 and 80
#' new.year <- seq(1990,2010,length=30)
#' new.time <- seq(0,5,length=50)
#' 
#' Z_50 <- outer(new.time,new.year,function(t,y) predict(mod6,data.frame(fu=t,yod=y,age=50))$haz)
#' Z_60 <- outer(new.time,new.year,function(t,y) predict(mod6,data.frame(fu=t,yod=y,age=60))$haz)
#' Z_70 <- outer(new.time,new.year,function(t,y) predict(mod6,data.frame(fu=t,yod=y,age=70))$haz)
#' Z_80 <- outer(new.time,new.year,function(t,y) predict(mod6,data.frame(fu=t,yod=y,age=80))$haz)
#' 
#' 
#' # plot the hazard surfaces for a given age
#' par(mfrow=c(2,2))
#' persp(new.time,new.year,Z_50,col=colors[facet(Z_50)],main="age 50",theta=20,
#' xlab="time since diagnosis",ylab="yod",zlab="excess hazard",ticktype="detailed")
#' persp(new.time,new.year,Z_60,col=colors[facet(Z_60)],main="age 60",theta=20,
#' xlab="time since diagnosis",ylab="yod",zlab="excess hazard",ticktype="detailed")
#' persp(new.time,new.year,Z_70,col=colors[facet(Z_70)],main="age 70",theta=20,
#' xlab="time since diagnosis",ylab="yod",zlab="excess hazard",ticktype="detailed")
#' persp(new.time,new.year,Z_80,col=colors[facet(Z_80)],main="age 80",theta=20,
#' xlab="time since diagnosis",ylab="yod",zlab="excess hazard",ticktype="detailed")
#' 
#' ########################################
#'
#' }
#' 
survPen <- function(formula,data,t1,t0=NULL,event,expected=NULL,lambda=NULL,rho.ini=NULL,max.it.beta=200,max.it.rho=30,beta.ini=NULL,detail.rho=FALSE,detail.beta=FALSE,n.legendre=20,method="LAML",tol.beta=1e-04,tol.rho=1e-04,step.max=5){

	#------------------------------------------
	# initialization
	
	cl <- match.call()

	if (missing(formula) | missing(data) | missing(t1) | missing(event)) stop("Must have at least a formula, data, t1 and event arguments")

	formula <- stats::as.formula(formula)
	
	if (!(method %in% c("LAML","LCV"))) stop("method should be LAML or LCV")

	data <- as.data.frame(unclass(data),stringsAsFactors=TRUE) # converts all characters to factors
	
	# keeps track of factor levels
	factor.term <- names(data)[sapply(data,is.factor)]
	
	# erase factor modalities for which there is no observation
	for (factor.name in names(data)[names(data)%in%factor.term]){

		data[,factor.name]<-factor(data[,factor.name])

	}
	
	# copy factor attributes for prediction
	factor.structure <- lapply(as.data.frame(data[,names(data)%in%factor.term]),attributes)
	names(factor.structure) <- factor.term
	
	t1.name <- deparse(substitute(t1))
	t1 <- eval(substitute(t1), data)

	if (!is.numeric(t1)) stop("t1 variable is not numeric")

	n <- length(t1)

	t0.name <- deparse(substitute(t0))
	t0 <- eval(substitute(t0), data)

	event.name <- deparse(substitute(event))
	event <- eval(substitute(event), data)

	expected.name <- deparse(substitute(expected))
	expected <- eval(substitute(expected), data)

	if (is.null(expected)){
	
		type <- "overall"
		
	}else{
	
		type <- "net"
		
	}

	if (is.null(t0)) t0 <- rep(0,n)
	if (length(t0) == 1) t0 <- rep(t0,n)
	
	if (is.null(event)) event <- rep(1,n)
	if (is.null(expected)) expected <- rep(0,n)

	if (any(t0>t1)) stop("some t0 values are superior to t1 values")
	if (length(t0) != n) stop("t0 and t1 are different lengths")
	if (length(event) != n) stop("event and t1 are different lengths")
	if (length(expected) != n) stop("expected and t1 are different lengths")

	#------------------------------------------
	# setting up the design and penalty matrices 
	
	build <- model.cons(formula,lambda,data,t1,t1.name,t0,t0.name,event,event.name,expected,expected.name,type,n.legendre,cl,beta.ini)
	
	#------------------------------------------
	# optimization procedures. For given smoothing parameters we call survPen.fit. Otherwise we call NR.rho that
	# will call survPen.fit for each trial of smoothing parameters in the Newton-Raphson algorithm.
	if (is.null(lambda)){

		nb.smooth <- build$nb.smooth

		if(nb.smooth!=0){ # Are there any penalized terms ?

			if (is.null(rho.ini)) rho.ini <- rep(-1,nb.smooth) # initial values for log(lambda)
		
			if (length(rho.ini)!=nb.smooth){ 
				
				if (length(rho.ini)==1){ 
				
					rho.ini <- rep(rho.ini,nb.smooth)
				
				}else{
					stop("number of initial log smoothing parameters incorrect")
				}
			}
			# Initial reparameterization
			param <- repam(build)
			build <- param$build
			X.ini <- param$X.ini
			S.pen.ini <- param$S.pen.ini
			beta.ini <- build$beta.ini
			
			# smoothing parameters are to be selected, optimization of LCV or LAML criterion
			model <- NR.rho(build,rho.ini=rho.ini,data=data,formula=formula,max.it.beta=max.it.beta,max.it.rho=max.it.rho,beta.ini=beta.ini,
			detail.rho=detail.rho,detail.beta=detail.beta,nb.smooth=nb.smooth,tol.beta=tol.beta,tol.rho=tol.rho,step.max=step.max,method=method)

			# Inversion of initial reparameterization
			model <- inv.repam(model, X.ini, S.pen.ini)
			
			# corrected variance implementation
			if (model$method=="LAML") model <- cor.var(model)
			
			# factor levels for prediction
			model$factor.structure <- factor.structure
			
			# convergence
			model$converged <- !(model$Hess.beta.modif | model$Hess.rho.modif)
			
			return(model)

		}else{

			# only fully parametric terms in the model so no need for smoothing parameter selection
			build$lambda <- 0

			model <- survPen.fit(build,data=data,formula=formula,max.it.beta=max.it.beta,beta.ini=beta.ini,detail.beta=detail.beta,method=method,tol.beta=tol.beta)

			# factor levels for prediction
			model$factor.structure <- factor.structure
			
			# convergence
			model$converged <- !(model$Hess.beta.modif)
			
			return(model)
			
		  }

	}else{

		# Initial reparameterization
		param <- repam(build)
		build <- param$build
		X.ini <- param$X.ini
		S.pen.ini <- param$S.pen.ini
		beta.ini <- build$beta.ini
		
		# smoothing parameters are given by the user so no need for smoothing parameter selection
		model <- survPen.fit(build,data=data,formula=formula,max.it.beta=max.it.beta,beta.ini=beta.ini,detail.beta=detail.beta,method=method,tol.beta=tol.beta)

		# Inversion of initial reparameterization
		model <- inv.repam(model, X.ini, S.pen.ini)
		
		# factor levels for prediction
		model$factor.structure <- factor.structure
			
		# convergence
		model$converged <- !(model$Hess.beta.modif)
		
		return(model)
		
	}

}

#----------------------------------------------------------------------------------------------------------------
# END of code : survPen
#----------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------
# survPen.fit : fits a multidimensionnal penalized survival model on the logarithm of the (excess) hazard
# The smoohting parameters must be specified
#----------------------------------------------------------------------------------------------------------------


#' (Excess) hazard model with multidimensional penalized splines for given smoothing parameters
#'
#' Fits an (excess) hazard model. If penalized splines are present, the smoothing parameters are specified.
#' @param build list of objects returned by \code{\link{model.cons}}
#' @param data an optional data frame containing the variables in the model
#' @param formula formula object specifying the model
#' @param max.it.beta maximum number of iterations to reach convergence in the regression parameters; default is 200
#' @param beta.ini vector of initial regression parameters; default is NULL, in which case the first beta will be \code{log(sum(event)/sum(t1))} and the others will be zero (except if there are "by" variables in which case all betas are set to zero)
#' @param detail.beta if TRUE, details concerning the optimization process in the regression parameters are displayed; default is FALSE
#' @param method criterion used to select the smoothing parameters. Should be "LAML" or "LCV"; default is "LAML"
#' @param tol.beta convergence tolerance for regression parameters; default is \code{1e-04}. See \code{\link{NR.beta}} for details
#' @return Object of class "survPen" (see \code{\link{survPenObject}} for details)
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 4 knots
#'
#' data <- data.frame(time=seq(0,5,length=100),event=1,t0=0)
#' 
#' form <- ~ smf(time,knots=c(0,1,3,5))
#' 
#' t1 <- eval(substitute(time), data)
#' t0 <- eval(substitute(t0), data)
#' event <- eval(substitute(event), data)
#' 	
#' # Setting up the model before fitting
#' model.c <- model.cons(form,lambda=0,data.spec=data,t1=t1,t1.name="time",
#' t0=rep(0,100),t0.name="t0",event=event,event.name="event",
#' expected=NULL,expected.name=NULL,type="overall",n.legendre=20,
#' cl="survPen(form,data,t1=time,event=event)",beta.ini=NULL)
#'  
#' # fitting
#' mod <- survPen.fit(model.c,data,form)
#'
survPen.fit <- function(build,data,formula,max.it.beta=200,beta.ini=NULL,detail.beta=FALSE,method="LAML",tol.beta=1e-04)
{
  # collecting information from the formula (design matrix, penalty matrices, ...)
  formula <- stats::as.formula(formula)
  
  # we collect every element returned by the model.cons function
  cl <- build$cl

  n <- build$n
  X <- build$X
  
  S <- build$S
  
  S.smf <- build$S.smf
  S.tensor <- build$S.tensor
  S.tint <- build$S.tint
  S.rd <- build$S.rd 
  
  smooth.name.smf <- build$smooth.name.smf
  smooth.name.tensor <- build$smooth.name.tensor
  smooth.name.tint <- build$smooth.name.tint
  smooth.name.rd <- build$smooth.name.rd 
  
  S.scale <- build$S.scale
  lambda <- build$lambda
  rank.S <- build$rank.S
  S.list <- build$S.list
  S.F <- build$S.F
  S.F.list <- build$S.F.list
  U.F <- build$U.F
  # number of parameters
  p <- build$p

  # number of unpenalized parameters
  df.para <- build$df.para
  # number of penalized parameters
  df.smooth <- build$df.smooth

  # We get the penalized terms with their knots and degrees of freedom
  list.smf <- build$list.smf

  list.tensor <- build$list.tensor

  list.tint <- build$list.tint

  list.rd <- build$list.rd
  # number of smoothing parameters
  nb.smooth <- build$nb.smooth

  t0 <- build$t0
  t1 <- build$t1
  tm <- build$tm
  event <- build$event
  expected <- build$expected

  type <- build$type

  Z.smf <- build$Z.smf
  Z.tensor <- build$Z.tensor
  Z.tint <- build$Z.tint

  leg <- build$leg
  n.legendre <- build$n.legendre
  X.GL <- build$X.GL
  
  #-------------------------------------------------------------------
  # Optimization algorithm : Newton-Raphson

  # Initialization of beta.ini is very important
  # All betas except intercept will be initialized to zero
  # intercept will be initialized to log(sum(event)/sum(t1))
  # (except if there are "by" variables in which case all betas are set to zero)

  if (is.null(beta.ini)) {beta.ini=c(log(sum(event)/sum(t1)),rep(0,df.para+df.smooth-1))}

  # if there are "by" variables, we set all initial betas to zero
  if (any(sapply(list.smf,`[`,"by")!="NULL") | 
	any(sapply(list.tensor,`[`,"by")!="NULL") |
	any(sapply(list.tint,`[`,"by")!="NULL") ){

	beta.ini=rep(0,df.para+df.smooth)

  }	

  # Optimization at given smoothing parameters (if any)
  Algo.optim <- NR.beta(build,beta.ini,detail.beta=detail.beta,max.it.beta=max.it.beta,tol.beta=tol.beta)

  # Estimations and likelihoods
  beta <- Algo.optim$beta
  names(beta) <- colnames(X)
  ll.unpen <- Algo.optim$ll.unpen
  ll.pen <- Algo.optim$ll.pen
  haz.GL <- Algo.optim$haz.GL
  iter.beta <- Algo.optim$iter.beta
  #-------------------------------------------------------------------

  # fitted (excess) hazard
  pred1=X%vec%beta
  ft1=exp(pred1)

  #-------------------------------------------------------------------
  # Gradient and Hessian at convergence

  deriv.list <- lapply(1:n.legendre, function(i) X.GL[[i]]*haz.GL[[i]]*tm*leg$weights[i])

  deriv.2.list <- lapply(1:n.legendre, function(i) X.GL[[i]]%cross%(deriv.list[[i]]))

  f.first <- Reduce("+",deriv.list)

  # gradient
  if (type=="net"){
	  grad.unpen.beta <- colSums2(-f.first + X*event*ft1/(ft1+expected))
  }else{
	  grad.unpen.beta <- colSums2(-f.first + X*event)
  }

  grad.beta <- grad.unpen.beta-S%vec%beta

  # Hessian

  f.second <- Reduce("+",deriv.2.list)

  if (type=="net"){
	Hess.unpen.beta <- -f.second + X%cross%(X*event*expected*ft1/(ft1+expected)^2)
  }else{
	Hess.unpen.beta <- -f.second
  }

  # negative Hessian
  neg.Hess.beta <- -Hess.unpen.beta + S

  R <- try(chol(neg.Hess.beta),silent=TRUE)

  Hess.beta.modif <- FALSE

  # Hessian perturbation if necessary (should not be the case at convergence though)
  if(inherits(R,"try-error"))
	{
		Hess.beta.modif <- TRUE
		eigen.temp <- eigen(neg.Hess.beta,symmetric=TRUE)
		U.temp <- eigen.temp$vectors
		vp.temp <- eigen.temp$values

		vp.temp[which(vp.temp<1e-7)] <- 1e-7

		R <- try(chol(U.temp%mult%diag(vp.temp)%mult%t(U.temp)),silent=TRUE)

		warning("beta Hessian was perturbed at convergence")
	}


  neg.Hess.beta <- crossprod(R)

  # Variance
  Vp <- chol2inv(R) # Bayesian variance
  Ve <- -Vp%mult%Hess.unpen.beta%mult%Vp # frequentist variance
  
  
  rownames(Ve) <- colnames(Ve) <- rownames(Vp) <- colnames(Vp) <- colnames(X)
  #-------------------------------------------------------------------

  
  #------------------------------------------------------------------------
  #------------------------------------------------------------------------
  # Smoothing parameter estimation
  #------------------------------------------------------------------------
  #------------------------------------------------------------------------

	
  if(nb.smooth!=0){

	#------------------------------------------------------------------------
	# LCV criterion
	#------------------------------------------------------------------------

	# effective degrees of freedom
	edf <- rowSums(-Hess.unpen.beta*Vp)
	
	# alternative definition
	edf1 <- 2*edf - rowSums(t(Vp%mult%Hess.unpen.beta)*(Vp%mult%Hess.unpen.beta))

	# LCV
	LCV <- -ll.unpen+sum(edf)

	if(method=="LCV"){

			criterion.val <- LCV
			
	}
	#------------------------------------------------------------------------
	# LAML criterion
	#------------------------------------------------------------------------

	if (sum(lambda)<.Machine$double.eps) {

		log.abs.S <- 0
		M.p <- 0
		sub.S <- S[1:rank.S,1:rank.S]
		
	}else{
	
		M.p <- p-rank.S
		sub.S <- S[1:rank.S,1:rank.S]
		qr.S <- qr(sub.S)
		log.abs.S <- sum(log(abs(diag(qr.S$qr))))

	}

	log.det.Hess.beta <- as.numeric(2*determinant(R,logarithm=TRUE)$modulus)

	# this is actually the negative LAML so that we can minimize it
	LAML <- -(ll.pen+0.5*log.abs.S-0.5*log.det.Hess.beta+0.5*M.p*log(2*pi))

	if(method=="LAML"){

			criterion.val <- LAML
			
	}
	#------------------------------------------------------------------------
	# derivatives of LCV and LAML with respect to the smoothing parameters
	#------------------------------------------------------------------------

	if (sum(lambda)>.Machine$double.eps) {
	
		S.beta <- lapply(1:nb.smooth,function(i) S.list[[i]]%vec%beta)
		
		deriv.rho.beta <- matrix(0,nrow=nb.smooth,ncol=p)
		GL.temp <- vector("list",nb.smooth)
		
		for (i in 1:nb.smooth){

			deriv.rho.beta[i,] <- (-Vp)%vec%S.beta[[i]]

			GL.temp[[i]] <- lapply(1:n.legendre, function(j) (X.GL[[j]]%vec%deriv.rho.beta[i,])*haz.GL[[j]])
			
		}
			
		if (type=="net"){
		
			temp.deriv3 <- (X*ft1*(-ft1+expected)/(ft1+expected)^3)
				
			temp.deriv4 <- (X*ft1*(ft1^2-4*expected*ft1+expected^2)/(ft1+expected)^4)
			
		}else{

			temp.deriv3 <- temp.deriv4 <- matrix(0)
		
		}
			
			
		if(method=="LCV"){

			# this calculation is done before to save time
			mat.temp <- -Vp + Ve
			temp.LAML <- vector("list", 0)
			temp.LAML2 <- vector("list", 0)
			inverse.new.S <- matrix(0)
			minus.eigen.inv.Hess.beta <- 0
			
			Hess.LCV1 <- matrix(0,nb.smooth,nb.smooth)
		}
		if(method=="LAML"){

			mat.temp <- matrix(0)
			eigen.mat.temp <- 0
			deriv.mat.temp <- vector("list",0)
			deriv.rho.Ve <- vector("list",0)

			# stable LU decomposition through solve.default that
			# calls an internal R function written in C called La_solve. La_solve itself calls a Fortran routine
			# called DGESV from LAPACK
			inverse.new.S <- try(solve.default(sub.S),silent=TRUE) 
			
			if(inherits(inverse.new.S,"try-error")){
				
				cat("\n","LU decomposition failed to invert penalty matrix, trying QR","\n",
				"set detail.rho=TRUE for details","\n")
				
				inverse.new.S <- try(qr.solve(qr.S))
			
			}
			
			if(inherits(inverse.new.S,"try-error")){

				cat("\n","LU and QR decompositions failed to invert penalty matrix, trying Cholesky","\n",
				"set detail.rho=TRUE for details","\n")
			
				inverse.new.S <- chol2inv(chol(sub.S))
			
			}
			
			temp.LAML <- lapply(1:nb.smooth,function(i) S.list[[i]][1:rank.S,1:rank.S])

			temp.LAML2 <- lapply(1:nb.smooth,function(i) -inverse.new.S%mult%temp.LAML[[i]]%mult%inverse.new.S)
		}
		

	#------------------------------- gradient of LAML and LCV
			# first derivatives of beta Hessian
			
			grad.list <- grad_rho(X.GL, GL.temp, haz.GL, deriv.rho.beta, leg$weights,
			tm, nb.smooth, p, n.legendre, S.list, temp.LAML, Vp, S.beta, beta, inverse.new.S,
			X, temp.deriv3, event, expected, type, Ve, mat.temp, method)
			
			grad.rho <- grad.list$grad_rho
			
			if (method == "LCV") grad.rho <- grad.rho + deriv.rho.beta%vec%(-grad.unpen.beta)
			
			deriv.rho.inv.Hess.beta <- grad.list$deriv_rho_inv_Hess_beta
			deriv.rho.Hess.unpen.beta <- grad.list$deriv_rho_Hess_unpen_beta
			
			
	#------------------------------- Hessian of LAML and LCV
	# Implicit second derivative of beta wrt rho
			deriv2.rho.beta <- lapply(1:nb.smooth, function(i) matrix(0,nrow=nb.smooth,ncol=p))

			for (j in 1:nb.smooth){

				for (j2 in 1:nb.smooth){

					deriv2.rho.beta[[j2]][j,] <- deriv.rho.inv.Hess.beta[[j2]]%vec%S.beta[[j]] - 
					Vp%vec%(S.list[[j]]%vec%deriv.rho.beta[j2,])

					if (j==j2){

					deriv2.rho.beta[[j2]][j,] <- deriv2.rho.beta[[j2]][j,] - Vp%mult%S.beta[[j2]]

					}

				}

			}

			
			if (method=="LCV"){

				# first part of the Hessian of LCV
				for (j2 in 1:nb.smooth){

					Hess.LCV1[,j2] <- deriv2.rho.beta[[j2]]%vec%(-grad.unpen.beta)+deriv.rho.beta%vec%(-Hess.unpen.beta%vec%deriv.rho.beta[j2,])

				}

				# this calculation is done before to save time

				deriv.rho.Ve <- lapply(1:nb.smooth, function(j2) -(  (deriv.rho.inv.Hess.beta[[j2]]%mult%Hess.unpen.beta -
				Vp%mult%deriv.rho.Hess.unpen.beta[[j2]] )%mult%(-Vp) -  Vp%mult%Hess.unpen.beta%mult%deriv.rho.inv.Hess.beta[[j2]])  )

				deriv.mat.temp <- lapply(1:nb.smooth, function(j2) deriv.rho.Ve[[j2]]+deriv.rho.inv.Hess.beta[[j2]] )

				# if we perform an eigen decomposition of mat.temp, we only need to compute the diagonal of 
				# the second derivatives of Hess.unpen.beta (see section 6.6.2 of Wood (2017) Generalized Additive Models
				# An introduction with R, Second Edition)
					eigen2 <- eigen(mat.temp,symmetric=TRUE)
					eigen.mat.temp <- eigen2$values
					
			}
			
			if(method=="LAML"){
			
			# if we perform an eigen decomposition of -inv.Hess.beta, we only need to compute the diagonal of 
			# the second derivatives of Hess.unpen.beta (see section 6.6.2 of Wood (2017) Generalized Additive Models
			# An introduction with R, Second Edition)
					eigen2 <- eigen(Vp,symmetric=TRUE)
					minus.eigen.inv.Hess.beta <- eigen2$values
					
			}	

			Q <- eigen2$vectors
					
			X.Q <- X%mult%Q
					
			X.GL.Q <- lapply(1:n.legendre, function(i) X.GL[[i]]%mult%Q)	
			#----------------------------------------------------------------------------------------
		
			Hess.rho <- Hess_rho(X.GL, X.GL.Q, GL.temp, haz.GL, deriv2.rho.beta, deriv.rho.beta, leg$weights,
			tm, nb.smooth, p, n.legendre, deriv.rho.inv.Hess.beta, deriv.rho.Hess.unpen.beta, S.list, minus.eigen.inv.Hess.beta,
			temp.LAML, temp.LAML2, Vp, S.beta, beta, inverse.new.S,
			X, X.Q, temp.deriv3, temp.deriv4, event, expected, type,
			Ve, deriv.rho.Ve, mat.temp, deriv.mat.temp, eigen.mat.temp, method)

		
			if(method=="LCV") Hess.rho <- Hess.rho + Hess.LCV1
	

	}else{
	
		grad.rho <- Hess.rho <- deriv.rho.beta <- deriv.rho.inv.Hess.beta <- NULL
	
	}
		
  }else{

	edf <- edf1 <- p

	LCV <- LAML <- criterion.val <- grad.rho <- Hess.rho <- deriv.rho.beta <- deriv.rho.inv.Hess.beta <- NULL

  }

	optim.rho <- iter.rho <- edf2 <- aic2 <- inv.Hess.rho <- Hess.rho.modif <- Vc.approx <- Vc <- NULL
	
	# returns a model of class survPen
	res <- list(call=cl,formula=formula,t0.name=build$t0.name,t1.name=build$t1.name,event.name=build$event.name,expected.name=build$expected.name,
	haz=ft1,coefficients=beta,type=type,df.para=df.para,df.smooth=df.smooth,p=p,edf=edf,edf1=edf1,edf2=edf2,aic=2*sum(edf)-2*ll.unpen,aic2=aic2,iter.beta=iter.beta,X=X,S=S,S.scale=S.scale,
	S.list=S.list,S.smf=S.smf,S.tensor=S.tensor,S.tint=S.tint,S.rd=S.rd,smooth.name.smf=smooth.name.smf,smooth.name.tensor=smooth.name.tensor,smooth.name.tint=smooth.name.tint,smooth.name.rd=smooth.name.rd,
	S.pen=build$S.pen,grad.unpen.beta=grad.unpen.beta,grad.beta=grad.beta,Hess.unpen.beta=Hess.unpen.beta,Hess.beta=-neg.Hess.beta,
	Hess.beta.modif=Hess.beta.modif,ll.unpen=ll.unpen,ll.pen=ll.pen,deriv.rho.beta=deriv.rho.beta,deriv.rho.inv.Hess.beta=deriv.rho.inv.Hess.beta,lambda=lambda,
	nb.smooth=nb.smooth,iter.rho=iter.rho,optim.rho=optim.rho,method=method,criterion.val=criterion.val,LCV=LCV,LAML=LAML,grad.rho=grad.rho,Hess.rho=Hess.rho,inv.Hess.rho=inv.Hess.rho,
	Hess.rho.modif=Hess.rho.modif,Ve=Ve,Vp=Vp,Vc=Vc,Vc.approx=Vc.approx,Z.smf=Z.smf,Z.tensor=Z.tensor,Z.tint=Z.tint,
	list.smf=list.smf,list.tensor=list.tensor,list.tint=list.tint,list.rd=list.rd,U.F=U.F)

	class(res) <- "survPen"
	res
}

#----------------------------------------------------------------------------------------------------------------
# END of code : survPen.fit
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# predict.survPen : makes hazard and survival predictions from a model for a new data frame
#----------------------------------------------------------------------------------------------------------------

#' Hazard and Survival prediction from fitted \code{survPen} model
#'
#' Takes a fitted \code{survPen} object and produces hazard and survival predictions given a new set of values for the model covariates.
#' @param object a fitted \code{survPen} object as produced by \code{\link{survPen.fit}}
#' @param newdata data frame giving the new covariates value
#' @param newdata.ref data frame giving the new covariates value for the reference population (used only when type="HR")
#' @param n.legendre number of nodes to approximate the cumulative hazard by Gauss-Legendre quadrature; default is 50
#' @param conf.int numeric value giving the precision of the confidence intervals; default is 0.95
#' @param do.surv If TRUE, the survival and its lower and upper confidence values are computed. Survival computation requires numerical integration and can be time-consuming so if you only want the hazard use do.surv=FALSE; default is TRUE
#' @param type, if type="lpmatrix" returns the design matrix (or linear predictor matrix) corresponding to the new values of the covariates; if equals "HR", returns the predicted HR and CIs between newdata and newdata.ref; default is "standard" for classical hazard and survival estimation
#' @param exclude.random if TRUE all random effects are set to zero; default is FALSE
#' @param get.deriv.H if TRUE, the derivatives wrt to the regression parameters of the cumulative hazard are returned; default is FALSE
#' @param ... other arguments
#' @details
#' The confidence intervals noted CI.U are built on the log cumulative hazard scale U=log(H) (efficient scale in terms of respect towards the normality assumption)
#' using Delta method. The confidence intervals on the survival scale are then \code{CI.surv = exp(-exp(CI.U))}
#' @return List of objects:
#' \item{haz}{hazard predicted by the model}
#' \item{haz.inf}{lower value for the confidence interval on the hazard based on the Bayesian covariance matrix Vp (Wood et al. 2016)}
#' \item{haz.sup}{Upper value for the confidence interval on the hazard based on the Bayesian covariance matrix Vp}
#' \item{surv}{survival predicted by the model}
#' \item{surv.inf}{lower value for the confidence interval on the survival based on the Bayesian covariance matrix Vp}
#' \item{surv.sup}{Upper value for the confidence interval on the survival based on the Bayesian covariance matrix Vp}
#' \item{deriv.H}{derivatives wrt to the regression parameters of the cumulative hazard. Useful to calculate standardized survival}
#' \item{HR}{predicted hazard ratio ; only when type = "HR"}
#' \item{HR.inf}{lower value for the confidence interval on the hazard ratio based on the Bayesian covariance matrix Vp  ; only when type = "HR"}
#' \item{HR.sup}{Upper value for the confidence interval on the hazard ratio based on the Bayesian covariance matrix Vp  ; only when type = "HR"}
#' @export
#'
#' @references
#' Wood, S.N., Pya, N. and Saefken, B. (2016), Smoothing parameter and model selection for general smooth models (with discussion). Journal of the American Statistical Association 111, 1548-1575
#'
#' @examples
#'
#' library(survPen)
#'
#' data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer
#'
#' # model : unidimensional penalized spline for time since diagnosis with 5 knots
#' f1 <- ~smf(fu,df=5)
#'
#' # hazard model
#' mod1 <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LAML")
#'
#' # predicting hazard and survival at time 1
#' pred <- predict(mod1,data.frame(fu=1))
#' pred$haz
#' pred$surv
#'
#' # predicting hazard ratio between age 70 and age 30
#' pred.HR <- predict(mod1,data.frame(fu=1,age=70),newdata.ref=data.frame(fu=1,age=30),type="HR")
#' pred.HR$HR
#' pred.HR$HR.inf
#' pred.HR$HR.sup
#'
predict.survPen <- function(object,newdata,newdata.ref=NULL,n.legendre=50,conf.int=0.95,do.surv=TRUE,type="standard",exclude.random=FALSE,get.deriv.H=FALSE,...){

	if (!inherits(object,"survPen")) stop("object is not of class survPen")
	
	# we apply the initial factor structure to newdata
	factor.structure <- object$factor.structure
	
	for (factor.name in names(newdata)[names(newdata)%in%names(factor.structure)]){

		newdata[,factor.name] <- factor(newdata[,factor.name],
		levels=factor.structure[[factor.name]]$levels,ordered=factor.structure[[factor.name]]$class[1]=="ordered")

	}
	#-------------------------------------------------------------

	qt.norm <- stats::qnorm(1-(1-conf.int)/2)

	t1 <- newdata[,object$t1.name]

	t0 <- rep(0,length(t1))

	tm <- (t1-t0)/2

	myMat <- design.matrix(object$formula,data.spec=newdata,Z.smf=object$Z.smf,Z.tensor=object$Z.tensor,Z.tint=object$Z.tint,list.smf=object$list.smf,list.tensor=object$list.tensor,list.tint=object$list.tint,list.rd=object$list.rd)

	if (type=="lpmatrix") return(myMat)
	
	# estimated regression parameters
	beta <- object$coefficients
	
	if (exclude.random){
	
		vec.name <- names(beta)

		# positions of regression parameters corresponding to random effects in regression parameters vector
		pos.rd <- which(substr(vec.name,1,3)=="rd(")

		# If there are random effects, we set the corresponding regression parameters to zero for prediction
		if (length(pos.rd)!=0){
		
			beta[pos.rd] <- 0
		
		}

	}
	
	
	if (type=="HR"){
	
		myMat.ref <- design.matrix(object$formula,data.spec=newdata.ref,Z.smf=object$Z.smf,Z.tensor=object$Z.tensor,Z.tint=object$Z.tint,list.smf=object$list.smf,list.tensor=object$list.tensor,list.tint=object$list.tint,list.rd=object$list.rd)

		X <- myMat - myMat.ref
		
		log.haz.ratio <- as.vector(X%vec%beta)
		
		haz.ratio <- exp(log.haz.ratio)
		
		# Confidence intervals
		if (!is.null(object$Vp)){
		
			std<-sqrt(rowSums((X%mult%object$Vp)*X))
			haz.ratio.inf <- as.vector(exp(log.haz.ratio-qt.norm*std))
			haz.ratio.sup <- as.vector(exp(log.haz.ratio+qt.norm*std))
		
		}else{
		
			haz.ratio.inf <- NULL
			haz.ratio.sup <- NULL
		
		}
		
		return(list(HR=haz.ratio,HR.inf=haz.ratio.inf,HR.sup=haz.ratio.sup))
		
	}

	
	
	# estimated linear predictor
	pred.haz <- myMat%vec%beta

	# estimated (excess) hazard
	haz <- exp(pred.haz)
	
	if (do.surv){ # Gauss Legendre quadrature for hazard integration and survival estimation
		
		leg <- statmod::gauss.quad(n=n.legendre,kind="legendre")
		
		# Design matrices for Gauss-Legendre quadrature
		X.func <- function(t1,data,object){

			data.t <- data
			data.t[,object$t1.name] <- t1
			design.matrix(object$formula,data.spec=data.t,Z.smf=object$Z.smf,Z.tensor=object$Z.tensor,Z.tint=object$Z.tint,list.smf=object$list.smf,list.tensor=object$list.tensor,list.tint=object$list.tint,list.rd=object$list.rd)

		}

		X.GL <- lapply(1:n.legendre, function(i) X.func(tm*leg$nodes[i]+(t0+t1)/2,newdata,object))

		cumul.haz <- lapply(1:n.legendre, function(i) (exp((X.GL[[i]]%vec%beta)))*leg$weights[i])

		cumul.haz <- tm*Reduce("+",cumul.haz)

		surv=exp(-cumul.haz)

	}else{
	
		surv=NULL
	
	}

	if (!is.null(object$Vp)){ # only the Bayesian covariance matrix Vp is used for confidence intervals

		# confidence intervals for hazard
		std <- sqrt(rowSums((myMat%mult%object$Vp)*myMat))
		haz.inf <- as.vector(exp(pred.haz-qt.norm*std))
		haz.sup <- as.vector(exp(pred.haz+qt.norm*std))

		if (do.surv){
			# if any cumul hazard is zero, we put it at a very low positive value
			cumul.haz[cumul.haz==0] <- 1e-16

			deriv.cumul.haz <- lapply(1:n.legendre, function(i) X.GL[[i]]*(exp((X.GL[[i]]%vec%beta)))*leg$weights[i])

			deriv.cumul.haz <- tm*Reduce("+",deriv.cumul.haz)

			log.cumul.haz <- log(cumul.haz)

			deriv.log.cumul.haz <- deriv.cumul.haz/cumul.haz

			#---------------------------
			# Delta method
			std.log.cumul <- sqrt(rowSums((deriv.log.cumul.haz%mult%object$Vp)*deriv.log.cumul.haz))

			surv.inf=exp(-exp(log.cumul.haz+qt.norm*std.log.cumul))
			surv.sup=exp(-exp(log.cumul.haz-qt.norm*std.log.cumul))
			
		}else{
		
			surv.inf=NULL
			surv.sup=NULL
			deriv.cumul.haz=NULL
		
		}

	}else{

		haz.inf=NULL
		haz.sup=NULL
		surv.inf=NULL
		surv.sup=NULL

	}
	
	if (!get.deriv.H) deriv.cumul.haz <- NULL
	
	res<-list(haz=haz,haz.inf=haz.inf,haz.sup=haz.sup,
	     surv=surv,surv.inf=surv.inf,surv.sup=surv.sup,deriv.H=deriv.cumul.haz)

	class(res) <- "predict.survPen"
	
	res	 
}

#----------------------------------------------------------------------------------------------------------------
# END of code : predict.survPen
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# summary.survPen
#----------------------------------------------------------------------------------------------------------------

#' Summary for a \code{survPen} fit
#'
#' Takes a fitted \code{survPen} object and produces various useful summaries from it.
#' @param object a fitted \code{survPen} object as produced by \code{\link{survPen.fit}}
#' @param ... other arguments
#' @return List of objects:
#' \item{call}{the original survPen call}
#' \item{formula}{the original survPen formula}
#' \item{coefficients}{reports the regression parameters estimates for unpenalized terms with the associated standard errors}
#' \item{edf.per.smooth}{reports the edf associated with each smooth term}
#' \item{random}{TRUE if there are random effects in the model}
#' \item{random.effects}{reports the estimates of the log standard deviation (log(sd)) of every random effects plus the estimated standard error (also on the log(sd) scale)}
#' \item{likelihood}{unpenalized likelihood of the model}
#' \item{penalized.likelihood}{penalized likelihood of the model}
#' \item{nb.smooth}{number of smoothing parameters}
#' \item{smoothing.parameter}{smoothing parameters estimates}
#' \item{parameters}{number of regression parameters}
#' \item{edf}{effective degrees of freedom}
#' \item{method}{smoothing selection criterion used (LAML or LCV)}
#' \item{val.criterion}{minimized value of criterion. For LAML, what is reported is the negative log marginal likelihood}
#' \item{converged}{convergence indicator, TRUE or FALSE. TRUE if Hess.beta.modif=FALSE and Hess.rho.modif=FALSE (or NULL)}
#' @export
#'
#' @examples
#'
#' library(survPen)
#'
#' data(datCancer) # simulated dataset with 2000 individuals diagnosed with cervical cancer
#'
#' # model : unidimensional penalized spline for time since diagnosis with 5 knots
#' f1 <- ~smf(fu,df=5)
#'
#' # fitting hazard model
#' mod1 <- survPen(f1,data=datCancer,t1=fu,event=dead,expected=NULL,method="LAML")
#'
#' # summary
#' summary(mod1)
#'
summary.survPen <- function(object,...){

	if (!inherits(object,"survPen")) stop("object is not of class survPen")

	if (object$nb.smooth==0){
		
		if(object$type=="net"){
			
			type <- "excess hazard model"
		
		}else{
		
			type <- "hazard model"
		
		}
		
		SE.rho <- NULL
		TAB.random <- NULL
		random <- FALSE
		edf.per.smooth <- NULL
		
	}else{
	
		if(object$type=="net"){
				
				type <- "penalized excess hazard model"
			
		}else{
			
				type <- "penalized hazard model"
			
		}
		

		# effective degrees of freedom of smooth terms
		edf.smooth <- object$edf[(object$df.para+1):length(object$edf)]
		
		name.edf <- names(edf.smooth)
		
		list.name <- sapply(1:length(name.edf),function(i) substr(name.edf[i],1,instr(name.edf[i],"\\.")-1))
		
		list.name <- factor(list.name,levels=unique(list.name)) # to preserve the order in names
		
		edf.per.smooth <- tapply(edf.smooth, list.name, sum)


		if (is.null(object$optim.rho)){
		
			SE.rho <- NULL
			TAB.random <- NULL
			random <- FALSE
		
		}else{
	
			# standard errors for smoothing parameters
			if (object$nb.smooth==1){
		
				SE.rho <- sqrt(object$inv.Hess.rho)
		
			}else{
		
				SE.rho <- sqrt(diag(object$inv.Hess.rho))
		
			}
	
			# Are there any random effects ?
			random <- any(substr(names(object$lambda),1,2)=="rd")
	
			if (random){
			
				TAB.random <- cbind(Estimate = -0.5*log(object$lambda)-0.5*log(object$S.scale),`Std. Error` = 0.5*SE.rho)
		
				colnames(TAB.random) <- c("Estimate","Std. Error")
			
				TAB.random <- TAB.random[substr(rownames(TAB.random),1,2)=="rd",,drop=FALSE]
			
			}else{
			
				TAB.random <- NULL
			
			}
			
			
		}

	}
	
	# standard errors
	if (object$p==1){
	
		SE <- sqrt(object$Vp)
	
	}else{
	
		SE <- sqrt(diag(object$Vp))
	
	}
	
	# number of parameters associated with fully parametric terms
	len <- object$df.para
	
	zvalue <- object$coefficients[1:len]/SE[1:len]
	pvalue <- 2 * stats::pnorm(-abs(zvalue))
	
	TAB <- cbind(Estimate = object$coefficients[1:len], `Std. Error` = SE[1:len], 
        `z value` = zvalue, `Pr(>|z|)` = pvalue)
	
	attrs <- attributes(object$lambda)
	
	res <- list(type = type,
			call=object$call,
			formula=object$formula,
			coefficients=TAB,
			edf.per.smooth=edf.per.smooth,
			random=random,
			random.effects=TAB.random,
			likelihood = object$ll.unpen,
			penalized.likelihood = object$ll.pen,
			nb.smooth = object$nb.smooth,
			smoothing.parameter = object$lambda,
			parameters = object$p,
			edf = sum(object$edf),
			method = object$method,
			criterion.val = object$criterion.val,
			converged = object$converged)
	
	attributes(res$smoothing.parameter) <- attrs
	
	res <- c(res)
	
	class(res) <- "summary.survPen"
	
	res
	
}

#----------------------------------------------------------------------------------------------------------------
# END of code : summary.survPen
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# print.summary.survPen
#----------------------------------------------------------------------------------------------------------------

#' print summary for a \code{survPen} fit
#'
#' @param x an object of class \code{summary.survPen}
#' @param digits controls number of digits printed in output.
#' @param signif.stars Should significance stars be printed alongside output.
#' @param ... other arguments
#' @return print of summary
#' @export
#'
print.summary.survPen <- function(x, digits = max(3, getOption("digits") - 2), 
    signif.stars = getOption("show.signif.stars"), ...)
{
	cat(paste(noquote(x$type),"\n","\n"))

	cat("Call:\n")
	print(x$call)
	cat("\nParametric coefficients:\n")
	stats::printCoefmat(x$coefficients, P.value=TRUE, has.Pvalue=TRUE, digits = digits, signif.stars = signif.stars, na.print = "NA", ...)

	if (x$random){

		cat("\nRandom effects (log(sd)):\n")
		print(x$random.effects)

	}
	
	if (substr(x$type,1,9)=="penalized"){

		cat("\n")
		cat(paste0("log-likelihood = ",signif(x$likelihood,digits),","," penalized log-likelihood = ",signif(x$penalized.likelihood,digits)))

		cat("\n")
		cat(paste0("Number of parameters = ",x$parameters,","," effective degrees of freedom = ",signif(x$edf,digits)))

		cat("\n")
		cat(paste(x$method,"=",signif(x$criterion.val,digits)),"\n","\n")
		
		cat("Smoothing parameter(s):\n")
		print(signif(x$smoothing.parameter,digits))
		
		cat("\n")
		cat("edf of smooth terms:\n")
		print(signif(x$edf.per.smooth,digits))

	}else{
	
		cat("\n")
		cat(paste("likelihood =",signif(x$likelihood,digits)))

		cat("\n")
		cat(paste("Number of parameters =",x$parameters))

	}

	cat("\n")
	cat(paste("converged=",x$converged))

	invisible(x)
	cat("\n")
}

#----------------------------------------------------------------------------------------------------------------
# END of code : print.summary.survPen
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# NR.beta : Newton-Raphson algotihm for regression beta estimation
#----------------------------------------------------------------------------------------------------------------

#' Inner Newton-Raphson algorithm for regression parameters estimation
#'
#' Applies Newton-Raphson algorithm for beta estimation. Two specific modifications aims at guaranteeing
#' convergence : first the hessian is perturbed whenever it is not positive definite and second, at each step, if the penalized
#' log-likelihood is not maximized, the step is halved until it is.
#' @param build list of objects returned by \code{\link{model.cons}}
#' @param beta.ini vector of initial regression parameters; default is NULL, in which case the first beta will be \code{log(sum(event)/sum(t1))} and the others will be zero (except if there are "by" variables in which case all betas are set to zero)
#' @param detail.beta if TRUE, details concerning the optimization process in the regression parameters are displayed; default is FALSE
#' @param max.it.beta maximum number of iterations to reach convergence in the regression parameters; default is 200
#' @param tol.beta convergence tolerance for regression parameters; default is \code{1e-04}
#'
#' @details
#' If we note \code{ll.pen} and \code{beta} respectively the current penalized log-likelihood and estimated parameters and
#' \code{ll.pen.old} and \code{betaold} the previous ones, the algorithm goes on while
#' (abs(ll.pen-ll.pen.old)>tol.beta) or any(abs((beta-betaold)/betaold)>tol.beta)
#'
#' @return List of objects:
#' \item{beta}{estimated regression parameters}
#' \item{ll.unpen}{log-likelihood at convergence}
#' \item{ll.pen}{penalized log-likelihood at convergence}
#' \item{haz.GL}{list of all the matrix-vector multiplications X.GL[[i]]\%*\%beta for Gauss Legendre integration. Useful to avoid repeating operations in \code{\link{survPen.fit}}}
#' \item{iter.beta}{number of iterations needed to converge}
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 4 knots
#'
#' data <- data.frame(time=seq(0,5,length=100),event=1,t0=0)
#' 
#' form <- ~ smf(time,knots=c(0,1,3,5))
#' 
#' t1 <- eval(substitute(time), data)
#' t0 <- eval(substitute(t0), data)
#' event <- eval(substitute(event), data)
#' 	
#' # Setting up the model before fitting
#' model.c <- model.cons(form,lambda=0,data.spec=data,t1=t1,t1.name="time",
#' t0=rep(0,100),t0.name="t0",event=event,event.name="event",
#' expected=NULL,expected.name=NULL,type="overall",n.legendre=20,
#' cl="survPen(form,data,t1=time,event=event)",beta.ini=NULL)
#'  
#' # Estimating the regression parameters at given smoothing parameter (here lambda=0)
#' Newton1 <- NR.beta(model.c,beta.ini=rep(0,4),detail.beta=TRUE)
#'
NR.beta <- function(build,beta.ini,detail.beta,max.it.beta=200,tol.beta=1e-04){

	# get all the build elements
	type <- build$type # net or overall 
	X <- build$X # design matrix

	X.GL <- build$X.GL # list of Gauss-Legendre design matrices
	
	event <- build$event # censoring indicators
	expected <- build$expected # expected mortality rates
	
	leg <- build$leg # weights and nodes for Gauss-Legendre quadrature
	n.legendre <- build$n.legendre # number of nodes for Gauss-Legendre quadrature
	t1 <- build$t1 # time-to-event vector
	t0 <- build$t0 # left-truncature vector
	tm <- build$tm # mean of t0 and t1
	S <- build$S # final penalty matrix
	p <- build$p # number of regression parameters

	k=1
	ll.pen=100
	ll.pen.old=1

	if (length(beta.ini)==1) beta.ini <- rep(beta.ini,p)
	
	if (length(beta.ini)!=p) stop("message NR.beta: the length of beta.ini does not equal the number of regression parameters") 
	
	betaold <- beta.ini
	beta1 <- betaold

	if (detail.beta){
  
	cat("---------------------------------------------------------------------------------------","\n",
	"Beginning regression parameter estimation","\n","\n")
	
    }
	# beginning of the while loop
	while(abs(ll.pen-ll.pen.old)>tol.beta|any(abs((beta1-betaold)/betaold)>tol.beta))
	{

		if(k > max.it.beta)
		{
			stop("message NR.beta: Ran out of iterations (", k, "), and did not converge ")

		}

		if(k>=2)
		{
			ll.pen.old <- ll.pen
			betaold <- beta1
		}

		# hazard
		predold=X%vec%betaold
		
		ftold=exp(predold)

		# first derivatives of the cumulative hazard
		haz.GL.old <- lapply(1:n.legendre, function(i) exp(X.GL[[i]]%vec%betaold))
		
		deriv.list <- lapply(1:n.legendre, function(i) X.GL[[i]]*haz.GL.old[[i]]*leg$weights[i]*tm)

		f.first <- Reduce("+",deriv.list)
		
		# log-likelihoods gradients
		if (type=="net"){
			grad.unpen.beta <- colSums2(-f.first + (event*X*ftold)/(ftold+expected))
		}else{
			grad.unpen.beta <- colSums2(-f.first + event*X)
		}

		grad <- grad.unpen.beta-S%vec%betaold
		
		# second derivatives of the cumulative hazard
		deriv.2.list <- lapply(1:n.legendre, function(i) X.GL[[i]]%cross%(deriv.list[[i]]))
		
		f.second <- Reduce("+",deriv.2.list)
        
		# log-likelihoods Hessians
		if (type=="net"){
			Hess.unpen <- -f.second + X%cross%(event*X*expected*ftold/(ftold+expected)^2)
    	}else{
			Hess.unpen <- -f.second
		}
		
		Hess <- Hess.unpen-S
	
		# negative Hessian of penalized log-likelihood
		neg.Hess <- -Hess

		R <- try(chol(neg.Hess),silent=TRUE)
		
		# Hessian perturbation if need be (see Nocedal and Wright 2006)
		if(inherits(R,"try-error"))
		{
			u=0.001
			cpt.while <- 0
			while(inherits(R,"try-error"))
			{
				if(cpt.while > 100)
				{
				
					stop("message NR.beta: did not succeed in inverting Hessian at iteration ", k)
					
				}

				R <- try(chol(neg.Hess+u*diag(p)),silent=TRUE)

				u <- 5*u

				cpt.while <- cpt.while+1

			}

			if (detail.beta) {cat("beta Hessian perturbation, ", cpt.while, "iterations","\n","\n")}
		}
		
		Vp <- chol2inv(R)


		# cumulative hazard
		integral <- lapply(1:n.legendre, function(i) haz.GL.old[[i]]*leg$weights[i])

		integral <- tm*Reduce("+",integral)

		# log-likelihoods
		if (type=="net"){
			ll.unpenold <- sum(-integral + event*log(ftold+expected))
		}else{
			ll.unpenold <- sum(-integral + event*predold)
		}

		ll.pen.old <- ll.unpenold-0.5*sum(betaold*(S%vec%betaold))

		if (is.nan(ll.pen.old)) stop("message NR.beta: convergence issues, cannot evaluate log-likelihood")

		# New set of parameters
		pas <- Vp%vec%grad
		
		beta1 <- betaold+pas

		# New hazard
		pred1 <- X%vec%beta1
		ft1=exp(pred1)
		
		# New cumulative hazard
		# haz.GL will serve in survPen.fit to derive the derivatives with respect to the smoothing parameters
		haz.GL <- lapply(1:n.legendre, function(i) exp(X.GL[[i]]%vec%beta1))

		integral <- lapply(1:n.legendre, function(i) haz.GL[[i]]*leg$weights[i])
	
		integral <- tm*Reduce("+",integral)
		
		# New log-likelihoods
		if (type=="net"){
			ll.unpen <- sum(-integral + event*log(ft1+expected))
		}else{
			ll.unpen <- sum(-integral + event*pred1)
		}
		
		ll.pen <- ll.unpen - 0.5*sum(beta1*(S%vec%beta1))

		if (is.nan(ll.pen)) {ll.pen <- ll.pen.old - 1}
		
		if (ll.pen < ll.pen.old - 1e-03){ # at each step, the current log-likelihood should not be inferior to
		# the previous one with a certain tolerence (1e-03)
			cpt.beta <- 1
			# if the penalized log-likelihood is not maximized, the step is halved until it is
			while (ll.pen < ll.pen.old - 1e-03){

				if(cpt.beta>52) stop("message NR.beta: step has been divided by two 52 times in a row, Log-likelihood could not be optimized")
				# we use 52 because 2^(-52) is machine precision
				cpt.beta <- cpt.beta + 1
				
				pas <- 0.5*pas

				beta1 <- betaold+pas

				# New hazard
				pred1 <- X%vec%beta1
				ft1=exp(pred1)

				# New cumulative hazard
				haz.GL <- lapply(1:n.legendre, function(i) exp(X.GL[[i]]%vec%beta1))

				integral <- lapply(1:n.legendre, function(i) haz.GL[[i]]*leg$weights[i])

				integral <- tm*Reduce("+",integral)

				# New log-likelihoods
				if (type=="net"){
					ll.unpen <- sum(-integral + event*log(ft1+expected))
				}else{
					ll.unpen <- sum(-integral + event*pred1)
				}

				ll.pen <- ll.unpen - 0.5*sum(beta1*(S%vec%beta1))

				if (is.nan(ll.pen)) {ll.pen <- ll.pen.old - 1}
			}

		}
	
		# convergence details
		if (detail.beta){
		  cat("iter beta: ",k,"\n",
			  "betaold= ", round(betaold,4),"\n",
			  "beta= ", round(beta1,4),"\n",
			  "abs((beta-betaold)/betaold)= ", round(abs((beta1-betaold)/betaold),5),"\n",
			  "ll.pen.old= ", round(ll.pen.old,4),"\n",
			  "ll.pen= ", round(ll.pen,4),"\n",
			  "ll.pen-ll.pen.old= ", round(ll.pen-ll.pen.old,5),"\n",
			  "\n"
		  )
		}

		# next iteration
		k=k+1

	}

	if (detail.beta) {

		cat("\n",
		"Beta optimization ok, ", k-1, "iterations","\n",
		"--------------------------------------------------------------------------------------","\n")

	}

	list(beta=beta1,ll.unpen=ll.unpen,ll.pen=ll.pen,haz.GL=haz.GL,iter.beta=k-1)

}

#----------------------------------------------------------------------------------------------------------------
# END of code : NR.beta
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# NR.rho : Newton-Raphson algotihm for smoothing parameter estimation via LCV or LAML optimization
#----------------------------------------------------------------------------------------------------------------

#' Outer Newton-Raphson algorithm for smoothing parameters estimation via LCV or LAML optimization
#'
#' Applies Newton-Raphson algorithm for smoothing parameters estimation. Two specific modifications aims at guaranteeing
#' convergence : first the hessian is perturbed whenever it is not positive definite and second, at each step, if LCV or -LAML
#' is not minimized, the step is halved until it is.
#' @param build list of objects returned by \code{\link{model.cons}}
#' @param rho.ini vector of initial log smoothing parameters; if it is NULL, all log lambda are set to -1
#' @param data an optional data frame containing the variables in the model
#' @param formula formula object specifying the model
#' @param max.it.beta maximum number of iterations to reach convergence in the regression parameters; default is 200
#' @param max.it.rho maximum number of iterations to reach convergence in the smoothing parameters; default is 30
#' @param beta.ini vector of initial regression parameters; default is NULL, in which case the first beta will be \code{log(sum(event)/sum(t1))} and the others will be zero (except if there are "by" variables in which case all betas are set to zero)
#' @param detail.rho if TRUE, details concerning the optimization process in the smoothing parameters are displayed; default is FALSE
#' @param detail.beta if TRUE, details concerning the optimization process in the regression parameters are displayed; default is FALSE
#' @param nb.smooth number of smoothing parameters
#' @param tol.beta convergence tolerance for regression parameters; default is \code{1e-04}
#' @param tol.rho convergence tolerance for smoothing parameters; default is \code{1e-04}
#' @param step.max maximum absolute value possible for any component of the step vector (on the log smoothing parameter scale); default is 5
#' @param method LCV or LAML; default is LAML
#' @details
#' If we note \code{val} the current LCV or LAML value,
#' \code{val.old} the previous one and \code{grad} the gradient vector of LCV or LAML with respect to the log smoothing parameters, the algorithm goes on
#' \code{while(abs(val-val.old)>tol.rho|any(abs(grad)>tol.rho))}
#'
#' @return object of class survPen (see \code{\link{survPen.fit}} for details)
#' @export
#'
#' @examples
#' 
#' library(survPen)
#'
#' # standard spline of time with 4 knots
#'
#' data <- data.frame(time=seq(0,5,length=100),event=1,t0=0)
#' 
#' form <- ~ smf(time,knots=c(0,1,3,5))
#' 
#' t1 <- eval(substitute(time), data)
#' t0 <- eval(substitute(t0), data)
#' event <- eval(substitute(event), data)
#' 	
#' # Setting up the model before fitting
#' model.c <- model.cons(form,lambda=0,data.spec=data,t1=t1,t1.name="time",
#' t0=rep(0,100),t0.name="t0",event=event,event.name="event",
#' expected=0,expected.name=NULL,type="overall",n.legendre=20,
#' cl="survPen(form,data,t1=time,event=event)",beta.ini=NULL)
#'  
#' # Estimating the smoothing parameter and the regression parameters
#' # we need to apply a reparameterization to model.c before fitting
#' Newton2 <- NR.rho(repam(model.c)$build,rho.ini=-1,data,form,nb.smooth=1,detail.rho=TRUE)
#'
NR.rho <- function(build,rho.ini,data,formula,max.it.beta=200,max.it.rho=30,beta.ini=NULL,detail.rho=FALSE,detail.beta=FALSE,nb.smooth,tol.beta=1e-04,tol.rho=1e-04,step.max=5,method="LAML"){

  df.tot <- build$df.tot
 
  iter.beta <- NULL

  k.rho=1
  val=1
  val.old=100

  rho <- rho.ini
  rho.old <- rho.ini

  grad <- rep(1,length(rho))

  if (detail.rho){
  
	cat(
	"_______________________________________________________________________________________","\n","\n",
	"Beginning smoothing parameter estimation via ",method," optimization","\n",
    "______________________________________________________________________________________","\n","\n")
	
  }

  while(abs(val-val.old)>tol.rho|any(abs(grad)>tol.rho))
  {
    if(k.rho > max.it.rho)
    {
      stop("message NR.rho: Ran out of iterations (", k.rho, "), and did not converge ")

    }

    if(k.rho>=2)
    {
      val.old <- val
      rho.old <- rho
    }

	if(k.rho==1)
	{

	  lambda=exp(rho.old)

	  name.lambda <- names(build$lambda)
	  build$lambda <- lambda
	  names(build$lambda) <- name.lambda
	  
	  build$S <- matrix(0,df.tot,df.tot)

	  for (i in 1:nb.smooth){

	    build$S.list[[i]] <- lambda[i]*build$S.pen[[i]]

	    build$S <- build$S+build$S.list[[i]]

	  }

	  if (detail.rho){
  
		cat(
		"--------------------","\n",
		" Initial calculation","\n",
		"-------------------","\n","\n"
		)
	
	  }
	
	  model <- survPen.fit(build,data=data,formula=formula,max.it.beta=max.it.beta,beta.ini=beta.ini,detail.beta=detail.beta,method=method,tol.beta=tol.beta)
	  beta1 <- model$coefficients
	  iter.beta <- c(iter.beta,model$iter.beta)
	}
	
	# LAML
	val.old=model$criterion.val

    # gradient
	grad <- model$grad.rho

	# Hessian
	Hess <- model$Hess.rho

	R <- try(chol(Hess),silent=TRUE)

	# Hessian perturbation
	if(inherits(R,"try-error"))
	{

		u=0.001
		cpt.while <- 0
		while(inherits(R,"try-error"))
		{

			if(cpt.while > 100)
			{
				stop("message NR.rho : did not succeed in inverting Hessian at iteration ", k.rho)
				
			}

			R <- try(chol(Hess+u*diag(nb.smooth)),silent=TRUE)

			u <- 5*u

			cpt.while <- cpt.while+1

		}

		if (detail.rho) {cat(method," Hessian perturbation, ", cpt.while, "iterations","\n","\n")}
	}

	inv.Hess <- chol2inv(R)

	# New rho
	pas <- -inv.Hess%vec%grad
	
	norm.pas <- max(abs(pas))

	if (norm.pas>step.max){
		
		if (detail.rho) {
		
			cat("\n","\n","new step = ", signif(pas,3),"\n")
		
		}
		
		pas <- (step.max/norm.pas)*pas

		if (detail.rho) {
		
			cat("new step corrected = ", signif(pas,3),"\n","\n")
		
		}
		
	}

	rho <- rho.old+pas

	lambda=exp(rho)
	
	name.lambda <- names(build$lambda)
	build$lambda <- lambda
	names(build$lambda) <- name.lambda
	
	build$S <- matrix(0,df.tot,df.tot)

	for (i in 1:nb.smooth){

	  build$S.list[[i]] <- lambda[i]*build$S.pen[[i]]

	  build$S <- build$S+build$S.list[[i]]

	}

	
	if (detail.rho){
  
		cat(
		"\n","Smoothing parameter selection, iteration ",k.rho,"\n","\n"
		)
	
	}
	
	model <- survPen.fit(build,data=data,formula=formula,max.it.beta=max.it.beta,beta.ini=beta1,detail.beta=detail.beta,method=method,tol.beta=tol.beta)
	beta1 <- model$coefficients

	val <- model$criterion.val

	if (is.nan(val)) {val <- val.old+1}

	if (val>val.old+1e-03){

	  cpt.rho <- 1

		while (val>val.old+1e-03){

			if (detail.rho) {

			cat("---------------------------------------------------------------------------------------","\n",
			"val= ", val," et val.old= ", val.old,"\n",
			method," is not optimized at iteration ", k.rho,"\n",
			"Step is divided by 10","\n",
		    "--------------------------------------------------------------------------------------","\n","\n")

			}

			if(cpt.rho>16) stop("message NR.rho: step has been divided by ten 16 times in a row, ",method," could not be optimized")
			# we use 16 because 10^(-16) is near machine precision
		  	cpt.rho <- cpt.rho+1

			pas <- 0.1*pas

			rho <- rho.old+pas

			lambda=exp(rho)

			name.lambda <- names(build$lambda)
			build$lambda <- lambda
			names(build$lambda) <- name.lambda
			
			build$S <- matrix(0,df.tot,df.tot)

			for (i in 1:nb.smooth){

			  build$S.list[[i]] <- lambda[i]*build$S.pen[[i]]

			  build$S <- build$S+build$S.list[[i]]

			}

			model <- survPen.fit(build,data=data,formula=formula,max.it.beta=max.it.beta,beta.ini=beta1,detail.beta=detail.beta,method=method,tol.beta=tol.beta)
			beta1 <- model$coefficients

			val <- model$criterion.val

			if (is.nan(val)) {val <- val.old+1}

		}

	}

	iter.beta <- c(iter.beta,model$iter.beta)

	beta1 <- model$coefficients
	val <- model$criterion.val
	grad <- model$grad.rho
	Hess <- model$Hess.rho

	# convergence details
    if (detail.rho){
      cat("_______________________________________________________________________________________","\n",
		  "\n","iter ",method,": ",k.rho,"\n",
          "rho.old= ", round(rho.old,4),"\n",
          "rho= ", round(rho,4),"\n",
		  "val.old= ", round(val.old,4),"\n",
		  "val= ", round(val,4),"\n",
          "val-val.old= ", round(val-val.old,5),"\n",
		  "gradient= ", signif(grad,2),"\n",
          "\n"
      )
	  cat("_______________________________________________________________________________________","\n","\n","\n","\n")
    }

	# next iteration
    k.rho=k.rho+1

  }

	if (detail.rho) {

	cat("Smoothing parameter(s) selection via ",method," ok, ", k.rho-1, "iterations","\n",
    "______________________________________________________________________________________","\n")

	}

	Hess.rho.modif <- FALSE

	R <- try(chol(Hess),silent=TRUE)

	# Hessian perturbation at convergence
	if(inherits(R,"try-error"))
	{
		Hess.rho.modif <- TRUE
		
		eigen.temp <- eigen(Hess,symmetric=TRUE)
		U.temp <- eigen.temp$vectors
		vp.temp <- eigen.temp$values

		vp.temp[which(vp.temp<1e-7)] <- 1e-7

		R <- try(chol(U.temp%mult%diag(vp.temp)%mult%t(U.temp)),silent=TRUE)

		warning("message NR.rho: rho Hessian was perturbed at convergence")
	}

	model$inv.Hess.rho <- chol2inv(R)
	
	model$Hess.rho <- crossprod(R)
	
	model$Hess.rho.modif <- Hess.rho.modif

	model$iter.rho <- k.rho-1

	model$iter.beta <- iter.beta

	model$optim.rho <- 1 # so we can identify whether the smoothing parameters were estimated or not
	
	model

}

#----------------------------------------------------------------------------------------------------------------
# END of code : NR.rho
#----------------------------------------------------------------------------------------------------------------









#################################################################################################################

