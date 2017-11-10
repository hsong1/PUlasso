#' PUlasso : An efficient algorithm to solve Positive and Unlabelled(PU) problem with lasso or group lasso penalty
#' @description The package efficiently solves PU problem in low or high dimensional setting using Maximization-Minorization and (block) coordinate descent. It allows simultaneous feature selection and parameter estimation for classification. Sparse calculation and parallel computing via OpenMP are supported for the further computational speed-up.
#' @details
#' Main functions: grpPUlasso, cv.grpPUlasso, coef, predict
#' @author Hyebin Song, \email{hsong@@stat.wisc.edu}
#' @keywords PUlearning, Lasso, Group Lasso
#' @examples 
#' data("simulatedPUdata")
#' attach(simulatedPUdata)
#' fit<-grpPUlasso(X=X,z=z,pi=truePrevalence)
#' cvfit<-cv.grpPUlasso(X=X,z=z,pi=truePrevalence)
#' coef(fit)
#' predict(fit,newdata = head(X),type = "response")
#' predict(cvfit,newdata = head(X), lambda=cvfit$lambda.1se,type = "response")
"_PACKAGE"