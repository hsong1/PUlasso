% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/grpPUlasso.R
\name{grpPUlasso}
\alias{grpPUlasso}
\title{Solve PU problem with lasso or group lasso penalty.}
\usage{
grpPUlasso(
  X,
  z,
  py1,
  initial_coef = NULL,
  group = 1:ncol(X),
  penalty = NULL,
  lambda = NULL,
  nlambda = 100,
  lambdaMinRatio = ifelse(N < p, 0.05, 0.005),
  maxit = ifelse(method == "CD", 1000, N * 10),
  maxit_inner = 1e+05,
  weights = NULL,
  eps = 1e-04,
  inner_eps = 0.01,
  verbose = FALSE,
  stepSize = NULL,
  stepSizeAdjustment = NULL,
  batchSize = 1,
  updateFrequency = N,
  samplingProbabilities = NULL,
  method = c("CD", "GD", "SGD", "SVRG", "SAG"),
  trace = c("none", "param", "fVal", "all")
)
}
\arguments{
\item{X}{Input matrix; each row is an observation. Can be a matrix or a sparse matrix.}

\item{z}{Response vector representing whether an observation is labeled or unlabeled.}

\item{py1}{True prevalence Pr(Y=1)}

\item{initial_coef}{A vector representing an initial point where we start PUlasso algorithm from.}

\item{group}{A vector representing grouping of the coefficients. For the least ambiguity, it is recommended if group is provided in the form of vector of consecutive ascending integers.}

\item{penalty}{penalty to be applied to the model. Default is sqrt(group size) for each of the group.}

\item{lambda}{A user supplied sequence of lambda values. If unspecified, the function automatically generates its own lambda sequence based on nlambda and lambdaMinRatio.}

\item{nlambda}{The number of lambda values.}

\item{lambdaMinRatio}{Smallest value for lambda, as a fraction of lambda.max which leads to the intercept only model.}

\item{maxit}{Maximum number of iterations.}

\item{maxit_inner}{Maximum number of iterations for a quadratic sub-problem for CD.}

\item{weights}{observation weights. Default is 1 for each observation.}

\item{eps}{Convergence threshold for the outer loop. The algorithm iterates until the maximum change in coefficients is less than eps in the outer loop.}

\item{inner_eps}{Convergence threshold for the inner loop. The algorithm iterates until the maximum change in coefficients is less than eps in the inner loop.}

\item{verbose}{A logical value. if TRUE, the function prints out the fitting process.}

\item{stepSize}{A step size for gradient-based optimization. if NULL, a step size is taken to be stepSizeAdj/mean(Li) where Li is a Lipschitz constant for ith sample}

\item{stepSizeAdjustment}{A step size adjustment. By default, adjustment is 1 for GD and SGD, 1/8 for SVRG and 1/16 for SAG.}

\item{batchSize}{A batch size. Default is 1.}

\item{updateFrequency}{An update frequency of full gradient for method =="SVRG"}

\item{samplingProbabilities}{sampling probabilities for each of samples for stochastic gradient-based optimization. if NULL, each sample is chosen proportionally to Li.}

\item{method}{Optimization method. Default is Coordinate Descent. CD for Coordinate Descent, GD for Gradient Descent, SGD for Stochastic Gradient Descent, SVRG for Stochastic Variance Reduction Gradient, SAG for Stochastic Averaging Gradient.}

\item{trace}{An option for saving intermediate quantities. All intermediate standardized-scale parameter estimates(trace=="param"), objective function values at each iteration(trace=="fVal"), or both(trace=="all") are saved in optResult. Since this is computationally very heavy, it should be only used for decently small-sized dataset and small maxit. A default is "none".}
}
\value{
coef A p by length(lambda) matrix of coefficients

std_coef A p by length(lambda) matrix of coefficients in a standardized scale

lambda The actual sequence of lambda values used.

nullDev Null deviance defined to be 2*(logLik_sat -logLik_null)

deviance Deviance defined to be 2*(logLik_sat -logLik(model))

optResult A list containing the result of the optimization. fValues, subGradients contain objective function values and subgradient vectors at each lambda value. If trace = TRUE, corresponding intermediate quantities are saved as well.

iters Number of iterations(EM updates) if method = "CD". Number of steps taken otherwise.
}
\description{
Fit a model using PUlasso algorithm over a regularization path. The regularization path is computed at a grid of values for the regularization parameter lambda.
}
\examples{
data("simulPU")
fit<-grpPUlasso(X=simulPU$X,z=simulPU$z,py1=simulPU$truePY1)
}
