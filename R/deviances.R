#'Deviance
#'
#'Calculate deviances at provided coefficients
#'
#'@param X Input matrix
#'@param z Response vector
#'@param pi True prevalence Pr(Y=1)
#'@param coefMat A coefficient matrix whose column corresponds to a set of coefficients
#'@return deviances
#'@examples
#'data("simulPU")
#'coef0<-replicate(2,runif(ncol(simulPU$X)+1))
#'deviances(simulPU$X,simulPU$z,pi=simulPU$truePY1,coefMat = coef0)
#'@importFrom Rcpp evalCpp
#'@importFrom methods as
#'@useDynLib PUlasso
#'@export
#'
deviances <-function(X,z,pi,coefMat)
{
  X_lu <- X[order(z,decreasing = T),,drop=F]
  z_lu <- z[order(z,decreasing = T)]
  
  if(typeof(X_lu)=="double"){X_lu <- Matrix(X_lu)}
  
  # Dimensions
  N <- nrow(X_lu)
  p <- ncol(X_lu)
  nl <- sum(z_lu)
  nu <- N-nl
  
  is.sparse = FALSE
  if (inherits(X_lu, "sparseMatrix")) {
    is.sparse = TRUE
    X_lu = as(X_lu, "CsparseMatrix")
    X_lu = as(X_lu, "dgCMatrix")
  } else if (inherits(X_lu,"dgeMatrix")){
    X_lu = as.matrix(X_lu)
  }
  if(!(class(X_lu)=="matrix"||class(X_lu)=="dgCMatrix")){stop("X must be a matrix, or a sparse matrix")}
  if(is.null(colnames(X_lu))){colnames(X_lu) <- paste("V",1:p,sep = "")}
  if(typeof(coefMat)=="double"){coefMat <- as.matrix(coefMat)}
  if(nrow(coefMat)!=(p+1)){stop("nrow(coefMat) must be the same as p+1")}
  
  X_lu_int = cbind(rep(1,N),X_lu)
  
  dev<- deviances_cpp(X_ = X_lu_int,z_ = z_lu,pi_ = pi,coefMat_ = coefMat,isSparse = is.sparse)
  
  return(dev)
}
