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
  if(is.null(dim(X))){stop("not a valid X")}
    if(is.null(colnames(X))){colnames(X) <- paste("V",1:ncol(X),sep = "")}
    X_lu <- X[order(z,decreasing = T),,drop=F] # Copy of X, namely X_lu, is created
    z_lu <- z[order(z,decreasing = T)]
    if(typeof(X_lu)!="double"){X_lu <- X_lu + 0.0} # Ensure type of X is double 
  is.sparse = FALSE
  if (inherits(X_lu, "sparseMatrix")) {
    is.sparse = TRUE
    X_lu = as(X_lu, "CsparseMatrix")
    X_lu = as(X_lu, "dgCMatrix")
  } else if (inherits(X_lu,"dgeMatrix")){
    X_lu = as.matrix(X_lu)
  }
  if(!(class(X_lu)=="matrix"||class(X_lu)=="dgCMatrix")){stop("X must be a matrix, or a sparse matrix")}
  if(typeof(coefMat)=="double"){coefMat <- as.matrix(coefMat)}
  if(nrow(coefMat)!=(ncol(X)+1)){stop("nrow(coefMat) must be the same as p+1")}
  
  if(!is.sparse){
    dev<- deviances_dense_cpp(X_ = X_lu,z_ = z_lu,pi_ = pi,coefMat_ = coefMat)
  }else{
    dev<- deviances_sparse_cpp(X_ = X_lu,z_ = z_lu,pi_ = pi,coefMat_ = coefMat)
  }
  return(c(dev))
  
}

