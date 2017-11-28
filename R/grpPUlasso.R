#' Solve PU problem with lasso or group lasso penalty.
#' 
#' Fit a model using PUlasso algorithm over a regularization path. The regularization path is computed at a grid of values for the regularization parameter lambda. 
#' 
#'@importFrom Rcpp evalCpp
#'@importFrom methods as
#'@import Matrix
#'@import bigmemory
#'@useDynLib PUlasso
#'@param X Input matrix; each row is an observation. Can be in matrix, sparse matrix or big matrix format.
#'@param z Response vector representing whether an observation is labelled or unlabelled.
#'@param pi True prevalence Pr(Y=1)
#'@param initial_coef A vector representing an initial point where we start PUlasso algorithm from.
#'@param group A vector representing grouping of the coefficients. For the least ambiguity, it is recommended if group is provided in the form of vector of consecutive ascending integers.
#'@param penalty penalty to be applied to the model. Default is sqrt(group size) for each of the group.
#'@param lambda A user supplied sequence of lambda values. If unspecified, the function automatically generates its own lambda sequence based on nlambda and lambdaMinRatio.
#'@param nlambda The number of lambda values. 
#'@param lambdaMinRatio Smallest value for lambda, as a fraction of lambda.max which leads to the intercept only model.
#'@param maxit Maximum number of iterations. 
#'@param eps Convergence threshold for the outer loop. The algorithm iterates until the maximum change in coefficients is less than eps in the outer loop.
#'@param inner_eps Convergence threshold for the inner loop. The algorithm iterates until the maximum change in coefficients is less than eps in the inner loop.
#'@param verbose a logical value. if TRUE, the function prints out the fitting process.
#'@return coef A p by length(lambda) matrix of coefficients
#'@return std_coef A p by length(lambda) matrix of coefficients in a standardized scale
#'@return lambda The actual sequence of lambda values used.
#'@return nullDev Null deviance defined to be 2*(logLik_sat -logLik_null)
#'@return deviance Deviance defined to be 2*(logLik_sat -logLik(model))
#'@return iters number of iterations
#'@examples
#'data("simulPU")
#'fit<-grpPUlasso(X=simulPU$X,z=simulPU$z,pi=simulPU$truePY1)
#'@export
#'
grpPUlasso <-function(X,z,pi,initial_coef=NULL,group=1:ncol(X),
                penalty=NULL,lambda=NULL, nlambda = 100, 
                lambdaMinRatio=ifelse(N < p, 0.05, 0.005),maxit=100000,
                eps=1e-04,inner_eps = 1e-02, 
                verbose = FALSE)
{
  if(is.big.matrix(X)){
    invPermute<-function(ind){
      rind<-c()
      for(i in 1:length(ind)){
        rind[ind[i]]=i
      }
      return(rind)
    }
    
    if(!is.numeric(group)){
      group <-  as.factor(group)
      levels(group) <-1:length(unique(group))
      group <- as.numeric(levels(group)[group])
    }
    
    X_lu<-X
    rPermute=as.numeric(order(z,decreasing = T))
    cPermute=as.numeric(order(group))
    mpermute(X_lu,order=rPermute)
    mpermuteCols(X_lu,order=cPermute)
    irPermute=invPermute(rPermute)
    icPermute=invPermute(cPermute)
    group <- group[order(group)]
    z_lu <- z[order(z,decreasing = T)]

    # Dimensions
    N <- nrow(X_lu)
    p <- ncol(X_lu)
    nl <- sum(z_lu)
    nu <- N-nl
    J <-  length(unique(group))+1
    
    # Apply strong set screening if p >N
    usestrongSet=ifelse(N<p,FALSE,TRUE)
    
    #input check
    
    if(length(z_lu)!=N){stop("nrow(X) should be the same as length(z)")}
    if(length(group)!=p){stop("lenght(group) should be the same as ncol(X)")}
    if(!all(group==sort(group))){stop("columns must be in order")}
    if(!is.null(penalty)){
      if(length(penalty)!=(J-1)){stop("length(penalty) should be the same as the group size")}
    }
    if(!all(z_lu%in%c(0,1))){stop("z should be 0 or 1")}
    # if(mean(z_lu)==0||mean(z_lu)==1){stop("y can't be all 0 or 1")}
    if (is.null(lambda)) {
      if (lambdaMinRatio >= 1){stop("lambdaMinRatio should be less than 1")}
      user_lambdaseq = FALSE
      lambdaseq = c(0.1,0.01) # will not be used
    } else {
      if (any(lambda < 0)){stop("lambdas should be non-negative")}
      user_lambdaseq = TRUE
      lambdaseq = sort(lambda, decreasing = TRUE)
    }
    
    if(!(class(X_lu)=="matrix"||class(X_lu)=="dgCMatrix"||class(X_lu)=="big.matrix")){stop("X must be a matrix, a sparse matrix, or a big matrix")}
    if(is.null(colnames(X_lu))){colnames(X_lu) <- paste("V",1:p,sep = "")}
    
    
    if(is.null(initial_coef))
    {
      icoef <- rep(0,p+1)
      pr <-  pi
      icoef[1] = log(pr/(1-pr))
      if(is.nan(icoef[1])){stop("not a valid pi=P(Y=1)")}
    }else
    {
      if(length(initial_coef)!=(p+1)){stop("length of initial_coef should be the same as ncol(X_lu)+1")}
      icoef <- initial_coef
    }
    
    gsize <-  c(1,table(group))
    if(is.null(penalty)){
      pen <- c(0,rep(1,J-1))*sqrt(gsize)
    } else{
      pen <- c(0, penalty)
    }
    
    g<-LU_big_cpp(X_ = X_lu@address,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                  lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
                  lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                  inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                  verbose_ = verbose)
    
    coef <-  g$coef
    colnames(coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(coef) <- c("(Intercept)",colnames(X_lu))
    
    std_coef <- g$std_coef
    colnames(std_coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(std_coef) <- c("(Intercept)",colnames(X_lu))
    
    widx<-which(g$convFlag==1)
    if(length(widx)>0){
      for(i in 1:length(widx)){
        warning(paste("convergence failed at ",widx[i],"th lambda, ", g$iters[widx[i]],"th iterations",sep=""))
      }
    }
    
    result <- structure(list(coef = coef, std_coef = std_coef, lambda=g$lambda,
                             nullDev=g$nullDev,deviance=g$deviance,
                             iters= g$iters),class="PUfit")
    
    #Permute back X matrix
    mpermute(X_lu,order=as.numeric(irPermute))
    mpermuteCols(X_lu,order=as.numeric(icPermute))
    return(result)
    
  }else{
    if(!is.numeric(group)){
      group <-  as.factor(group)
      levels(group) <-1:length(unique(group))
      group <- as.numeric(levels(group)[group])
    }
    X_lu <- X[order(z,decreasing = T),order(group),drop=F]
    group <- group[order(group)]
    z_lu <- z[order(z,decreasing = T)]
    if(typeof(X_lu)=="double"){X_lu <- Matrix::Matrix(X_lu)}
    
    # Dimensions
    N <- nrow(X_lu)
    p <- ncol(X_lu)
    nl <- sum(z_lu)
    nu <- N-nl
    J <-  length(unique(group))+1
    
    # Apply strong set screening if p >N
    usestrongSet=ifelse(N<p,FALSE,TRUE)
    
    #input check
    
    if(length(z_lu)!=N){stop("nrow(X) should be the same as length(z)")}
    if(length(group)!=p){stop("lenght(group) should be the same as ncol(X)")}
    if(!all(group==sort(group))){stop("columns must be in order")}
    if(!is.null(penalty)){
      if(length(penalty)!=(J-1)){stop("length(penalty) should be the same as the group size")}
    }
    if(!all(z_lu%in%c(0,1))){stop("z should be 0 or 1")}
    # if(mean(z_lu)==0||mean(z_lu)==1){stop("y can't be all 0 or 1")}
    if (is.null(lambda)) {
      if (lambdaMinRatio >= 1){stop("lambdaMinRatio should be less than 1")}
      user_lambdaseq = FALSE
      lambdaseq = c(0.1,0.01) # will not be used
    } else {
      if (any(lambda < 0)){stop("lambdas should be non-negative")}
      user_lambdaseq = TRUE
      lambdaseq = sort(lambda, decreasing = TRUE)
    }
    
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
    
    
    if(is.null(initial_coef))
    {
      icoef <- rep(0,p+1)
      pr <-  pi
      icoef[1] = log(pr/(1-pr))
      if(is.nan(icoef[1])){stop("not a valid pi=P(Y=1)")}
    }else
    {
      if(length(initial_coef)!=(p+1)){stop("length of initial_coef should be the same as ncol(X_lu)+1")}
      icoef <- initial_coef
    }
    
    gsize <-  c(1,table(group))
    if(is.null(penalty)){
      pen <- c(0,rep(1,J-1))*sqrt(gsize)
    } else{
      pen <- c(0, penalty)
    }
    
    
    g<-LU_cpp(X_ = X_lu,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
              lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
              lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
              inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
              isSparse = is.sparse,verbose_ = verbose)
    
    coef <-  g$coef
    colnames(coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(coef) <- c("(Intercept)",colnames(X_lu))
    
    std_coef <- g$std_coef
    colnames(std_coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(std_coef) <- c("(Intercept)",colnames(X_lu))
    
    widx<-which(g$convFlag==1)
    if(length(widx)>0){
      for(i in 1:length(widx)){
        warning(paste("convergence failed at ",widx[i],"th lambda, ", g$iters[widx[i]],"th iterations",sep=""))
      }
    }
    
    result <- structure(list(coef = coef, std_coef = std_coef, lambda=g$lambda,
                             nullDev=g$nullDev,deviance=g$deviance,
                             iters= g$iters),class="PUfit")
    
    
    
    return(result)
  }
  
}
