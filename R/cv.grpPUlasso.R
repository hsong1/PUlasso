#' Cross-validation for PUlasso
#' 
#' Do a n-fold cross-validation for PUlasso.
#' 
#'@importFrom Rcpp evalCpp
#'@import Matrix
#'@import bigmemory
#'@importFrom stats sd
#'@importFrom methods as
#'@useDynLib PUlasso
#'@param X Input matrix; each row is an observation. Can be in matrix, sparse matrix, or big matrix format.
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
#'@param verbose A logical value. if TRUE, the function prints out the fitting process.
#'@param nfolds Number of cross-validation folds to be created.
#'@param nfits Number of cross-validation models which will be fitted. Default is to fit the model for each of the cross-validation fold.
#'@param nCores Number of OpenMP threads to be used for parallel computing. If nCores=0, it is set to be the number of processors available. Default value is 1. 
#'@return cvm Mean cross-validation error
#'@return cvsd Estimate of standard error of cvm
#'@return cvcoef Coefficients for each of the fitted CV models
#'@return cvstdcoef Coefficients in a standardized scale for each of the fitted CV models
#'@return lambda The actual sequence of lambda values used.
#'@return lambda.min Value of lambda that gives minimum cvm.
#'@return lambda.1se The largest value of lambda such that the error is within 1 standard error of the minimum cvm.
#'@return PUfit A fitted PUfit object for the full data
#'@examples
#'data("simulPU")
#'fit<-cv.grpPUlasso(X=simulPU$X,z=simulPU$z,pi=simulPU$truePY1)
#'@export
#'
cv.grpPUlasso <-function(X,z,pi,initial_coef=NULL,group=1:ncol(X),
                      penalty=NULL,lambda=NULL, nlambda = 100,
                      lambdaMinRatio=ifelse(N < p, 0.05, 0.005),maxit=100000,
                      eps=1e-04,inner_eps = 1e-02,
                      verbose = FALSE,nfolds=10,nfits=nfolds,nCores=1)
{
  if(is.null(dim(X))){stop("not a valid X")}
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
    if(is.null(colnames(X))){colnames(X) <- paste("V",1:ncol(X),sep = "")}
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

    # #input check
    # 
    if(length(z_lu)!=N){stop("nrow(X) should be the same as length(z)")}
    if(length(group)!=p){stop("lenght(group) should be the same as ncol(X)")}
    if(!all(group==sort(group))){stop("columns must be in order")}
    if(!is.null(penalty)){
      if(length(penalty)!=(J-1)){stop("length(penalty) should be the same as the group size")}
    }
    if(!all(z_lu%in%c(0,1))){stop("z should be 0 or 1")}
    if(mean(z_lu)==0||mean(z_lu)==1){stop("y can't be all 0 or 1")}
    if (is.null(lambda)) {
      if (lambdaMinRatio >= 1){stop("lambdaMinRatio should be less than 1")}
      user_lambdaseq = FALSE
      lambdaseq = c(0.1,0.01) # will not be used
    } else {
      if (any(lambda < 0)){stop("lambdas should be non-negative")}
      user_lambdaseq = TRUE
      lambdaseq = sort(lambda, decreasing = TRUE)
    }

    if(!(class(X_lu)=="matrix"||class(X_lu)=="dgCMatrix"||class(X_lu)=="big.matrix")){stop("X must be a matrix, a sparse matrix or a big matrix")}
    
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

    # shuffle X_lu
    pl <- sample(1:nl)
    pu <- sample(1:nu)+nl
    #pl = 1:nl
    #pu = (1:nu)+nl
    rPermute2 = c(pl,pu)
    mpermute(X_lu,order=as.numeric(rPermute2))
    irPermute2=invPermute(rPermute2)

    g<-cv_LU_big_cpp(X_ = X_lu@address,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                 lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
                 lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                 inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                 verbose_ = verbose,nfolds_ = nfolds,nfits_ = nfits, ncores_ = nCores)

    result <- structure(list(coef = coef, iters= g$iters,
                             nUpdates=g$nUpdates,
                             lambda=g$lambda),class="PUfits")
    coefmat <- list()
    std_coefmat <- list()
    for (i in 1:min(nfolds,nfits))
    {
      coefmat[[i]] <- g$coef[((p+1)*(i-1)+1):((p+1)*i),,drop=FALSE]
      std_coefmat[[i]] <- g$std_coef[((p+1)*(i-1)+1):((p+1)*i),,drop=FALSE]
      colnames(coefmat[[i]]) <-  paste("l",1:length(g$lambda),sep = "")
      colnames(std_coefmat[[i]]) <-  paste("l",1:length(g$lambda),sep = "")
      rownames(coefmat[[i]]) <- c("(Intercept)",colnames(X_lu))
      rownames(std_coefmat[[i]]) <- c("(Intercept)",colnames(X_lu))
    }
    names(coefmat) <- paste("cv",1:min(nfolds,nfits),sep="")
    names(std_coefmat) <- paste("cv",1:min(nfolds,nfits),sep="")
    cvm <- apply(g$deviance,1,mean)
    cvsd <- apply(g$deviance,1,sd)/sqrt(min(nfolds,nfits))
    indmin <- min(which(cvm==min(cvm)))
    lambda.min <- g$lambda[indmin]

    ind <-  intersect(which(cvm>=cvm[indmin]+cvsd[indmin]),(1:indmin))
    if(length(ind)==0){ind1se <-  indmin
    } else {
      ind1se <-  max(ind)
    }
    lambda.1se <- g$lambda[ind1se]

    coef <-  g$f_coef
    colnames(coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(coef) <- c("(Intercept)",colnames(X_lu))

    std_coef <- g$f_std_coef
    colnames(std_coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(std_coef) <- c("(Intercept)",colnames(X_lu))

    widx<-which(g$f_convFlag==1)
    if(length(widx)>0){
      for(i in 1:length(widx)){
        warning(paste("convergence failed at ",widx[i],"th lambda, ", g$f_iters[widx[i]],"th iterations",sep=""))
      }
    }
    for(j in 1:min(nfolds,nfits)){
      widx<-which(g$convFlagMat[,j]==1)
      if(length(widx)>0){
        for(i in 1:length(widx)){
          warning(paste("cvset",j," convergence failed at ",widx[i],"th lambda",sep=""))
        }
      }
    }

    PUfit <- structure(list(coef = coef, std_coef = std_coef, lambda=g$f_lambda,
                            nullDev=g$f_nullDev,deviance=g$f_deviance,
                            iters= g$f_iters),class="PUfit")
    perm.ind <- list(lind=pl,uind=pu)
    result<-structure(list(cvm=cvm,cvsd=cvsd, cvcoef = coefmat, cvstdcoef = std_coefmat, lambda = g$lambda, lambda.min= lambda.min,
                           lambda.1se=lambda.1se,PUfit=PUfit,perm.ind = perm.ind),class="cvPUfit")
    #Permute back X matrix
    mpermute(X_lu,order=as.numeric(irPermute2))
    mpermute(X_lu,order=as.numeric(irPermute))
    mpermuteCols(X_lu,order=as.numeric(icPermute))
  return(result)
  }else{
    if(!is.numeric(group)){
      group <-  as.factor(group)
      levels(group) <-1:length(unique(group))
      group <- as.numeric(levels(group)[group])
    }
    if(is.null(colnames(X))){colnames(X) <- paste("V",1:ncol(X),sep = "")}
    X_lu <- X[order(z,decreasing = T),order(group),drop=F]
    group <- group[order(group)]
    z_lu <- z[order(z,decreasing = T)]
    if(typeof(X_lu)!="double"){X_lu <- X_lu + 0.0} # Ensure type of X is double
    
    
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
    if(mean(z_lu)==0||mean(z_lu)==1){stop("y can't be all 0 or 1")}
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
    if(!(class(X_lu)=="matrix"||class(X_lu)=="dgCMatrix"||class(X_lu)=="big.matrix")){stop("X must be a matrix, a sparse matrix, or a big matrix")}

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
    
    # shuffle X_lu
    pl <- sample(1:nl)
    pu <- sample(1:nu)
    X_l <- X_lu[1:nl,]
    X_u <- X_lu[(nl+1):(nl+nu),]
    X_l <- X_l[pl,]
    X_u <- X_u[pu,]
    X_lu <- rbind(X_l,X_u)
    #X_lu_int = cbind(rep(1,N),X_lu)
    
    if(!is.sparse){
      g<-cv_LU_dense_cpp(X_ = X_lu,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                   lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
                   lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                   inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                   verbose_ = verbose,nfolds_ = nfolds,nfits_ = nfits, ncores_ = nCores)
    }else{
      g<-cv_LU_sparse_cpp(X_ = X_lu,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                   lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
                   lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                   inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                   verbose_ = verbose,nfolds_ = nfolds,nfits_ = nfits, ncores_ = nCores)
    }
   
    #
    # result <- structure(list(coef = coef, iters= g$iters,
    #                          nUpdates=g$nUpdates,
    #                          lambda=g$lambda),class="PUfits")
    coefmat <- list()
    std_coefmat <- list()
    for (i in 1:min(nfolds,nfits))
    {
      coefmat[[i]] <- g$coef[((p+1)*(i-1)+1):((p+1)*i),,drop=FALSE]
      std_coefmat[[i]] <- g$std_coef[((p+1)*(i-1)+1):((p+1)*i),,drop=FALSE]
      colnames(coefmat[[i]]) <-  paste("l",1:length(g$lambda),sep = "")
      colnames(std_coefmat[[i]]) <-  paste("l",1:length(g$lambda),sep = "")
      rownames(coefmat[[i]]) <- c("(Intercept)",colnames(X_lu))
      rownames(std_coefmat[[i]]) <- c("(Intercept)",colnames(X_lu))
    }
    names(coefmat) <- paste("cv",1:min(nfolds,nfits),sep="")
    names(std_coefmat) <- paste("cv",1:min(nfolds,nfits),sep="")
    cvm <- apply(g$deviance,1,mean)
    cvsd <- apply(g$deviance,1,sd)/sqrt(min(nfolds,nfits))
    indmin <- min(which(cvm==min(cvm)))
    lambda.min <- g$lambda[indmin]
    
    ind <-  intersect(which(cvm>=cvm[indmin]+cvsd[indmin]),(1:indmin))
    if(length(ind)==0){ind1se <-  indmin
    } else {
      ind1se <-  max(ind)
    }
    lambda.1se <- g$lambda[ind1se]
    
    coef <-  g$f_coef
    colnames(coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(coef) <- c("(Intercept)",colnames(X_lu))
    
    std_coef <- g$f_std_coef
    colnames(std_coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(std_coef) <- c("(Intercept)",colnames(X_lu))
    
    widx<-which(g$f_convFlag==1)
    if(length(widx)>0){
      for(i in 1:length(widx)){
        warning(paste("convergence failed at ",widx[i],"th lambda, ", g$f_iters[widx[i]],"th iterations",sep=""))
      }
    }
    for(j in 1:min(nfolds,nfits)){
      widx<-which(g$convFlagMat[,j]==1)
      if(length(widx)>0){
        for(i in 1:length(widx)){
          warning(paste("cvset",j," convergence failed at ",widx[i],"th lambda",sep=""))
        }
      }
    }
    
    PUfit <- structure(list(coef = coef, std_coef = std_coef, lambda=g$f_lambda,
                            nullDev=g$f_nullDev,deviance=g$f_deviance,
                            iters= g$f_iters),class="PUfit")
    perm.ind <- list(lind=pl,uind=pu)
    result<-structure(list(cvm=cvm,cvsd=cvsd, cvcoef = coefmat, cvstdcoef = std_coefmat, lambda = g$lambda, lambda.min= lambda.min,
                           lambda.1se=lambda.1se,PUfit=PUfit,perm.ind = perm.ind),class="cvPUfit")
    
    return(result)
  }
  
}
