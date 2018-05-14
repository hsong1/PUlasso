#' Solve PU problem with lasso or group lasso penalty.
#' 
#' Fit a model using PUlasso algorithm over a regularization path. The regularization path is computed at a grid of values for the regularization parameter lambda. 
#' 
#'@importFrom Rcpp evalCpp
#'@importFrom methods as
#'@import Matrix
#'@useDynLib PUlasso
#'@param X Input matrix; each row is an observation. Can be a matrix or a sparse matrix.
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
#'@param stepSize A step size for gradient-based optimization. if NULL, a step size is taken to be stepSizeAdj/mean(Li) where Li is a Lipschitz constant for ith sample
#'@param stepSizeAdjustment A step size adjustment. By default, adjustment is 1 for GD and SGD, 1/8 for SVRG and 1/16 for SAG.
#'@param batchSize A batch size. Default is 1.
#'@param updateFrequency An update frequency of full gradient for method =="SVRG"
#'@param samplingProbabilities sampling probabilities for each of samples for stochastic gradient-based optimization. if NULL, each sample is chosen proportionally to Li.
#'@param method Optimization method. Default is Coordinate Descent. CD for Coordinate Descent, GD for Gradient Descent, SGD for Stochastic Gradient Descent, SVRG for Stochastic Variance Reduction Gradient, SAG for Stochastic Averaging Gradient.
#'@param trace An option for saving intermediate quantities. All intermediate standardized-scale parameter estimates(trace=="param"), objective function values at each iteration(trace=="fVal"), or both(trace=="all") are saved in optResult. Since this is computationally very heavy, it should be only used for decently small-sized dataset and small maxit. A default is "none".
#'@return coef A p by length(lambda) matrix of coefficients
#'@return std_coef A p by length(lambda) matrix of coefficients in a standardized scale
#'@return lambda The actual sequence of lambda values used.
#'@return nullDev Null deviance defined to be 2*(logLik_sat -logLik_null)
#'@return deviance Deviance defined to be 2*(logLik_sat -logLik(model))
#'@return optResult A list containing the result of the optimization. fValues, subGradients contain objective function values and subgradient vectors at each lambda value. If trace = TRUE, corresponding intermediate quantities are saved as well.
#'@return iters Number of iterations(EM updates) if method = "CD". Number of steps taken otherwise.
#'@examples
#'data("simulPU")
#'fit<-grpPUlasso(X=simulPU$X,z=simulPU$z,pi=simulPU$truePY1)
#'@export
#'
grpPUlasso <-function(X,z,pi,initial_coef=NULL,group=1:ncol(X),
                penalty=NULL,lambda=NULL, nlambda = 100, 
                lambdaMinRatio=ifelse(N < p, 0.05, 0.005),maxit=ifelse(method=="CD",10000,N*10),
                eps=1e-04,inner_eps = 1e-02, 
                verbose = FALSE, stepSize=NULL, stepSizeAdjustment = NULL, batchSize=1, updateFrequency=N,
                samplingProbabilities=NULL, method=c("CD","GD","SGD","SVRG","SAG"),trace=c("none","param","fVal","all"))
{
  if(is.null(dim(X))){stop("not a valid X")}
  if(nrow(X)!=length(z)){stop("nrow(X) must be same as length(z)")}
    group0 <- group
    if(!is.numeric(group)){
      group <-  as.factor(group)
      levels(group) <-1:length(unique(group))
      group <- as.numeric(levels(group)[group])
    }
    if(is.null(colnames(X))){colnames(X) <- paste("V",1:ncol(X),sep = "")}
    X_lu <- X[order(z,decreasing = T),order(group),drop=F] # Copy of X, namely X_lu, is created
    remove(X) # Now delete original copy to save some memory
    group0 <- group0[order(group)]
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
    
    #input checks
    method = match.arg(method,choices=c("CD","GD","SGD","SVRG","SAG"))
    trace = match.arg(trace,choices=c("none","param","fVal","all"))
    trace = switch(trace,"none"=0,"param"=1,"fVal"=2,"all"=3)
    
    if(length(z_lu)!=N){stop("nrow(X) should be the same as length(z)")}
    if(length(group)!=p){stop("lenght(group) should be the same as ncol(X)")}
    if(!all(group==sort(group))){stop("columns must be in order")}
    if(!is.null(penalty)){
      if(length(penalty)!=(J-1)){stop("length(penalty) should be the same as the group size")}
    }
    if(!all(z_lu%in%c(0,1))){stop("z should be 0 or 1")}
    # if(mean(z_lu)==0||mean(z_lu)==1){stop("y can't be all 0 or 1")}
    if(!is.null(stepSize)){
      if(stepSize <= 0){stop("step size should be > 0 ")}
    }
    if(!is.null(samplingProbabilities)){
      if(length(samplingProbabilities)!=N){stop("length of sampling probability should be equal to the nrow(X)")}
      if(any(samplingProbabilities<0)){stop("all sampling probabilities should >= 0")}
    }
    if (is.null(lambda)) {
      if (lambdaMinRatio >= 1){stop("lambdaMinRatio should be less than 1")}
      if (nlambda < 1){stop("nlambda should be at least 1")}
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
    if(!(class(X_lu)=="matrix"||class(X_lu)=="dgCMatrix")){stop("X must be a matrix or a sparse matrix")}
   
    if(is.null(initial_coef)){
      icoef <- rep(0,p+1)
      pr <-  pi
      icoef[1] = log(pr/(1-pr))
      if(is.nan(icoef[1])){stop("not a valid pi=P(Y=1)")}
    }else{
      if(length(initial_coef)!=(p+1)){stop("length of initial_coef should be the same as ncol(X_lu)+1")}
      icoef <- initial_coef
    }
    
    gsize <-  c(1,table(group))
    if(is.null(penalty)){
      pen <- c(0,rep(1,J-1))*sqrt(gsize)
    } else{
      pen <- c(0, penalty)
    }
    
    if(is.null(stepSize)||is.null(samplingProbabilities)){useLipschitz = TRUE}
    if(is.null(stepSize)){stepSize=0}
    if(is.null(samplingProbabilities)){samplingProbabilities=1}
    if(is.null(stepSizeAdjustment)){
      if(method %in% c("GD","SGD")){
        adj = 1
        }else if(method =="SVRG"){
        adj = 1/8
        }else{
        adj = 1/16
      }
    }else{adj = stepSizeAdjustment}
    
    if(!is.sparse){
      g<-LU_dense_cpp(X_ = X_lu,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
                lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                verbose_ = verbose, stepSize_=stepSize,stepSizeAdj_= adj, batchSize_=batchSize,updateFreq_=updateFrequency,
                samplingProbabilities_=samplingProbabilities,useLipschitz_=useLipschitz,method_=method,trace_=trace)
    }else{
      g<-LU_sparse_cpp(X_ = X_lu,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                      lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
                      lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                      inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                      verbose_ = verbose, stepSize_=stepSize,stepSizeAdj_= adj,batchSize_=batchSize,updateFreq_=updateFrequency,
                      samplingProbabilities_=samplingProbabilities,useLipschitz_=useLipschitz,method_=method,trace_=trace)
    }
    
    coef <-  g$coef
    colnames(coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(coef) <- c("(Intercept)",colnames(X_lu))
    
    std_coef <- g$std_coef
    colnames(std_coef) <-  paste("l",1:length(g$lambda),sep = "")
    rownames(std_coef) <- c("(Intercept)",paste("group",group0))
  
    if(method=="CD"){iters = g$nUpdates
    }else{iters=g$iters}
    
    if(trace==1){
      std_coef_all= list()
      for(k in 1:length(g$lambda)){
        std_coef_all[[k]] <- g$beta_all[((k-1)*(p+1)+1):(k*(p+1)),1:(iters[k]+1)]
        rownames(std_coef_all[[k]]) <- rownames(std_coef)
        colnames(std_coef_all[[k]]) <- 1:(iters[k]+1)
      }
      names(std_coef_all) <- paste("lambda",1:length(g$lambda),sep="")
    } else if(trace==2){
      fVals_all = g$fVals_all[1:max(iters),,drop=F]
    } else if(trace==3) {
      std_coef_all= list()
      for(k in 1:length(g$lambda)){
        std_coef_all[[k]] <- g$beta_all[((k-1)*(p+1)+1):(k*(p+1)),1:(iters[k]+1)]
        rownames(std_coef_all[[k]]) <- rownames(std_coef)
        colnames(std_coef_all[[k]]) <- 1:(iters[k]+1)
      }
      names(std_coef_all) <- paste("lambda",1:length(g$lambda),sep="")
      
      fVals_all = g$fVals_all[1:max(iters),,drop=F]
      
    }
    
    if(method=="CD"){
      if(trace==0){
        optResult<-list(method=method,fValues=g$fVals,subGradients=g$subgrads,maxit=maxit)
      }else if(trace==1){
        optResult<-list(method=method,fValues=g$fVals,subGradients=g$subgrads, std_coef_all = std_coef_all,maxit=maxit)
      }else if(trace==2){
        optResult<-list(method=method,fValues=g$fVals,subGradients=g$subgrads, fValues_all = fVals_all, maxit=maxit)
      }else if(trace==3){
        optResult<-list(method=method,fValues=g$fVals,subGradients=g$subgrads, std_coef_all = std_coef_all, fValues_all = fVals_all,maxit=maxit)
      }else{
          stop("optResult error")
        }
      
    }else{ #if not CD,
      
      if(trace==0){
        optResult<-list(method=method,fValues=g$fVals,subGradients=g$subgrads,
                        stepSize=g$stepSize,samplingProbabilities=g$samplingProbabilities,maxit=maxit)
      }else if(trace==1){
        optResult<-list(method=method,fValues=g$fVals,subGradients=g$subgrads,
                        stepSize=g$stepSize,samplingProbabilities=g$samplingProbabilities,
                        std_coef_all = std_coef_all,maxit=maxit)
      }else if(trace==2){
        optResult<-list(method=method,fValues=g$fVals,subGradients=g$subgrads,
                        stepSize=g$stepSize,samplingProbabilities=g$samplingProbabilities,
                        fValues_all = fVals_all,maxit=maxit)
      }else if(trace==3){
        optResult<-list(method=method,fValues=g$fVals,subGradients=g$subgrads,
                        stepSize=g$stepSize,samplingProbabilities=g$samplingProbabilities,
                        std_coef_all = std_coef_all,fValues_all = fVals_all,maxit=maxit)
        
      }else{
        stop("optResult error")
      }
    }
    
    # warning
    if(method %in% c("CD","GD")){
      widx<-which(g$convFlag==1)
      if(length(widx)>0){
        for(i in 1:length(widx)){
          warning(paste("convergence failed at ",widx[i],"th lambda, ", iters[widx[i]],"th iterations",sep=""))
        }
      }
    }else{
      if(verbose){
        widx<-which(g$convFlag==0)
        if(length(widx)>0){
          for(i in 1:length(widx)){
           cat('|param.diff| < eps at',widx[i],'th lambda,', iters[widx[i]],'th iterations\n')
          }
        }
      }
    }
  
    result <- structure(list(coef = coef, std_coef = std_coef, lambda=g$lambda,
                             nullDev=g$nullDev,deviance=g$deviance,optResult=optResult,
                             iters= iters,call=match.call()),class="PUfit")

    return(result)
}
