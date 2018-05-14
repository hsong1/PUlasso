#' Cross-validation for PUlasso
#' 
#' Do a n-fold cross-validation for PUlasso.
#' 
#'@importFrom Rcpp evalCpp
#'@import methods
#'@import Matrix
#'@import parallel
#'@import doParallel
#'@import foreach
#'@importFrom stats sd
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
#'@param trace An option for saving intermediate quantities when fitting a full dataset.
#'@param nfolds Number of cross-validation folds to be created.
#'@param nfits Number of cross-validation models which will be fitted. Default is to fit the model for each of the cross-validation fold.
#'@param nCores Number of threads to be used for parallel computing. If nCores=0, it is set to be (the number of processors available-1) . Default value is 1. 
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
                         lambdaMinRatio=ifelse(N < p, 0.05, 0.005),maxit=ifelse(method=="CD",1000,N*10),
                         eps=1e-04,inner_eps = 1e-02, 
                         verbose = FALSE, stepSize=NULL, stepSizeAdjustment = NULL, batchSize=1, updateFrequency = N,
                         samplingProbabilities=NULL, method=c("CD","GD","SGD","SVRG","SAG"),
                         nfolds=10,nfits=nfolds,nCores=1,trace=c("none","param","fVal","all"))
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
  remove(X)
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
    if (nlambda ==1){stop("More than one lambda needed for cross-validation")}
    user_lambdaseq = FALSE
    lambdaseq = c(0.1,0.01) # will not be used
  } else {
    if (any(lambda < 0)){stop("lambdas should be non-negative")}
    if (length(lambda) ==1){stop("More than one lambda needed for cross-validation")}
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
  
  if(verbose){cat("Run grpPUlasso on full dataset\n")}
  if(!is.sparse){
    g_f<-LU_dense_cpp(X_ = X_lu,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                      lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
                      lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                      inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                      verbose_ = verbose, stepSize_=stepSize,stepSizeAdj_= adj, batchSize_=batchSize,updateFreq_ = updateFrequency,
                      samplingProbabilities_=samplingProbabilities,useLipschitz_=useLipschitz,method_=method,trace_=trace)
  }else{
    g_f<-LU_sparse_cpp(X_ = X_lu,z_ = z_lu,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                      lambdaseq_ = lambdaseq,user_lambdaseq_ = user_lambdaseq,pathLength_ = nlambda,
                      lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                      inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                      verbose_ = verbose, stepSize_=stepSize,stepSizeAdj_= adj, batchSize_=batchSize,updateFreq_ = updateFrequency,
                      samplingProbabilities_=samplingProbabilities,useLipschitz_=useLipschitz,method_=method,trace_=trace)
  }
 
  
  coef <-  g_f$coef
  colnames(coef) <-  paste("l",1:length(g_f$lambda),sep = "")
  rownames(coef) <- c("(Intercept)",colnames(X_lu))
  
  std_coef <- g_f$std_coef
  colnames(std_coef) <-  paste("l",1:length(g_f$lambda),sep = "")
  rownames(std_coef) <- c("(Intercept)",paste("group",group0))
  
  if(method=="CD"){iters = g_f$nUpdates
  }else{iters=g_f$iters}
  
  if(trace==1){
    std_coef_all= list()
    for(k in 1:length(g_f$lambda)){
      std_coef_all[[k]] <- g_f$beta_all[((k-1)*(p+1)+1):(k*(p+1)),1:(iters[k]+1)]
      rownames(std_coef_all[[k]]) <- rownames(std_coef)
      colnames(std_coef_all[[k]]) <- 1:(iters[k]+1)
    }
    names(std_coef_all) <- paste("lambda",1:length(g_f$lambda),sep="")
  } else if(trace==2){
    fVals_all = g_f$fVals_all[1:max(iters),,drop=F]
    
  } else if(trace==3) {
    std_coef_all= list()
    for(k in 1:length(g_f$lambda)){
      std_coef_all[[k]] <- g_f$beta_all[((k-1)*(p+1)+1):(k*(p+1)),1:(iters[k]+1)]
      rownames(std_coef_all[[k]]) <- rownames(std_coef)
      colnames(std_coef_all[[k]]) <- 1:(iters[k]+1)
    }
    names(std_coef_all) <- paste("lambda",1:length(g_f$lambda),sep="")
    fVals_all = g_f$fVals_all[1:max(iters),,drop=F]
    
  }
  
  if(method=="CD"){
    if(trace==0){
      optResult<-list(method=method,fValues=g_f$fVals,subGradients=g_f$subgrads,maxit=maxit)
    }else if(trace==1){
      optResult<-list(method=method,fValues=g_f$fVals,subGradients=g_f$subgrads, std_coef_all = std_coef_all,maxit=maxit)
    }else if(trace==2){
      optResult<-list(method=method,fValues=g_f$fVals,subGradients=g_f$subgrads, fValues_all = fVals_all,maxit=maxit)
    }else if(trace==3){
      optResult<-list(method=method,fValues=g_f$fVals,subGradients=g_f$subgrads, std_coef_all = std_coef_all, fValues_all = fVals_all,maxit=maxit)
    }else{
      stop("optResult error")
    }
    
  }else{ #if not CD,
    iters=g_f$iters
    if(trace==0){
      optResult<-list(method=method,fValues=g_f$fVals,subGradients=g_f$subgrads,
                      stepSize=g_f$stepSize,samplingProbabilities=g_f$samplingProbabilities,maxit=maxit)
    }else if(trace==1){
      optResult<-list(method=method,fValues=g_f$fVals,subGradients=g_f$subgrads,
                      stepSize=g_f$stepSize,samplingProbabilities=g_f$samplingProbabilities,
                      std_coef_all = std_coef_all,maxit=maxit)
    }else if(trace==2){
      optResult<-list(method=method,fValues=g_f$fVals,subGradients=g_f$subgrads,
                      stepSize=g_f$stepSize,samplingProbabilities=g_f$samplingProbabilities,
                      std_coef_all = std_coef_all,fValues_all = fVals_all,maxit=maxit)
    }else if(trace==3){
      optResult<-list(method=method,fValues=g_f$fVals,subGradients=g_f$subgrads,
                      stepSize=g_f$stepSize,samplingProbabilities=g_f$samplingProbabilities,
                      std_coef_all = std_coef_all,fValues_all = fVals_all,maxit=maxit)
      
    }else{
      stop("optResult error")
    }
  }
  PUfit <- structure(list(coef = coef, std_coef = std_coef, lambda=g_f$lambda,
                          nullDev=g_f$f_nullDev,deviance=g_f$deviance,optResult=optResult,
                          iters= iters,call=match.call()),class="PUfit")
  convFlag_f = g_f$convFlag
  remove(g_f)
  ############################################################################
  ## CV
  ############################################################################
  if(verbose){cat("Start Cross-Validation\n")}
  # shuffle X_lu
  pl <- sample(1:nl)
  pu <- sample(1:nu)
  # pl=1:nl
  # pu=1:nu
  X_l <- X_lu[1:nl,]
  X_u <- X_lu[(nl+1):(nl+nu),]
  X_l <- X_l[pl,]
  X_u <- X_u[pu,]
  
  rmrdl=nl%%nfolds
  rmrdu=nu%%nfolds
  cvnl=floor(nl/nfolds)
  cvnu=floor(nu/nfolds)
  cvsizel<-c()
  cvsizeu<-c()
  for(i in 1:nfolds){
    cvsizel[i]<-ifelse(i<=rmrdl,cvnl+1,cvnl)
    cvsizeu[i]<-ifelse(i<=rmrdu,cvnu+1,cvnu)
  }
  sidxl<-rep(1,nfolds)
  sidxu<-rep(1,nfolds)
  for(i in 2:nfolds){
    sidxl[i]=sidxl[i-1]+cvsizel[i-1]
    sidxu[i]=sidxu[i-1]+cvsizeu[i-1]
  }
  if (nCores<1||nCores>1){isParallel=TRUE
  }else{isParallel=FALSE}
  
  if (isParallel) {
    
    if(nCores==0){nCores = max(1, detectCores() - 1)
    }else{
      nCores = min(nCores,detectCores())
    }
    cl <- makeCluster(nCores)
    registerDoParallel(cl)
    clusterCall(cl, function(x) .libPaths(x), .libPaths())
    if(verbose){cat('Cross-Validation with',nCores, 'workers\n')}
    
    g=foreach(k = 1:nfits,
              .packages = "PUlasso",
              .combine = list,
              .multicombine = TRUE)  %dopar%  
              {
                vlidx<-sidxl[k]:(sidxl[k]+cvsizel[k]-1)
                vuidx<-sidxl[k]:(sidxu[k]+cvsizeu[k]-1)
                train_X=rbind(X_l[-vlidx,],rbind(X_u[-vuidx,]))
                train_z = c(rep(1,nl-cvsizel[k]),rep(0,nu-cvsizeu[k]))
                if(!is.sparse){
                  g.cv<-LU_dense_cpp(X_ = train_X,z_ = train_z,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                                     lambdaseq_ =PUfit$lambda,user_lambdaseq_ = FALSE,pathLength_ = nlambda,
                                     lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                                     inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                                     verbose_ = verbose, stepSize_=stepSize,stepSizeAdj_= adj, batchSize_=batchSize,updateFreq_ = updateFrequency,
                                     samplingProbabilities_=samplingProbabilities,useLipschitz_=useLipschitz,method_=method,trace_=trace)
                }else{
                  g.cv<-LU_sparse_cpp(X_ = train_X,z_ = train_z,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                                      lambdaseq_ =PUfit$lambda,user_lambdaseq_ = FALSE,pathLength_ = nlambda,
                                      lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                                      inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                                      verbose_ = verbose, stepSize_=stepSize,stepSizeAdj_= adj, batchSize_=batchSize,updateFreq_ = updateFrequency,
                                      samplingProbabilities_=samplingProbabilities,useLipschitz_=useLipschitz,method_=method,trace_=trace)
                }
                return(g.cv)
              }#end of foreach
    cvdev = foreach(k=1:nfits,
                    .packages = "PUlasso",
                    .combine  = "cbind" )%dopar%
                    {
                      vlidx<-sidxl[k]:(sidxl[k]+cvsizel[k]-1)
                      vuidx<-sidxl[k]:(sidxu[k]+cvsizeu[k]-1)
                      
                      test_X =rbind(X_l[vlidx,],rbind(X_u[vuidx,]))
                      test_z = c(rep(1,cvsizel[k]),rep(0,cvsizeu[k]))
                      
                      cvdev<- deviances(X = test_X,z = test_z,pi = pi,coefMat = g[[k]]$coef)
                      return(cvdev)
                    }
    
    
  } else {
    g=list()
    cvdev = matrix(0,ncol=nfits,nrow=length(PUfit$lambda))
    for(k in 1:nfits){
      if(verbose){cat('Cross-Validation for dataset',k,'\n')}
      vlidx<-sidxl[k]:(sidxl[k]+cvsizel[k]-1)
      vuidx<-sidxl[k]:(sidxu[k]+cvsizeu[k]-1)
      train_X=rbind(X_l[-vlidx,],rbind(X_u[-vuidx,]))
      train_z = c(rep(1,nl-cvsizel[k]),rep(0,nu-cvsizeu[k]))
      test_X =rbind(X_l[vlidx,],rbind(X_u[vuidx,]))
      test_z = c(rep(1,cvsizel[k]),rep(0,cvsizeu[k]))
      
      if(!is.sparse){
        g[[k]]<-LU_dense_cpp(X_ = train_X,z_ = train_z,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                             lambdaseq_ =PUfit$lambda,user_lambdaseq_ = TRUE,pathLength_ = nlambda,
                             lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                             inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                             verbose_ = verbose, stepSize_=stepSize,stepSizeAdj_= adj, batchSize_=batchSize,updateFreq_ = updateFrequency,
                             samplingProbabilities_=samplingProbabilities,useLipschitz_=useLipschitz,method_=method,trace_=trace)
      }else{
        g[[k]]<-LU_sparse_cpp(X_ = train_X,z_ = train_z,icoef_ = icoef,gsize_ = gsize,pen_ = pen,
                              lambdaseq_ =PUfit$lambda,user_lambdaseq_ = TRUE,pathLength_ = nlambda,
                              lambdaMinRatio_ = lambdaMinRatio,pi_ = pi,maxit_ = maxit,tol_ = eps,
                              inner_tol_ = inner_eps,useStrongSet_=usestrongSet,
                              verbose_ = verbose, stepSize_=stepSize,stepSizeAdj_= adj, batchSize_=batchSize,updateFreq_ = updateFrequency,
                              samplingProbabilities_=samplingProbabilities,useLipschitz_=useLipschitz,method_=method,trace_=trace)
      }
      cvdev[,k] <- deviances(X = test_X,z = test_z,pi = pi,coefMat = g[[k]]$coef)
      
    }
  }# End of Fitting
  # Summary
  coefmat <- list()
  std_coefmat <- list()
  for (i in 1:min(nfolds,nfits)){
    coefmat[[i]] <- g[[i]]$coef
    std_coefmat[[i]] <- g[[i]]$std_coef
    colnames(coefmat[[i]]) <-  paste("l",1:length(PUfit$lambda),sep = "")
    colnames(std_coefmat[[i]]) <-  paste("l",1:length(PUfit$lambda),sep = "")
    rownames(coefmat[[i]]) <- c("(Intercept)",colnames(X_lu))
    rownames(std_coefmat[[i]]) <- c("(Intercept)",paste("group",group0))
  }
  names(coefmat) <- paste("cv",1:min(nfolds,nfits),sep="")
  names(std_coefmat) <- paste("cv",1:min(nfolds,nfits),sep="")
  
  # cvdev=sapply(g,function(x){x$deviance})
  # rownames(cvdev)=paste("l",1:length(PUfit$lambda),sep = "")
  cvm=apply(cvdev,1,mean)
  cvsd <- apply(cvdev,1,sd)/sqrt(min(nfolds,nfits))
  names(cvm)=paste("l",1:length(PUfit$lambda),sep = "")
  names(cvsd)=paste("l",1:length(PUfit$lambda),sep = "")
  indmin <- min(which(cvm==min(cvm)))
  lambda.min <- PUfit$lambda[indmin]
  
  ind <-  intersect(which(cvm>=cvm[indmin]+cvsd[indmin]),(1:indmin))
  if(length(ind)==0){ind1se <-  indmin
  } else {
    ind1se <-  max(ind)
  }
  lambda.1se <- PUfit$lambda[ind1se]
  
  convFlagMat=sapply(g,function(x){x$convFlag})
  
  # Warning
  if(method %in% c("CD","GD")){
    widx<-which(convFlag_f==1)
    if(length(widx)>0){
      for(i in 1:length(widx)){
        warning(paste("convergence failed at ",widx[i],"th lambda, ", PUfit$iters[widx[i]],"th iterations",sep=""))
      }
    }
    
    for(j in 1:min(nfolds,nfits)){
      widx<-which(convFlagMat[,j]==1) 
      if(length(widx)>0){
        for(i in 1:length(widx)){
          warning(paste("cvset",j," convergence failed at ",widx[i],"th lambda",sep=""))}}
    }
  }else{
    if(verbose){
      widx<-which(convFlag_f==0)
      if(length(widx)>0){
        for(i in 1:length(widx)){
          cat('|param.diff| < eps at',widx[i],'th lambda,',PUfit$iters[widx[i]],'th iterations\n')
        }
      }
      
      for(j in 1:min(nfolds,nfits)){
        widx<-which(convFlagMat[,j]==0) 
        if(length(widx)>0){
          for(i in 1:length(widx)){
            cat('cvset',j,'|param.diff| < eps at',widx[i],'th lambda\n')}}
      }
    }
  }
  
  
  
  
  perm.ind <- list(lind=pl,uind=pu)
  result<-structure(list(cvm=cvm,cvsd=cvsd, cvcoef = coefmat, cvstdcoef = std_coefmat, lambda = PUfit$lambda, lambda.min= lambda.min,
                         lambda.1se=lambda.1se,PUfit=PUfit,perm.ind = perm.ind),class="cvPUfit")
  if(isParallel){stopCluster(cl)}
  return(result)
}
