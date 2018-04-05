library(PUlasso)
library(testthat)

data("simulPU")
load("sysdata.rda")
cvXrecover<-function(cvlufit,X,z,group=1:ncol(X),j,nfolds=length(cvlufit$cvcoef)){
  X_lu <- X[order(z,decreasing = T),order(group),drop=F]
  z_lu <- z[order(z,decreasing = T)]
  nl=sum(z_lu)
  nu=length(z_lu)-nl
  pl = cvlufit$perm.ind$lind
  pu = cvlufit$perm.ind$uind
  X_l <- X_lu[1:nl,]
  X_u <- X_lu[(nl+1):(nl+nu),]
  pXl <- X_l[pl,]
  pXu <- X_u[pu,]
  remove(X_l,X_u)
  
  rmdrl = nl %% nfolds;
  cvSizelt = nl / nfolds;
  rmdru = nu %% nfolds;
  cvSizeut = nu / nfolds;
  
  cvSizel = c()
  cvSizeu = c()
  
  for (i in 1:nfolds){
    cvSizel[i]=ifelse(i<rmdrl,(cvSizelt+1),(cvSizelt))
    cvSizeu[i]=ifelse(i<rmdru,(cvSizeut+1),(cvSizeut))
  }
  
  Xl_sIdx<-c(1)
  Xu_sIdx<-c(1)
  for(i in 2:nfolds){
    Xl_sIdx[i]=Xl_sIdx[i-1]+cvSizel[i-1];
    Xu_sIdx[i]=Xu_sIdx[i-1]+cvSizeu[i-1];
  }
  
  lidx<-Xl_sIdx[j]:(Xl_sIdx[j]+cvSizel[j]-1)
  uidx<-Xu_sIdx[j]:(Xu_sIdx[j]+cvSizeu[j]-1)
  
  testXl<-pXl[lidx,]
  trainXl<-pXl[-lidx,]
  testXu<-pXu[uidx,]
  trainXu<-pXu[-uidx,]
  
  X_lu_t<-rbind(trainXl,trainXu)
  z_lu_t<-c(rep(1,nl-cvSizel[j]),rep(0,nu-cvSizeu[j]))
  X_lu_v<-rbind(testXl,testXu)
  z_lu_v<-c(rep(1,cvSizel[j]),rep(0,cvSizeu[j]))
  return(list(pX_lu=rbind(pXl,pXu),X_lu_t=X_lu_t,z_lu_t=z_lu_t,X_lu_v=X_lu_v,z_lu_v=z_lu_v))
}

s<-sample(1:2000)
X=simulPU$X
z=simulPU$z
truePrevalence=simulPU$truePY1
X=X[s,]
z=z[s]
spX=spX[s,]

##################################################################################################
context("Input : Dense matrix")
gn=grpPUlasso(X=X,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
cv.gn=cv.grpPUlasso(X=X,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
a=cvXrecover(cv.gn,X,z,j=1)
cvgn1=grpPUlasso(a$X_lu_t,a$z_lu_t,pi=truePrevalence,lambda=cv.gn$lambda,eps = 1e-06)

gn.gd <-grpPUlasso(X=X,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3,
                         eps=1e-6,method = "GD",verbose = T,stepSizeAdjustment = 50)

gn.svrg<-grpPUlasso(X=X,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, 
                               eps=1e-6,method = "SVRG",verbose=F,stepSizeAdjustment = 1/4)


test_that("Input : Dense matrix", {
  expect_lt(max(abs(gn$coef-gn.prev.coef)),1e-4) #comparison with solution from previous EM
  expect_lt(max(cv.gn$PUfit$std_coef-gn$std_coef),1e-4) # cv PUfit == PUfit
  expect_lt(max(abs(cvgn1$coef-cv.gn$cvcoef$cv1)),1e-4) # CV check
  expect_lt(max(abs(gn.gd$coef-gn$coef)),1e-4) #check GD
  expect_lt(max(abs(gn.svrg$coef-gn$coef)),1e-2) #check SVRG
})
##################################################################################################
context("Input : Sparse matrix")
spgn=grpPUlasso(X = spX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
spgnd=grpPUlasso(X = as.matrix(spX),z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
cv.spgn = cv.grpPUlasso(X=spX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
a=cvXrecover(cv.spgn,spX,z,j=1)
cvspgn1=grpPUlasso(a$X_lu_t,a$z_lu_t,pi=truePrevalence,lambda=cv.spgn$lambda,eps = 1e-06)

spgn.gd=grpPUlasso(X=spX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, 
                   eps=1e-6,method = "GD",verbose = T,stepSizeAdjustment = 50)
spgnd.gd=grpPUlasso(X=as.matrix(spX),z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, 
                   eps=1e-6,method = "GD",verbose = T,stepSizeAdjustment = 50)

# spgn.svrg=grpPUlasso(X=spX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, 
#                      eps=1e-8,method = "SVRG",verbose = T)
# spgnd.svrg=grpPUlasso(X=as.matrix(spX),z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3,
#                       eps=1e-8,method = "SVRG",verbose = T)
# 
# max(abs(spgn.svrg$std_coef-spgnd.svrg$std_coef))<1e-4

test_that("Input : Sparse matrix",{
  expect_lt(max(abs(spgn$coef-spgn.prev.coef)),1e-4)
  expect_lt(max(spgn$coef-spgnd$coef),1e-4)
  expect_lt(max(cv.spgn$PUfit$std_coef-spgn$std_coef),1e-4)
  expect_lt(max(abs(cvspgn1$coef-cv.spgn$cvcoef$cv1)),1e-4)
  expect_lt(max(abs(spgn.gd$coef-spgnd.gd$coef)),1e-4)
})

##################################################################################################
context("Deviance")
test_that("Deviance",{
  expect_lt(max(gn$deviance-deviances(X=X,z=z,pi=truePrevalence,coefMat = gn$coef)),1e-4)
  expect_lt(max(spgn$deviance-deviances(X=spX,z=z,pi=truePrevalence,coefMat = spgn$coef)),1e-4)
  expect_lt(max(gn.gd$deviance-deviances(X=X,z=z,pi=truePrevalence,coefMat = gn.gd$coef)),1e-4)
})
##################################################################################################
