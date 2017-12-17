library(PUlasso)
library(testthat)

data("simulPU")
load("sysdata.rda")
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

test_that("Input : Dense matrix", {
  expect_lt(max(abs(gn$coef-gn.prev.coef)),1e-4)
  expect_lt(max(cv.gn$PUfit$std_coef-gn$std_coef),1e-4)
})
##################################################################################################
context("Input : Sparse matrix")
spgn=grpPUlasso(X = spX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
spgnd=grpPUlasso(X = as.matrix(spX),z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
cv.spgn = cv.grpPUlasso(X=spX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)

test_that("Input : Sparse matrix",{
  expect_lt(max(abs(spgn$coef-spgn.prev.coef)),1e-4)
  expect_lt(max(spgn$coef-spgnd$coef),1e-4)
  expect_lt(max(cv.spgn$PUfit$std_coef-spgn$std_coef),1e-4)
})

##################################################################################################
context("Input : Big matrix")
bmX = as.big.matrix(X)
before=bmX[,]
gnb<-grpPUlasso(X=bmX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
after=bmX[,]
before.cv=after
cv.gnb = cv.grpPUlasso(X=bmX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-3, eps=1e-06)
after.cv=bmX[,]
test_that("Input : Big matrix",{
  expect_lt(max(abs(gn$coef-gnb$coef)),1e-4)
  expect_lt(max(cv.gnb$PUfit$coef-gnb$coef),1e-4)
  expect_equal(max(abs(after-before)),0)
  expect_equal(max(abs(after.cv-before.cv)),0)
})

##################################################################################################
context("Deviance")
test_that("Deviance",{
  expect_lt(max(gn$deviance-deviances(X=X,z=z,pi=truePrevalence,coefMat = gn$coef)),1e-5)
  expect_lt(max(spgn$deviance-deviances(X=spX,z=z,pi=truePrevalence,coefMat = spgn$coef)),1e-5)
  expect_lt(max(gnb$deviance-deviances(X=bmX,z=z,pi=truePrevalence,coefMat = gn$coef)),1e-5)
})
##################################################################################################
