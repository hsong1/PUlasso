library(PUlasso)
library(bigmemory)
context("PUlasso fits models as expected")
data("simulatedPUdata")
load("sysdata.Rdata")
s<-sample(1:2000)
X=simulatedPUdata$X
z=simulatedPUdata$z
truePrevalence=simulatedPUdata$truePrevalence
X=X[s,]
z=z[s]
spX=spX[s,]

gn.prev.coef<-
  rbind(sapply(g.prev,function(x){x$a0}),
  sapply(g.prev,function(x){x$beta}))

spgn.prev.coef<-
  rbind(sapply(spg.prev,function(x){x$a0}),
        sapply(spg.prev,function(x){x$beta}))

##################################################################################################

gn=grpPUlasso(X=X,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-1, eps=1e-06)
cv.gn=cv.grpPUlasso(X=X,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-1, eps=1e-06)

test_that("Input = Dense matrix", {
  expect_lt(max(abs(gn$coef-gn.prev.coef)),1e-4)
  expect_lt(max(cv.gn$PUfit$std_coef-cv.g$PUfit$std_coef),1e-4)
})

##################################################################################################

spgn=grpPUlasso(X = spX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-1, eps=1e-06)
spgnd=grpPUlasso(X = as.matrix(spX),z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-1, eps=1e-06)
cv.spgn = cv.grpPUlasso(X=spX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-1, eps=1e-06)

test_that("Input = Sparse matrix",{
  expect_lt(max(abs(spgn$coef-spgn.prev.coef)),1e-4)
  expect_lt(max(spgn$coef-spgnd$coef),1e-4)
  expect_lt(max(cv.spgn$PUfit$std_coef-spgn$std_coef),1e-4)
})

##################################################################################################

bmX = as.big.matrix(X)
before=bmX[,]
gnb<-grpPUlasso(X=bmX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-1, eps=1e-06)
after=bmX[,]
before.cv=after
cv.gnb = cv.grpPUlasso(X=bmX,z=z,pi=truePrevalence,nlambda = 5,lambdaMinRatio = 1e-1, eps=1e-06)
after.cv=bmX[,]
test_that("Input = Big matrix",{
  expect_lt(max(abs(gn$coef-gnb$coef)),1e-4)
  expect_lt(max(cv.gnb$PUfit$coef-gnb$coef),1e-4)
  expect_equal(max(abs(after-before)),0)
  expect_equal(max(abs(after.cv-before.cv)),0)
})

##################################################################################################
