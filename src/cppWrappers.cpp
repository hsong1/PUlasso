#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include <iostream>
#include <vector>
#include <set>
#include <RcppEigen.h>
#include <string>
#include "groupLasso.h"
#include "LUfit.h"
#include "cv_LUfit.h"
//#include <bigmemory/MatrixAccessor.hpp>
#include <Rcpp.h>
using namespace Rcpp;
using namespace Eigen;

using Rcpp::as;

//[[Rcpp::export]]
Rcpp::List LU_dense_cpp(Eigen::Map<Eigen::MatrixXd> X_, Eigen::VectorXd & z_, Eigen::VectorXd & icoef_,
                        Eigen::ArrayXd & gsize_,Eigen::ArrayXd & pen_,
                        Eigen::ArrayXd & lambdaseq_,bool user_lambdaseq_,
                        int pathLength_,double lambdaMinRatio_, double pi_,
                        int maxit_,double tol_,double inner_tol_,
                        bool useStrongSet_, bool verbose_)
{
  try{
    LUfit<Eigen::Map<Eigen::MatrixXd> > lu(X_,z_,icoef_,gsize_,pen_,
                                           lambdaseq_,user_lambdaseq_,pathLength_,
                                           lambdaMinRatio_,pi_,maxit_,tol_,
                                           inner_tol_,useStrongSet_,verbose_);
    lu.LUfit_main();
    lu.decenterX();
    
    return Rcpp::List::create(Rcpp::Named("coef") = lu.getCoefficients(),
                              Rcpp::Named("std_coef") = lu.getStdCoefficients(),
                              Rcpp::Named("iters") = lu.getIters(),
                              Rcpp::Named("nUpdates") = lu.getnUpdates(),
                              Rcpp::Named("nullDev") = lu.getnullDev(),
                              Rcpp::Named("deviance") = lu.getDeviances(),
                              Rcpp::Named("lambda")=lu.getLambdaSequence(),
                              Rcpp::Named("convFlag")=lu.getconvFlag());
  }
  catch(const std::invalid_argument& e ){
    throw std::range_error(e.what());
  }
  return R_NilValue;
}

//[[Rcpp::export]]
Rcpp::List LU_sparse_cpp(Eigen::SparseMatrix<double> & X_, Eigen::VectorXd & z_, Eigen::VectorXd & icoef_,
                         Eigen::ArrayXd & gsize_,Eigen::ArrayXd & pen_,
                         Eigen::ArrayXd & lambdaseq_,bool user_lambdaseq_,
                         int pathLength_,double lambdaMinRatio_, double pi_,
                         int maxit_,double tol_,double inner_tol_,
                         bool useStrongSet_, bool verbose_)
{
  try{
    LUfit<Eigen::SparseMatrix<double> > lu(X_,z_,icoef_,gsize_,pen_,
                                           lambdaseq_,user_lambdaseq_,pathLength_,
                                           lambdaMinRatio_,pi_,maxit_,tol_,
                                           inner_tol_,useStrongSet_,verbose_);
    lu.LUfit_main();
    
    return Rcpp::List::create(Rcpp::Named("coef") = lu.getCoefficients(),
                              Rcpp::Named("std_coef") = lu.getStdCoefficients(),
                              Rcpp::Named("iters") = lu.getIters(),
                              Rcpp::Named("nUpdates") = lu.getnUpdates(),
                              Rcpp::Named("nullDev") = lu.getnullDev(),
                              Rcpp::Named("deviance") = lu.getDeviances(),
                              Rcpp::Named("lambda")=lu.getLambdaSequence(),
                              Rcpp::Named("convFlag")=lu.getconvFlag());
  }
  catch(const std::invalid_argument& e ){
    throw std::range_error(e.what());
  }
  return R_NilValue;
}


//[[Rcpp::plugins(openmp)]]
//[[Rcpp::export]]
Rcpp::List cv_LU_dense_cpp(Eigen::Map<Eigen::MatrixXd> X_, Eigen::VectorXd & z_, Eigen::VectorXd & icoef_,
                           Eigen::ArrayXd & gsize_,Eigen::ArrayXd & pen_,
                           Eigen::ArrayXd & lambdaseq_,bool user_lambdaseq_,
                           int pathLength_,double lambdaMinRatio_, double pi_,
                           int maxit_,double tol_,double inner_tol_,
                           bool useStrongSet_, bool verbose_, int nfolds_, int nfits_,int ncores_)
{
   try{
     
    cv_LUfit<Eigen::Map<Eigen::MatrixXd> > cvlu(X_,z_,icoef_,gsize_,pen_,
                                                lambdaseq_,user_lambdaseq_,pathLength_,
                                                lambdaMinRatio_,pi_,maxit_,tol_,
                                                inner_tol_,useStrongSet_,verbose_,nfolds_,nfits_,ncores_);
    cvlu.cv_LUfit_main();
    
    LUfit<Eigen::Map<MatrixXd> > lu_f = cvlu.getlu_f();
    return Rcpp::List::create(Rcpp::Named("nullDev")  = cvlu.getnullDev(),
                              Rcpp::Named("deviance") = cvlu.getDeviances(),
                              Rcpp::Named("coef") = cvlu.getCoefMat(),
                              Rcpp::Named("std_coef") = cvlu.getStdCoefMat(),
                              Rcpp::Named("lambda")   = cvlu.getLambdaSequence(),
                              Rcpp::Named("convFlagMat")   = cvlu.getconvFlagMat(),
                              Rcpp::Named("f_coef") = lu_f.getCoefficients(),
                              Rcpp::Named("f_std_coef") = lu_f.getStdCoefficients(),
                              Rcpp::Named("f_iters") = lu_f.getIters(),
                              Rcpp::Named("f_nUpdates") = lu_f.getnUpdates(),
                              Rcpp::Named("f_nullDev") = lu_f.getnullDev(),
                              Rcpp::Named("f_deviance") = lu_f.getDeviances(),
                              Rcpp::Named("f_lambda") = lu_f.getLambdaSequence(),
                              Rcpp::Named("f_convFlag")= lu_f.getconvFlag());
  }
  catch(const std::invalid_argument& e ){
    throw std::range_error(e.what());
  }
  return R_NilValue;
}

//[[Rcpp::plugins(openmp)]]
//[[Rcpp::export]]
Rcpp::List cv_LU_sparse_cpp(Eigen::SparseMatrix<double> & X_, Eigen::VectorXd & z_, Eigen::VectorXd & icoef_,
                            Eigen::ArrayXd & gsize_,Eigen::ArrayXd & pen_,
                            Eigen::ArrayXd & lambdaseq_,bool user_lambdaseq_,
                            int pathLength_,double lambdaMinRatio_, double pi_,
                            int maxit_,double tol_,double inner_tol_,
                            bool useStrongSet_, bool verbose_, int nfolds_, int nfits_,int ncores_)
{
  try{
    cv_LUfit<Eigen::SparseMatrix<double> > cvlu(X_,z_,icoef_,gsize_,pen_,
                                                lambdaseq_,user_lambdaseq_,pathLength_,
                                                lambdaMinRatio_,pi_,maxit_,tol_,
                                                inner_tol_,useStrongSet_,verbose_,nfolds_,nfits_,ncores_);
    cvlu.cv_LUfit_main();
    LUfit<Eigen::SparseMatrix<double> > lu_f = cvlu.getlu_f();
    return Rcpp::List::create(Rcpp::Named("nullDev")  = cvlu.getnullDev(),
                              Rcpp::Named("deviance") = cvlu.getDeviances(),
                              Rcpp::Named("coef") = cvlu.getCoefMat(),
                              Rcpp::Named("std_coef") = cvlu.getStdCoefMat(),
                              Rcpp::Named("lambda")   = cvlu.getLambdaSequence(),
                              Rcpp::Named("convFlagMat")   = cvlu.getconvFlagMat(),
                              Rcpp::Named("f_coef") = lu_f.getCoefficients(),
                              Rcpp::Named("f_std_coef") = lu_f.getStdCoefficients(),
                              Rcpp::Named("f_iters") = lu_f.getIters(),
                              Rcpp::Named("f_nUpdates") = lu_f.getnUpdates(),
                              Rcpp::Named("f_nullDev") = lu_f.getnullDev(),
                              Rcpp::Named("f_deviance") = lu_f.getDeviances(),
                              Rcpp::Named("f_lambda") = lu_f.getLambdaSequence(),
                              Rcpp::Named("f_convFlag")= lu_f.getconvFlag());
  }
  catch(const std::invalid_argument& e ){
    throw std::range_error(e.what());
  }
  return R_NilValue;
}


//[[Rcpp::export]]
Eigen::MatrixXd deviances_dense_cpp(Eigen::MatrixXd & coefMat_, Eigen::Map<Eigen::MatrixXd> & X_, Eigen::VectorXd & z_, double pi_)
{
  int K = coefMat_.cols();
  VectorXd deviances(K);
  
  for (int j=0;j<K;j++)
  {
    deviances(j) = evalDeviance(X_,z_,pi_,coefMat_.middleCols(j,1));
  }
  return deviances;
}

//[[Rcpp::export]]
Eigen::MatrixXd deviances_sparse_cpp(Eigen::MatrixXd & coefMat_, Eigen::SparseMatrix<double> & X_, Eigen::VectorXd & z_, double pi_)
{
  int K = coefMat_.cols();
  VectorXd deviances(K);
  
  for (int j=0;j<K;j++)
  {
    deviances(j) = evalDeviance(X_,z_,pi_,coefMat_.middleCols(j,1));
  }
  return deviances;
}


