#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include <iostream>
#include <vector>
#include <set>
#include <RcppEigen.h>
#include <string>
#include "groupLasso.h"
#include "LUfit.h"
// #include "cv_LUfit.h"
#include "pgLUfit.h"
// #include "cv_pgLUfit.h"
#include <Rcpp.h>
using namespace Rcpp;
using namespace Eigen;
using Rcpp::as;

// //' @export
//[[Rcpp::export]]
Rcpp::List LU_dense_cpp(Eigen::Map<Eigen::MatrixXd> X_, Eigen::VectorXd & z_, Eigen::VectorXd & icoef_,
                        Eigen::ArrayXd & gsize_,Eigen::ArrayXd & pen_,
                        Eigen::ArrayXd & lambdaseq_,bool user_lambdaseq_,
                        int pathLength_,double lambdaMinRatio_, double pi_,
                        int maxit_,double tol_,double inner_tol_,
                        bool useStrongSet_, bool verbose_, double stepSize_,double stepSizeAdj_,int batchSize_,
                        std::vector<double> samplingProbabilities_,bool useLipschitz_,std::string method_,int trace_)
{
  
  try{
    
    if(method_=="CD"){
      LUfit<Eigen::Map<Eigen::MatrixXd> > lu(X_,z_,icoef_,gsize_,pen_,
                                             lambdaseq_,user_lambdaseq_,pathLength_,
                                             lambdaMinRatio_,pi_,maxit_,tol_,
                                             inner_tol_,useStrongSet_,verbose_,trace_);
      lu.LUfit_main();
      return Rcpp::List::create(Rcpp::Named("coef") = lu.getCoefficients(),
                                Rcpp::Named("std_coef") = lu.getStdCoefficients(),
                                Rcpp::Named("iters") = lu.getIters(),
                                Rcpp::Named("nUpdates") = lu.getnUpdates(),
                                Rcpp::Named("nullDev") = lu.getnullDev(),
                                Rcpp::Named("deviance") = lu.getDeviances(),
                                Rcpp::Named("lambda")=lu.getLambdaSequence(),
                                Rcpp::Named("convFlag")=lu.getconvFlag(),
                                Rcpp::Named("fVals") = lu.getfVals(),
                                Rcpp::Named("subgrads") = lu.getSubGradients(),
                                Rcpp::Named("fVals_all") = lu.getfVals_all(),
                                Rcpp::Named("beta_all") = lu.getbeta_all(),
                                Rcpp::Named("method") = method_);
      
    }else{
      pgLUfit<Eigen::Map<MatrixXd> > lu(X_,z_,icoef_,gsize_,pen_,lambdaseq_,user_lambdaseq_,pathLength_,
                                        lambdaMinRatio_,pi_,maxit_,tol_,verbose_,
                                        stepSize_,stepSizeAdj_,batchSize_,samplingProbabilities_,useLipschitz_,method_,trace_);
      lu.pgLUfit_main();
      return Rcpp::List::create(Rcpp::Named("coef") = lu.getCoefficients(),
                                Rcpp::Named("std_coef") = lu.getStdCoefficients(),
                                Rcpp::Named("iters") = lu.getIters(),
                                Rcpp::Named("nullDev") = lu.getnullDev(),
                                Rcpp::Named("deviance") = lu.getDeviances(),
                                Rcpp::Named("lambda")=lu.getLambdaSequence(),
                                Rcpp::Named("convFlag")=lu.getconvFlag(),
                                Rcpp::Named("fVals") = lu.getfVals(),
                                Rcpp::Named("subgrads") = lu.getSubGradients(),
                                Rcpp::Named("fVals_all") = lu.getfVals_all(),
                                Rcpp::Named("beta_all") = lu.getbeta_all(),
                                Rcpp::Named("stepSize") = lu.getStepSize(),
                                Rcpp::Named("samplingProbabilities")=lu.getSamplingProbabilities(),
                                Rcpp::Named("method") = method_);
      // return R_NilValue;
    }
  }
  catch(const std::invalid_argument& e ){
    throw std::range_error(e.what());
  }
  return R_NilValue;
}

// //' @export
//[[Rcpp::export]]
Rcpp::List LU_sparse_cpp(Eigen::SparseMatrix<double> & X_, Eigen::VectorXd & z_, Eigen::VectorXd & icoef_,
                         Eigen::ArrayXd & gsize_,Eigen::ArrayXd & pen_,
                         Eigen::ArrayXd & lambdaseq_,bool user_lambdaseq_,
                         int pathLength_,double lambdaMinRatio_, double pi_,
                         int maxit_,double tol_,double inner_tol_,
                         bool useStrongSet_, bool verbose_,double stepSize_,double stepSizeAdj_,int batchSize_,
                         std::vector<double> samplingProbabilities_,bool useLipschitz_,std::string method_,int trace_)
{
  try{
    
    if(method_=="CD"){
      LUfit<Eigen::SparseMatrix<double> > lu(X_,z_,icoef_,gsize_,pen_,
                                             lambdaseq_,user_lambdaseq_,pathLength_,
                                             lambdaMinRatio_,pi_,maxit_,tol_,
                                             inner_tol_,useStrongSet_,verbose_,trace_);
      lu.LUfit_main();
      return Rcpp::List::create(Rcpp::Named("coef") = lu.getCoefficients(),
                                Rcpp::Named("std_coef") = lu.getStdCoefficients(),
                                Rcpp::Named("iters") = lu.getIters(),
                                Rcpp::Named("nUpdates") = lu.getnUpdates(),
                                Rcpp::Named("nullDev") = lu.getnullDev(),
                                Rcpp::Named("deviance") = lu.getDeviances(),
                                Rcpp::Named("lambda")=lu.getLambdaSequence(),
                                Rcpp::Named("convFlag")=lu.getconvFlag(),
                                Rcpp::Named("fVals") = lu.getfVals(),
                                Rcpp::Named("subgrads") = lu.getSubGradients(),
                                Rcpp::Named("fVals_all") = lu.getfVals_all(),
                                Rcpp::Named("beta_all") = lu.getbeta_all(),
                                Rcpp::Named("method") = method_);
      
    }else{
      pgLUfit<Eigen::SparseMatrix<double> > lu(X_,z_,icoef_,gsize_,pen_,lambdaseq_,user_lambdaseq_,pathLength_,
                                        lambdaMinRatio_,pi_,maxit_,tol_,verbose_,
                                        stepSize_,stepSizeAdj_,batchSize_,samplingProbabilities_,useLipschitz_,method_,trace_);
      lu.pgLUfit_main();
      return Rcpp::List::create(Rcpp::Named("coef") = lu.getCoefficients(),
                                Rcpp::Named("std_coef") = lu.getStdCoefficients(),
                                Rcpp::Named("iters") = lu.getIters(),
                                Rcpp::Named("nullDev") = lu.getnullDev(),
                                Rcpp::Named("deviance") = lu.getDeviances(),
                                Rcpp::Named("lambda")=lu.getLambdaSequence(),
                                Rcpp::Named("convFlag")=lu.getconvFlag(),
                                Rcpp::Named("fVals") = lu.getfVals(),
                                Rcpp::Named("subgrads") = lu.getSubGradients(),
                                Rcpp::Named("fVals_all") = lu.getfVals_all(),
                                Rcpp::Named("beta_all") = lu.getbeta_all(),
                                Rcpp::Named("stepSize") = lu.getStepSize(),
                                Rcpp::Named("samplingProbabilities")=lu.getSamplingProbabilities(),
                                Rcpp::Named("method") = method_);
      // return R_NilValue;
    }
  }
  catch(const std::invalid_argument& e ){
    throw std::range_error(e.what());
  }
  return R_NilValue;
}


// //[[Rcpp::plugins(openmp)]]
// //' @export
// //[[Rcpp::export]]
// Rcpp::List cv_LU_dense_cpp(Eigen::Map<Eigen::MatrixXd> X_, Eigen::VectorXd & z_, Eigen::VectorXd & icoef_,
//                            Eigen::ArrayXd & gsize_,Eigen::ArrayXd & pen_,
//                            Eigen::ArrayXd & lambdaseq_,bool user_lambdaseq_,
//                            int pathLength_,double lambdaMinRatio_, double pi_,
//                            int maxit_,double tol_,double inner_tol_,
//                            bool useStrongSet_, bool verbose_, double stepSize_,double stepSizeAdj_,int batchSize_,
//                            std::vector<double> samplingProbabilities_,bool useLipschitz_,std::string method_,int nfolds_, int nfits_,int ncores_,bool trace_)
// {
//    try{
//     if(method_=="CD"){
//       cv_LUfit<Eigen::Map<Eigen::MatrixXd> > cvlu(X_,z_,icoef_,gsize_,pen_,
//                                                   lambdaseq_,user_lambdaseq_,pathLength_,
//                                                   lambdaMinRatio_,pi_,maxit_,tol_,
//                                                   inner_tol_,useStrongSet_,verbose_,nfolds_,nfits_,ncores_,trace_);
//       cvlu.cv_LUfit_main();
//       
//       LUfit<Eigen::Map<MatrixXd> > lu_f = cvlu.getlu_f();
//       return Rcpp::List::create(Rcpp::Named("nullDev")  = cvlu.getnullDev(),
//                                 Rcpp::Named("deviance") = cvlu.getDeviances(),
//                                 Rcpp::Named("coef") = cvlu.getCoefMat(),
//                                 Rcpp::Named("std_coef") = cvlu.getStdCoefMat(),
//                                 Rcpp::Named("lambda")   = cvlu.getLambdaSequence(),
//                                 Rcpp::Named("convFlagMat")   = cvlu.getconvFlagMat(),
//                                 Rcpp::Named("f_coef") = lu_f.getCoefficients(),
//                                 Rcpp::Named("f_std_coef") = lu_f.getStdCoefficients(),
//                                 Rcpp::Named("f_iters") = lu_f.getIters(),
//                                 Rcpp::Named("f_nUpdates") = lu_f.getnUpdates(),
//                                 Rcpp::Named("f_nullDev") = lu_f.getnullDev(),
//                                 Rcpp::Named("f_deviance") = lu_f.getDeviances(),
//                                 Rcpp::Named("f_lambda") = lu_f.getLambdaSequence(),
//                                 Rcpp::Named("f_convFlag")= lu_f.getconvFlag(),
//                                 Rcpp::Named("f_fVals") = lu_f.getfVals(),
//                                 Rcpp::Named("f_subgrads") = lu_f.getSubGradients(),
//                                 Rcpp::Named("f_fVals_all") = lu_f.getfVals_all(),
//                                 Rcpp::Named("method") = method_);
//     }else{
//       cv_pgLUfit<Eigen::Map<Eigen::MatrixXd> > cvlu(X_,z_,icoef_,gsize_,pen_,
//                                                   lambdaseq_,user_lambdaseq_,pathLength_,
//                                                   lambdaMinRatio_,pi_,maxit_,tol_,
//                                                   verbose_,stepSize_,stepSizeAdj_,batchSize_,
//                                                   samplingProbabilities_,useLipschitz_,method_,nfolds_,nfits_,ncores_,trace_);
//       cvlu.cv_pgLUfit_main();
//       
//       pgLUfit<Eigen::Map<MatrixXd> > lu_f = cvlu.getlu_f();
//       return Rcpp::List::create(Rcpp::Named("nullDev")  = cvlu.getnullDev(),
//                                 Rcpp::Named("deviance") = cvlu.getDeviances(),
//                                 Rcpp::Named("coef") = cvlu.getCoefMat(),
//                                 Rcpp::Named("std_coef") = cvlu.getStdCoefMat(),
//                                 Rcpp::Named("lambda")   = cvlu.getLambdaSequence(),
//                                 Rcpp::Named("convFlagMat")   = cvlu.getconvFlagMat(),
//                                 Rcpp::Named("f_coef") = lu_f.getCoefficients(),
//                                 Rcpp::Named("f_std_coef") = lu_f.getStdCoefficients(),
//                                 Rcpp::Named("f_iters") = lu_f.getIters(),
//                                 Rcpp::Named("f_nullDev") = lu_f.getnullDev(),
//                                 Rcpp::Named("f_deviance") = lu_f.getDeviances(),
//                                 Rcpp::Named("f_lambda") = lu_f.getLambdaSequence(),
//                                 Rcpp::Named("f_convFlag")= lu_f.getconvFlag(),
//                                 Rcpp::Named("f_fVals") = lu_f.getfVals(),
//                                 Rcpp::Named("f_subgrads") = lu_f.getSubGradients(),
//                                 Rcpp::Named("f_fVals_all") = lu_f.getfVals_all(),
//                                 Rcpp::Named("method") = method_);
//     }
//     
//   }
//   catch(const std::invalid_argument& e ){
//     throw std::range_error(e.what());
//   }
//   return R_NilValue;
// }
// 
// 
// //[[Rcpp::plugins(openmp)]]
// //' @export
// //[[Rcpp::export]]
// Rcpp::List cv_LU_sparse_cpp(Eigen::SparseMatrix<double> & X_, Eigen::VectorXd & z_, Eigen::VectorXd & icoef_,
//                             Eigen::ArrayXd & gsize_,Eigen::ArrayXd & pen_,
//                             Eigen::ArrayXd & lambdaseq_,bool user_lambdaseq_,
//                             int pathLength_,double lambdaMinRatio_, double pi_,
//                             int maxit_,double tol_,double inner_tol_,
//                             bool useStrongSet_, bool verbose_, double stepSize_,double stepSizeAdj_,int batchSize_,
//                             std::vector<double> samplingProbabilities_,bool useLipschitz_,std::string method_,int nfolds_, int nfits_,int ncores_,bool trace_)
// {
//   try{
//     cv_LUfit<Eigen::SparseMatrix<double> > cvlu(X_,z_,icoef_,gsize_,pen_,
//                                                 lambdaseq_,user_lambdaseq_,pathLength_,
//                                                 lambdaMinRatio_,pi_,maxit_,tol_,
//                                                 inner_tol_,useStrongSet_,verbose_,nfolds_,nfits_,ncores_,trace_);
//     cvlu.cv_LUfit_main();
//     LUfit<Eigen::SparseMatrix<double> > lu_f = cvlu.getlu_f();
//     return Rcpp::List::create(Rcpp::Named("nullDev")  = cvlu.getnullDev(),
//                               Rcpp::Named("deviance") = cvlu.getDeviances(),
//                               Rcpp::Named("coef") = cvlu.getCoefMat(),
//                               Rcpp::Named("std_coef") = cvlu.getStdCoefMat(),
//                               Rcpp::Named("lambda")   = cvlu.getLambdaSequence(),
//                               Rcpp::Named("convFlagMat")   = cvlu.getconvFlagMat(),
//                               Rcpp::Named("f_coef") = lu_f.getCoefficients(),
//                               Rcpp::Named("f_std_coef") = lu_f.getStdCoefficients(),
//                               Rcpp::Named("f_iters") = lu_f.getIters(),
//                               Rcpp::Named("f_nUpdates") = lu_f.getnUpdates(),
//                               Rcpp::Named("f_nullDev") = lu_f.getnullDev(),
//                               Rcpp::Named("f_deviance") = lu_f.getDeviances(),
//                               Rcpp::Named("f_lambda") = lu_f.getLambdaSequence(),
//                               Rcpp::Named("f_convFlag")= lu_f.getconvFlag(),
//                               Rcpp::Named("f_fVals") = lu_f.getfVals(),
//                               Rcpp::Named("f_subgrads") = lu_f.getSubGradients(),
//                               Rcpp::Named("f_fVals_all") = lu_f.getfVals_all(),
//                               Rcpp::Named("method") = method_);
//   }
//   catch(const std::invalid_argument& e ){
//     throw std::range_error(e.what());
//   }
//   return R_NilValue;
// }


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


