#ifndef cv_LUfit_h
#define cv_LUfit_h
#include "LUfit.h"
#include "omp.h"
using namespace Eigen;
template <class TX>
class cv_LUfit
{
protected:
  TX X;// with intercept, N by p matrix, p = 1+k1+..+k(J-1)
  VectorXd z;// size N
  VectorXd icoef;
  ArrayXd gsize;// size J, first group = intercept
  ArrayXd pen; // size J, first element = 0;
  ArrayXd lambdaseq;//size K, default 100
  bool isUserLambdaseq;
  int pathLength;
  double lambdaMinRatio;
  double pi;
  int maxit;
  double tol;
  double inner_tol;
  bool useStrongSet;
  bool verbose;
  int nfolds;
  int nfits;
  int ncores;
  LUfit<TX> lu_f;
  
  int N,p,K,nl,nu;
  //    ArrayXi pl;
  //    ArrayXi pu;
  ArrayXi cvSizel;
  ArrayXi cvSizeu;
  //    std::vector<TX> pX_ls;
  //    std::vector<TX> pX_us;
  ArrayXi Xl_sIdx;
  ArrayXi Xu_sIdx;
  VectorXd nullDev;
  MatrixXd Deviances;
  MatrixXd coefMat;
  MatrixXd std_coefMat;
  MatrixXi convFlagMat;
  
public:
  //TX X_lu_t;//Training Sets
  //VectorXd z_lu_t;
  //TX X_lu_v;//Validation Sets
  //VectorXd z_lu_v;
  
  cv_LUfit(const TX & X_, VectorXd & z_, VectorXd & icoef_, ArrayXd & gsize_,ArrayXd & pen_,ArrayXd & lambdaseq_, bool isUserLambdaseq_,int pathLength_,double lambdaMinRatio_,double pi_, int maxit_, double tol_, double inner_tol_,bool useStrongSet_,bool verbose_,int nfolds_,int nfits_,int ncores_);
  void s_setup_t(SparseMatrix<double> & X_lu_t, VectorXd & z_lu_t, int j);
  void s_setup_v(SparseMatrix<double> & X_lu_v, VectorXd & z_lu_v, int j);
  void d_setup_t(MatrixXd & X_lu_t, VectorXd & z_lu_t, int j);
  void d_setup_v(MatrixXd & X_lu_v, VectorXd & z_lu_v, int j);
  
  void cv_LUfit_main();
  LUfit<TX> getlu_f();
  //    ArrayXi getpl();
  //    ArrayXi getpu();
  MatrixXd getnullDev();
  MatrixXd getDeviances();
  MatrixXd getCoefMat();
  MatrixXd getStdCoefMat();
  ArrayXd getLambdaSequence();
  MatrixXi getconvFlagMat();
  void checkDesignMatrix(const TX & X);
  
};

#endif /* cv_LUfit_h */
