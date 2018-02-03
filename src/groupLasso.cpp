#include "groupLasso.h"
using namespace Eigen;

//Constructor
template <class TX>
groupLassoFit<TX>::groupLassoFit(TX & X_, VectorXd & y_, VectorXd & icoef_, ArrayXd & gsize_,ArrayXd & pen_,ArrayXd & lambdaseq_, bool isUserLambdaseq_,int pathLength_,double lambdaMinRatio_,int maxit_, double tol_, bool verbose_)
  :X(X_),y(y_), gsize(gsize_), pen(pen_),lambdaseq(lambdaseq_), isUserLambdaseq(isUserLambdaseq_),pathLength(pathLength_),lambdaMinRatio(lambdaMinRatio_),maxit(maxit_), tol(tol_),verbose(verbose_),iter(0),resid(y_),converged_CD(false),converged_KKT(false)
{
  checkDesignMatrix(X);
  N = static_cast<int>(X.rows());
  p = static_cast<int>(X.cols())+1;
  J = static_cast<int>(gsize.size());
  K = isUserLambdaseq?(static_cast<int>(lambdaseq.size())):(pathLength);
  
  grpSIdx=ArrayXi::Zero(J);
  
  for(int ii=2;ii<J;++ii)
  {
    grpSIdx(ii)=grpSIdx(ii-1)+gsize(ii-1);
  }
  
  iters = ArrayXi::Zero(K);
  coefficients = MatrixXd::Zero(p, K);
  std_coefficients = MatrixXd::Zero(p, K);
  
  //Calculate Xcenter, Rinvs
  //For a dense class X, X = P0X, Sparse or Map class, no change in X
  
  Xcenter = VectorXd::Ones(p-1);
  Rinvs.resize(J);
  if(verbose){Rcpp::Rcout<<"QR decompositions\n";}
  Rinvs_X();
  
  //Initialize beta
  beta =org_to_std(icoef_);
  
  //Initialize gradient
  g.resize(J);
  
  //Initialize active/inactive set
  //inactiveSet={1,..,J-1}
  activeSet.clear();
  for(int j=1;j<J;j++)
  {
    if(beta[j]==0) {
      inactiveSet.insert(j);
    } else {
      activeSet.insert(j);
    }
  }
  inactiveSet1.clear();
  inactiveSet2.clear();
  convFlag.resize(K);
  convFlag.setZero();
  
  if(verbose){Rcpp::Rcout<<"end of construction\n";}
}

//Getters
template <typename TX>
MatrixXd groupLassoFit<TX>::getCoefficients(){return coefficients;}

template <typename TX>
MatrixXd groupLassoFit<TX>::getStdCoefficients(){return std_coefficients;}

template <typename TX>
ArrayXi groupLassoFit<TX>::getIters(){return iters;}

template <typename TX>
ArrayXd groupLassoFit<TX>::getLambdaSequence(){return lambdaseq;}

template <typename TX>
ArrayXi groupLassoFit<TX>::getconvFlag(){return convFlag;}

//Misc functions
template <typename TX>
VectorXd groupLassoFit<TX>::back_to_org(const VectorXd & beta)
{
  VectorXd gamma(beta);
  
  for(int j=1;j<J;++j)
  {
    gamma.segment(grpSIdx(j)+1,gsize(j))= Rinvs[j]*beta.segment(grpSIdx(j)+1,gsize(j));
  }
  gamma(0) = beta(0)-gamma.segment(1,p-1).adjoint()*Xcenter;
  
  return gamma;
}

template <typename TX>
VectorXd groupLassoFit<TX>::org_to_std(const VectorXd & gamma)
{
  VectorXd beta(gamma);
  
  for(int j=1;j<J;++j)
  {
    beta.segment(grpSIdx(j)+1,gsize(j)) =
      Rinvs[j].inverse()*gamma.segment(grpSIdx(j)+1,gsize(j));
  }
  
  beta(0)= gamma(0)+ gamma.segment(1,p-1).adjoint()*Xcenter ;
  
  return beta;
}

//linpred = beta0+Q1beta1+...+Qpbetap
template <typename TX>
VectorXd groupLassoFit<TX>::linpred(const VectorXd & beta)
{
  VectorXd lpred(N);
  lpred.setZero();
  lpred.setConstant(beta(0));
  
  for (int j=1; j<J; ++j)
  {
    int sind = grpSIdx(j);
    lpred+=X.block(0,sind,N,gsize(j))*(Rinvs[j]*beta.segment(sind+1,gsize(j)));
  }
  lpred = lpred.array()-(lpred.mean()-beta(0)); //if X already centered, lpred.mean = 0
  return lpred;
}

template <typename TX>
VectorXd groupLassoFit<TX>::linpred_update(const VectorXd & new_resid, const VectorXd & old_resid, const VectorXd & old_lpred)
{
  return old_resid-new_resid+old_lpred;
}

template <class TX>
ArrayXd groupLassoFit<TX>::computeLambdaSequence(const VectorXd & y)
{
  ArrayXd lambda_path(pathLength);
  VectorXd gradnorm(J);
  double lammax(1);
  double TOLERANCE(1e-08);
  
  gradnorm.setZero();
  VectorXd ycentered = y.array()-y.mean();
  
  for (int j=1; j<J;++j)
  {
    int sind = grpSIdx(j);
    g[j] = (Rinvs[j].adjoint()*(X.block(0,sind,N,gsize(j)).adjoint()*ycentered))/N;
    gradnorm(j) = g[j].norm()/pen(j);
    
  }
  
  lammax = gradnorm.maxCoeff()+TOLERANCE;
  
  double logDiff=std::log(lammax)-std::log(lambdaMinRatio*lammax);
  double ratio=std::exp(-logDiff/(pathLength-1));
  
  lambda_path(0)=lammax;
  for (int i=1; i<pathLength;++i)
  {
    lambda_path(i)=lambda_path(i-1)*ratio;
  }
  
  return lambda_path;
}


template <typename TX>
bool groupLassoFit<TX>::checkKKT_j(int j, const VectorXd & resid, const ArrayXd & lambda_k)
{
  bool kkt_at_j(true);
  int sind = grpSIdx(j);
  Map<VectorXd> bj(&beta.coeffRef(sind+1),gsize(j));
  VectorXd bj_old = bj;
  VectorXd zj;
  //VectorXd sresid;
  //if(std::abs(resid.mean())>1e-10){throw std::invalid_argument("resid_mean != 0");} Commented due to extra calculation needed
  if(j>0)
  {
    //sresid=resid.array()-resid.mean();
    g[j] =Rinvs[j].adjoint()*((X.block(0,sind,N,gsize(j)).adjoint()*resid))/N;
    zj = g[j]+bj_old;
  }
  else
  { zj = (X.block(0,sind,N,gsize(j)).adjoint()*resid)/N+bj_old;}
  
  kkt_at_j = zj.norm()<lambda_k(j);
  return kkt_at_j;
}

template <typename TX>
bool groupLassoFit<TX>::KKT(const VectorXd & resid, const ArrayXd & lambda_k, int setidx)
{
  bool violation(false);
  std::set<int> set;
  if(setidx==1){set = inactiveSet1;}
  else if (setidx==2){set = inactiveSet2;}
  
  if(set.size()>0)
  {
    std::set<int>::const_iterator it;
    it = set.begin();
    std::set<int> newActiveSet;
    
    while(it!=set.end())
    {
      if(!checkKKT_j(*it, resid, lambda_k))
        newActiveSet.insert(*it);
      it++;
    }
    
    if(newActiveSet.size()>0)
    {
      violation = true;
      if(setidx==1)
      {
        
        for(std::set<int>::const_iterator it_nAS = newActiveSet.begin();
            it_nAS!= newActiveSet.end();it_nAS++)
        {
          inactiveSet.erase(*it_nAS);
          inactiveSet1.erase(*it_nAS);
          activeSet.insert(*it_nAS);
        }
      }
      else if(setidx==2)
      {
        for(std::set<int>::const_iterator it_nAS = newActiveSet.begin();
            it_nAS!= newActiveSet.end();it_nAS++)
        {
          inactiveSet.erase(*it_nAS);
          inactiveSet2.erase(*it_nAS);
          activeSet.insert(*it_nAS);
        }
      }
    }//if newActiveSet.size > 0
    iter++;
  }//if set.size > 0
  
  return violation;
}
template <class TX>
void groupLassoFit<TX>::coordinateDescent_0(VectorXd & resid)
{
  int j=0;
  Map<VectorXd> bj(&beta.coeffRef(0),gsize(j));
  VectorXd bj_old = bj;
  
  bj = resid.mean()+bj_old.array();//bj = mean(y)
  resid = resid.array()-(bj-bj_old).coeff(0,0);
  
  //iter++;
}
//Do BCD in active set until active set coefficients converge or number of cycle reaches iter_max
//Do coordinate descent in activeset
template <typename TX>
void groupLassoFit<TX>::blockCoordinateDescent(VectorXd & resid, const ArrayXd & lambda_k, const double tol)
{
  std::set<int>::const_iterator it;
  VectorXd beta_old(p);
  VectorXd diff(p);
  double error(1);
  converged_CD = false;
  
  while(!converged_CD&&iter<maxit)
  {
    beta_old = beta;
    
    if(activeSet.size()>0){
      it = activeSet.begin();
      while(it!=activeSet.end())
      {
        D_coordinateDescent_j(*it, resid, lambda_k);
        ++it;
      }//end of one cycle
      
      iter++;
    }
    diff = beta-beta_old;
    error = diff.cwiseAbs().maxCoeff();
    
    if(error<tol)
    {
      converged_CD=true;
    }
    
  }//end of while loop
}

template <typename TX>
bool groupLassoFit<TX>::quadraticBCD(VectorXd & resid, const ArrayXd & lambda_k, const double tol)
{
  converged_CD = false;
  converged_KKT = false;
  bool violation(false);
  bool violation1(false);
  
  while(iter<maxit)
  {
    while(iter<maxit)
    {
      if(verbose)
      {
        if(iter % 10000==0&&iter!=0)
        {
          if(verbose){Rcpp::Rcout<<"current iter = "<<iter<<std::endl;}
        }
        
      }
      //BCD in active set
      blockCoordinateDescent(resid, lambda_k, tol);
      //KKT in strong set
      violation1 = KKT(resid,lambda_k,1);
      
      if(converged_CD&&!violation1)
        break;
    }//do BCD on A, check KKT on inactiveSet1
    
    
    violation = KKT(resid,lambda_k,2);
    
    if(!violation){break;}
  }//inner loop + KKT for inactiveSet2
  return !violation;
}

//Rinvs
template <typename TX>
void groupLassoFit<TX>::Rinvs_X()
{
  
  for (int l=0;l<(p-1);++l)
  {
    Xcenter(l) = X.col(l).mean();
    X.col(l) = X.col(l).array()-Xcenter(l);
  }
  
  
  for(int j=1;j<J;++j){
    int sind = grpSIdx(j);
    if(gsize(j)>1)
    {
      //Do QR decomposition
      ColPivHouseholderQR<MatrixXd> qr(X.block(0,sind,N,gsize(j)));
      
      
      if(qr.rank() < gsize(j)){throw std::invalid_argument("X(j) does not have full column rank");}
      
      MatrixXd R = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).triangularView<Upper>();
      MatrixXd P =qr.colsPermutation();
      R=R*P.inverse()/std::sqrt(N);
      
      Rinvs.at(j)= R.inverse();// QtQ = NIn. R' = R/sqrt(N)
    }
    else
    {
      Rinvs.at(j) = X.block(0,sind,N,gsize(j)).adjoint()*X.block(0,sind,N,gsize(j))/N;
      Rinvs.at(j) = Rinvs.at(j).array().sqrt().inverse();
    }
  }
}

template <class TX>
void groupLassoFit<TX>::decenterX()
{
  for (int l=0;l<(p-1);++l)
  {
    X.col(l) = X.col(l).array()+Xcenter(l);
  }
}

template <class TX>
void groupLassoFit<TX>::centerX()
{
  for (int l=0;l<(p-1);++l)
  {
    X.col(l) = X.col(l).array()-Xcenter(l);
  }
}
///////////////////////////////////////////////////////
//MatrixXd Specialization
///////////////////////////////////////////////////////

template <class TX>
void groupLassoFit<TX>::D_coordinateDescent_j(int j, VectorXd & resid, const ArrayXd & lambda_k)
{
  int sind = grpSIdx(j);
  Map<VectorXd> bj(&beta.coeffRef(sind+1),gsize(j));
  VectorXd bj_old = bj;
  VectorXd zj;
  VectorXd update;
  double zjnorm(0);
  
  g[j] = Rinvs[j].adjoint()*((X.block(0,sind,N,gsize(j)).adjoint()*resid))/N;
  zj = g[j]+bj_old;
  zjnorm = zj.norm();
  bj = ((zjnorm>lambda_k(j))?(1-(lambda_k(j)/zjnorm)):0)*zj;
  update =X.block(0,sind,N,gsize(j))*(Rinvs[j]*(bj-bj_old));
  resid -= update;
  //iter++;
}

///////////////////////////////////////////////////////
//Specialization Sparse
///////////////////////////////////////////////////////
template <>
void groupLassoFit<SparseMatrix<double> >::Rinvs_X()
{
  MatrixXd Xcentered;
  MatrixXd Xdl;
  for (int l=0;l<(p-1);++l)
  {
    Xdl = X.col(l);
    Xcenter(l) = Xdl.mean();
  }
  
  for(int j=1;j<J;++j)
  {
    int sind = grpSIdx(j);
    Xcentered = X.block(0,sind,N,gsize(j));
    
    int k(0);
    for(int l=sind; l<(sind+gsize(j)); ++l)
    {
      Xdl = X.col(l);
      k = l-sind;
      Xcentered.col(k) = Xdl.array()-Xcenter(l);
    }
    if(gsize(j)>1)
    {
      //Do QR decomposition
      ColPivHouseholderQR<MatrixXd> qr(Xcentered);
      
      if(qr.rank() < gsize(j)){throw std::invalid_argument("X(j) does not have full column rank");}
      
      MatrixXd R = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).triangularView<Upper>();
      MatrixXd P =qr.colsPermutation();
      R=R*P.inverse()/std::sqrt(N);
      
      Rinvs.at(j)= R.inverse();// QtQ = NIn. R' = R/sqrt(N)
    }
    else
    {
      Rinvs.at(j) = Xcentered.adjoint()*Xcentered/N;
      Rinvs.at(j) = Rinvs.at(j).array().sqrt().inverse();
    }
    
  }
  
}
template <>
void groupLassoFit<SparseMatrix<double> >::decenterX()
{
  //do nothing
}

template <>
void groupLassoFit<SparseMatrix<double> >::centerX()
{
  //do nothing
}
//Do BCD in active set until active set coefficients converge or number of cycle reaches iter_max
template <>
void groupLassoFit<SparseMatrix<double> >::blockCoordinateDescent(VectorXd & resid, const ArrayXd & lambda_k, const double tol)
{
  std::set<int>::const_iterator it;
  VectorXd beta_old(p);
  VectorXd diff(p);
  double error(1);
  converged_CD = false;
  double correction(0);
  
  while(!converged_CD&&iter<maxit)
  {
    beta_old = beta;
    correction = 0;
    if(activeSet.size()>0)
    {
      it = activeSet.begin();
      while(it!=activeSet.end())
      {
        correction += S_coordinateDescent_j(*it, resid, lambda_k);
        ++it;
      }//end of one cycle
      iter++;
    }
    
    resid =resid.array()-correction;
    diff = beta-beta_old;
    error = diff.cwiseAbs().maxCoeff();
    if(error<tol)
      converged_CD=true;
  }//end of while loop
}

template <class TX>
double groupLassoFit<TX>::S_coordinateDescent_j(int j, VectorXd & resid, const ArrayXd & lambda_k)
{
  int sind = grpSIdx(j);
  Map<VectorXd> bj(&beta.coeffRef(sind+1),gsize(j));
  VectorXd bj_old = bj;
  VectorXd zj;
  VectorXd update;
  VectorXd sresid;
  double zjnorm(0);
  MatrixXd cj;
  
  //New version
  g[j] = Rinvs[j].adjoint()*((X.block(0,sind,N,gsize(j)).adjoint()*resid))/N
    + Rinvs[j].adjoint()*Xcenter.segment(sind,gsize(j))*resid.mean();
    
    zj = g[j] + bj_old;
    zjnorm = zj.norm();
    bj = ((zjnorm>lambda_k(j))?(1-(lambda_k(j)/zjnorm)):0)*zj;
    cj = Rinvs[j]*(bj_old-bj);
    update =X.block(0,sind,N,gsize(j))*cj;
    
    //    update =X.block(0,sind,N,gsize(j))*(Rinvs[j]*(bj-bj_old));
    //    update = update.array()-update.mean();
    resid += update;
    cj = Xcenter.segment(sind,gsize(j)).adjoint()*cj;
    //iter++;
    
    return cj.coeff(0,0);
}
////////////////////////////////////////////////////////////////////////////////
template<class TX>
void groupLassoFit<TX>::checkDesignMatrix(const TX & X)
{
  for(int j=0;j<X.cols();j++)
  {
    if((X.col(j).array()==0).all()){throw std::invalid_argument("each column should have at least one non-zero element");}
  }
}


template<>
void groupLassoFit<SparseMatrix<double> >::checkDesignMatrix(const SparseMatrix<double> & X)
{
  for(int j=0;j<X.cols();j++)
  {
    if(X.col(j).nonZeros()==0){throw std::invalid_argument("each column should have at least one non-zero element");}
  }
}

//Explicit Instantiation
template class groupLassoFit<MatrixXd>;
template class groupLassoFit<SparseMatrix<double> >;
template class groupLassoFit<Map<MatrixXd> >;
