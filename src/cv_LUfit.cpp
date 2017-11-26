#include "cv_LUfit.h"
template <class TX>
cv_LUfit<TX>::cv_LUfit(const TX & X_, VectorXd & z_, VectorXd & icoef_, ArrayXd & gsize_,ArrayXd & pen_,ArrayXd & lambdaseq_,bool isUserLambdaseq_,int pathLength_,double lambdaMinRatio_, double pi_, int maxit_, double tol_,double inner_tol_,bool useStrongSet_,bool verbose_,int nfolds_,int nfits_,int ncores_)
  :X(X_),z(z_),icoef(icoef_), gsize(gsize_), pen(pen_),lambdaseq(lambdaseq_), isUserLambdaseq(isUserLambdaseq_),pathLength(pathLength_),lambdaMinRatio(lambdaMinRatio_),pi(pi_),maxit(maxit_), tol(tol_),inner_tol(inner_tol_),useStrongSet(useStrongSet_),verbose(verbose_),nfolds(nfolds_),nfits(nfits_),ncores(ncores_),lu_f(X_, z_, icoef_, gsize_,pen_,lambdaseq_,isUserLambdaseq_,pathLength_,lambdaMinRatio_,pi_,maxit_,tol_,inner_tol_,useStrongSet_,verbose_)
{
  
  N = static_cast<int>(X.rows());
  p = static_cast<int>(X.cols())+1;
  K = isUserLambdaseq?(static_cast<int>(lambdaseq.size())):(pathLength);
  nl = z.sum();
  nu = N-nl;
  
  //    TX pX_l = X.topRows(nl);
  //    TX pX_u = X.bottomRows(nu);
  
  int rmdrl = nl % nfolds;
  int cvSizelt = nl / nfolds;
  int rmdru = nu % nfolds;
  int cvSizeut = nu / nfolds;
  
  cvSizel = ArrayXi::Zero(nfolds);
  cvSizeu = ArrayXi::Zero(nfolds);
  
  for (int ii=0;ii<nfolds;++ii)
  {
    cvSizel(ii)=(ii<rmdrl)?(cvSizelt+1):(cvSizelt);
    cvSizeu(ii)=(ii<rmdru)?(cvSizeut+1):(cvSizeut);
  }
  
  //    ArrayXi Xl_sIdx;
  //    ArrayXi Xu_sIdx;
  
  Xl_sIdx=ArrayXi::Zero(nfolds);
  Xu_sIdx=ArrayXi::Zero(nfolds);
  
  for(int ii=1;ii<nfolds;++ii)
  {
    Xl_sIdx(ii)=Xl_sIdx(ii-1)+cvSizel(ii-1);
    Xu_sIdx(ii)=Xu_sIdx(ii-1)+cvSizeu(ii-1);
  }
  //    pX_ls.resize(nfolds);
  //    pX_us.resize(nfolds);
  //    
  //    for (int ii=0;ii<nfolds;++ii)
  //    {
  //        pX_ls[ii] = pX_l.block(Xl_sIdx(ii),0,cvSizel(ii),X.cols());
  //        pX_us[ii] = pX_u.block(Xu_sIdx(ii),0,cvSizeu(ii),X.cols());
  //        
  //    };
  nullDev.resize(K);
  Deviances.resize(K,std::min(nfolds,nfits));
  
  nullDev.setZero();
  Deviances.setZero();
  
  coefMat.resize(p*std::min(nfolds,nfits),K);
  coefMat.setZero();
  
  std_coefMat.resize(p*std::min(nfolds,nfits),K);
  std_coefMat.setZero();
  
  convFlagMat.resize(K, std::min(nfolds,nfits));
  convFlagMat.setZero();
};



template <class TX>
LUfit<TX> cv_LUfit<TX>::getlu_f(){return lu_f;}

template <class TX>
MatrixXd cv_LUfit<TX>::getnullDev(){return nullDev;}

template <class TX>
MatrixXd cv_LUfit<TX>::getDeviances(){return Deviances;}

template <class TX>
MatrixXd cv_LUfit<TX>::getCoefMat(){return coefMat;}

template <class TX>
MatrixXd cv_LUfit<TX>::getStdCoefMat(){return std_coefMat;}

template <typename TX>
ArrayXd cv_LUfit<TX>::getLambdaSequence(){return lambdaseq;}

template <typename TX>
MatrixXi cv_LUfit<TX>::getconvFlagMat(){return convFlagMat;}

template <class TX>
void cv_LUfit<TX>::cv_LUfit_main()
{
  
  lambdaseq = lu_f.getLambdaSequence();
  
  //setup Open-MP
  int mThreads = omp_get_num_procs();
  if(ncores < 1) {ncores = mThreads;}
  omp_set_dynamic(0);
  omp_set_num_threads(ncores);
  VectorXd nullDev_;
  MatrixXd Deviances_,coefMat_,std_coefMat_;
  MatrixXi convFlagMat_;
  nullDev_.resize(K);
  Deviances_.resize(K,std::min(nfolds,nfits));
  coefMat_.resize(p*std::min(nfolds,nfits),K);
  std_coefMat_.resize(p*std::min(nfolds,nfits),K);
  convFlagMat_.resize(K,std::min(nfolds,nfits));
  
  nullDev_.setZero();
  Deviances_.setZero();
  coefMat_.setZero();
  std_coefMat_.setZero();
  convFlagMat_.setZero();
  
  if(verbose){Rcpp::Rcout<<"Starting cross-validation"<<std::endl;}
  
  int errCount = 0;
#pragma omp parallel for shared(nullDev_,Deviances_,coefMat_) schedule(dynamic) reduction(+: errCount)
  for (int j=0;j<std::min(nfolds,nfits);++j)
  {
    int x;
    x=omp_get_thread_num();
    // #pragma omp critical
    // {if(verbose){std::cout<<"cross validation for "<<j<<" using thread "<<x<<std::endl;};}
    try
    {
      MatrixXd X_lu_t, X_lu_v;
      VectorXd z_lu_t, z_lu_v;
      d_setup_t(X_lu_t,z_lu_t,j);
      d_setup_v(X_lu_v,z_lu_v,j);
      
      LUfit<MatrixXd> lu_t(X_lu_t, z_lu_t, icoef, gsize,pen,lambdaseq,true,pathLength,lambdaMinRatio,pi,maxit,tol,inner_tol,useStrongSet,false);
      //LUfit<TX> lu_v(X_lu_v, z_lu_v, icoef, gsize,pen,lambdaseq,true,pathLength,lambdaMinRatio,pi,maxit,tol,inner_tol,useStrongSet,false);
      lu_t.LUfit_main();
      
      VectorXd coef(p),coef0(p);
      coef0 << std::log(pi/(1-pi)),VectorXd::Zero(p-1);
#pragma omp critical
{
  coefMat_.middleRows(p*j, p) = lu_t.getCoefficients();//RHS p by K matrix
  std_coefMat_.middleRows(p*j, p) = lu_t.getStdCoefficients();//RHS p by K matrix
  convFlagMat_.middleCols(j, 1) = lu_t.getconvFlag();
  for (int k=0; k<coefMat_.cols();k++)
  {
    coef = coefMat_.middleRows(p*j,p).middleCols(k,1); //jth cv coef for kth lambda
    //nullDev_(k) = lu_v.getnullDev();
    nullDev_(k) = evalDeviance(X_lu_v, z_lu_v, pi, coef0);
    //Deviances_(k,j) = lu_v.deviance(coef);
    Deviances_(k,j) = evalDeviance(X_lu_v, z_lu_v, pi, coef);
  }
}
    }//end of try
    catch(const std::invalid_argument& e){errCount++;}
    
  }// parallel loop
  
  if(errCount!=0){throw std::invalid_argument("at least one column in training X == 0 ");};
  
  nullDev = nullDev_;
  Deviances = Deviances_;
  coefMat = coefMat_;
  std_coefMat = std_coefMat_;
  convFlagMat = convFlagMat_;
  
  //Finally, we fit the full model
  if(verbose){Rcpp::Rcout<<"Fitting full data"<<std::endl;}
  
  lu_f.LUfit_main();
};

template <>
void cv_LUfit<SparseMatrix<double> >::cv_LUfit_main()
{
  
  lambdaseq = lu_f.getLambdaSequence();
  
  //setup Open-MP
  int mThreads = omp_get_num_procs();
  if(ncores < 1) {ncores = mThreads;}
  omp_set_dynamic(0);
  omp_set_num_threads(ncores);
  VectorXd nullDev_;
  MatrixXd Deviances_,coefMat_,std_coefMat_;
  MatrixXi convFlagMat_;
  nullDev_.resize(K);
  Deviances_.resize(K,std::min(nfolds,nfits));
  coefMat_.resize(p*std::min(nfolds,nfits),K);
  std_coefMat_.resize(p*std::min(nfolds,nfits),K);
  convFlagMat_.resize(K,std::min(nfolds,nfits));
  
  nullDev_.setZero();
  Deviances_.setZero();
  coefMat_.setZero();
  std_coefMat_.setZero();
  convFlagMat_.setZero();
  
  if(verbose){Rcpp::Rcout<<"Starting cross-validation"<<std::endl;}
  
  int errCount = 0;
#pragma omp parallel for shared(nullDev_,Deviances_,coefMat_) schedule(dynamic) reduction(+: errCount)
  for (int j=0;j<std::min(nfolds,nfits);++j)
  {
    int x;
    x=omp_get_thread_num();
    
    try
    {
      SparseMatrix<double> X_lu_t, X_lu_v;
      VectorXd z_lu_t, z_lu_v;
      s_setup_t(X_lu_t,z_lu_t,j);
      s_setup_v(X_lu_v,z_lu_v,j);
      
      LUfit<SparseMatrix<double> > lu_t(X_lu_t, z_lu_t, icoef, gsize,pen,lambdaseq,true,pathLength,lambdaMinRatio,pi,maxit,tol,inner_tol,useStrongSet,false);
      //            LUfit<TX> lu_v(X_lu_v, z_lu_v, icoef, gsize,pen,lambdaseq,true,pathLength,lambdaMinRatio,pi,maxit,tol,inner_tol,useStrongSet,false);
      lu_t.LUfit_main();
      
      VectorXd coef(p),coef0(p);
      coef0 << std::log(pi/(1-pi)),VectorXd::Zero(p-1);
#pragma omp critical
{
  coefMat_.middleRows(p*j, p) = lu_t.getCoefficients();//RHS p by K matrix
  std_coefMat_.middleRows(p*j, p) = lu_t.getStdCoefficients();//RHS p by K matrix
  convFlagMat_.middleCols(j, 1) = lu_t.getconvFlag();
  for (int k=0; k<coefMat_.cols();k++)
  {
    coef = coefMat_.middleRows(p*j,p).middleCols(k,1); //jth cv coef for kth lambda
    //nullDev_(k) = lu_v.getnullDev();
    nullDev_(k) = evalDeviance(X_lu_v, z_lu_v, pi, coef0);
    //Deviances_(k,j) = lu_v.deviance(coef);
    Deviances_(k,j) = evalDeviance(X_lu_v, z_lu_v, pi, coef);
  }
}
    }//end of try
    catch(const std::invalid_argument& e){errCount++;}
    
  }// parallel loop
  
  if(errCount!=0){throw std::invalid_argument("at least one column in training X == 0 ");};
  
  nullDev = nullDev_;
  Deviances = Deviances_;
  coefMat = coefMat_;
  std_coefMat = std_coefMat_;
  convFlagMat = convFlagMat_;
  
  //Finally, we fit the full model
  if(verbose){Rcpp::Rcout<<"Fitting full data"<<std::endl;}
  
  lu_f.LUfit_main();
};

template<class TX>
void cv_LUfit<TX>::d_setup_t(MatrixXd & X_lu_t, VectorXd & z_lu_t, int j)
{
  int nl_t = nl - cvSizel(j);
  int nu_t = nu - cvSizeu(j);
  
  X_lu_t.resize(nl_t+nu_t,X.cols());
  z_lu_t.resize(nl_t+nu_t);
  X_lu_t.setZero();
  z_lu_t.setZero();
  int sind(0);
  int nrow(0);
  
  //    TX pX_l = X.topRows(nl);
  //    TX pX_u = X.bottomRows(nu);
  
  for (int i=0;i<nfolds;++i)
  {
    if(i!=j)
    {
      //nrow = pX_ls[i].rows();
      nrow = cvSizel(i);
      //X_lu_t.middleRows(sind,nrow) = pX_ls[i];
      X_lu_t.middleRows(sind,nrow) = X.block(Xl_sIdx(i),0,cvSizel(i),X.cols());
      sind += nrow;
    }
  }
  for (int i=0;i<nfolds;++i)
  {
    if(i!=j)
    {
      //nrow = pX_us[i].rows();
      nrow = cvSizeu(i);
      //X_lu_t.middleRows(sind,nrow) = pX_us[i];
      X_lu_t.middleRows(sind,nrow) = X.block(nl+Xu_sIdx(i),0,cvSizeu(i),X.cols());
      sind += nrow;
      
    }
  }
  z_lu_t.segment(0,nl_t)= VectorXd::Ones(nl_t);
  z_lu_t.segment(nl_t,nu_t)= VectorXd::Zero(nu_t);    //X_lu_t.topRows(nl_t)= pX_l.block(Xl_sIdx(j),0,cvSizel(j),p);
  //X_lu_t.bottomRows(nu_t) = pX_u.block(Xu_sIdx(j),0,cvSizeu(j),p);
}

template<class TX>
void cv_LUfit<TX>::d_setup_v(MatrixXd & X_lu_v, VectorXd & z_lu_v, int j)
{
  int nl_v = cvSizel(j);
  int nu_v = cvSizeu(j);
  
  X_lu_v.resize(nl_v+nu_v,X.cols());
  z_lu_v.resize(nl_v+nu_v);
  X_lu_v.setZero();
  z_lu_v.setZero();
  int sind(0);
  int nrow(0);
  //nrow = pX_ls[j].rows();
  nrow = cvSizel(j);
  //X_lu_v.middleRows(sind,nrow) = pX_ls[j];
  X_lu_v.middleRows(sind,nrow) = X.block(Xl_sIdx(j),0,cvSizel(j),X.cols());
  sind += nrow;
  //nrow = pX_us[j].rows();
  nrow = cvSizeu(j);
  X_lu_v.middleRows(sind,nrow) = X.block(nl+Xu_sIdx(j),0,cvSizeu(j),X.cols());
  z_lu_v.segment(0,nl_v)= VectorXd::Ones(nl_v);
  z_lu_v.segment(nl_v,nu_v)= VectorXd::Zero(nu_v);
}

////////////

template<class TX>
void cv_LUfit<TX>::s_setup_t(SparseMatrix<double> & X_lu_t, VectorXd & z_lu_t, int j)
{
  std::vector<Triplet<double> > tripletList;
  int nl_t = nl - cvSizel(j);
  int nu_t = nu - cvSizeu(j);
  X_lu_t.resize(nl_t+nu_t,(p-1));
  z_lu_t.resize(nl_t+nu_t);
  X_lu_t.setZero();
  z_lu_t.setZero();
  int sind(0);
  int nrow(0);
  for (int i=0;i<nfolds;++i)
  {
    if(i!=j)
    {
      //nrow = pX_ls[i].rows();
      nrow = cvSizel(i);
      SparseMatrix<double> pX_ls;
      MatrixXd pX_lsd;
      //            std::cout<<X.block(Xl_sIdx(i),0,cvSizel(i),X.cols());
      pX_lsd = X.block(Xl_sIdx(i),0,cvSizel(i),X.cols());
      pX_ls = pX_lsd.sparseView();
      for (int k = 0; k < X.cols(); ++k)
      {
        for (SparseMatrix<double>::InnerIterator it(pX_ls,k); it; ++it)
        {
          tripletList.push_back(Triplet<double>(sind+it.row(), it.col(), it.value()));
        }
      }
      sind+=nrow;
    }
  }
  for (int i=0;i<nfolds;++i)
  {
    if(i!=j)
    {
      //nrow = pX_us[i].rows();
      nrow = cvSizeu(i);
      SparseMatrix<double> pX_us;
      MatrixXd pX_usd;
      pX_usd = X.block(nl+Xu_sIdx(i),0,cvSizeu(i),X.cols());
      pX_us = pX_usd.sparseView();
      for (int k = 0; k < X.cols(); ++k)
      {
        for (SparseMatrix<double>::InnerIterator it(pX_us, k); it; ++it)
        {
          tripletList.push_back(Triplet<double>(sind+it.row(), it.col(), it.value()));
        }
      }
      sind+=nrow;
    }
  }
  
  X_lu_t.setFromTriplets(tripletList.begin(), tripletList.end());
  z_lu_t.segment(0,nl_t)= VectorXd::Ones(nl_t);
  z_lu_t.segment(nl_t,nu_t)= VectorXd::Zero(nu_t);
}

template<class TX>
void cv_LUfit<TX>::s_setup_v(SparseMatrix<double> & X_lu_v, VectorXd & z_lu_v, int j)
{
  std::vector<Triplet<double> > tripletList;
  int nl_v = cvSizel(j);
  int nu_v = cvSizeu(j);
  
  X_lu_v.resize(nl_v+nu_v,(p-1));
  z_lu_v.resize(nl_v+nu_v);
  X_lu_v.setZero();
  z_lu_v.setZero();
  
  int sind(0);
  int nrow(0);
  
  nrow = cvSizel(j);
  SparseMatrix<double> pX_ls;
  MatrixXd pX_lsd;
  pX_lsd = X.block(Xl_sIdx(j),0,cvSizel(j),X.cols());
  pX_ls = pX_lsd.sparseView();
  for (int k = 0; k < X.cols(); ++k)
  {
    for (SparseMatrix<double>::InnerIterator it(pX_ls, k); it; ++it)
    {
      tripletList.push_back(Triplet<double>(sind+it.row(), it.col(), it.value()));
    }
  }
  sind+=nrow;
  
  nrow = cvSizeu(j);
  SparseMatrix<double> pX_us;
  MatrixXd pX_usd;
  
  pX_usd = X.block(nl+Xu_sIdx(j),0,cvSizeu(j),X.cols());
  pX_us = pX_usd.sparseView();
  for (int k = 0; k < X.cols(); ++k)
  {
    for (SparseMatrix<double>::InnerIterator it(pX_us, k); it; ++it)
    {
      tripletList.push_back(Triplet<double>(sind+it.row(), it.col(), it.value()));
    }
  }
  
  X_lu_v.setFromTriplets(tripletList.begin(), tripletList.end());
  z_lu_v.segment(0,nl_v)= VectorXd::Ones(nl_v);
  z_lu_v.segment(nl_v,nu_v)= VectorXd::Zero(nu_v);
}

//template<>
//void cv_LUfit<MatrixXd>::checkDesignMatrix(const MatrixXd & X)
//{
//    for(int j=0;j<X.cols();j++)
//    {
//        if((X.col(j).array()==0).all()){throw std::invalid_argument("each column should have at least one non-zero element");}
//    }
//}

//template<>
//void cv_LUfit<Map<MatrixXd> >::checkDesignMatrix(const MatrixXd & X)
//{
//    for(int j=0;j<X.cols();j++)
//    {
//        if((X.col(j).array()==0).all()){throw std::invalid_argument("each column should have at least one non-zero element");}
//    }
//}

//template<>
//void cv_LUfit<SparseMatrix<double> >::checkDesignMatrix(const SparseMatrix<double> & X)
//{
//    for(int j=0;j<X.cols();j++)
//    {
//        if(X.col(j).nonZeros()==0){throw std::invalid_argument("each column should have at least one non-zero element");}
//    }
//}
//

//The explicit instantiation part
template class cv_LUfit<MatrixXd>;
template class cv_LUfit<Map<MatrixXd> >;
template class cv_LUfit<SparseMatrix<double> >;
