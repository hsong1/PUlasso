#include "sgd.h"
using namespace Eigen;

std::tuple<VectorXd,double,VectorXd,VectorXd,int,bool> SVRG(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, std::function<VectorXd(int)> x, const VectorXd & ibeta, std::vector<double> samplingProbabilities,double stepSize, int updateFreq, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold,std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient,  ArrayXd lambdaj, double eps, bool trace)
{
  int N(samplingProbabilities.size()), m(batchsize);
  VectorXd beta_n(ibeta),beta_n1(ibeta);
  VectorXd subgrad(ibeta.size()); subgrad.setZero();
  double fVal(0);
  VectorXd fVal_all(nIters+1);
  VectorXd s_gradient_full(N),grad_full(ibeta.size()); //Full gradient
  VectorXd s_gradient_m(m), grad_n(ibeta.size()); //Each update
  VectorXd xs;
  double gs(0);
  s_gradient_full.setZero();s_gradient_m.setZero();
  xs.setZero();
  ArrayXi ridx_f(N), ridx(m);
  int iter(0);
  double betadiff(1);
  bool converged(false);
  // C++ version random number generator
  // std::discrete_distribution<> dist(samplingProbabilities.begin(),samplingProbabilities.end());
  // std::random_device r;
  // std::default_random_engine generator(r());
  
  //R
  Rcpp::NumericVector ridx_Rf(N), ridx_R(m);
  for(int i=0;i<N;i++){ridx_Rf[i]=i;}
  Rcpp::NumericVector prob(samplingProbabilities.begin(),samplingProbabilities.end());

   for (int s=0;s<N;s++){ridx_f(s)=s;}

   while(iter<nIters&&!converged){
   if(trace){fVal_all(iter)= objective(beta_n,lambdaj);}
   if(iter%updateFreq==0){
       grad_full.setZero();
       s_gradient_full = gradient(beta_n,ridx_f);
       for (int s=0; s<N; s++){
           xs = x(s);
           gs = s_gradient_full(s);
           grad_full += gs*xs;}
       grad_full/=N;
   }

   // Select random m samples
   // c++
    // for(int s=0;s<m;s++){
        // ridx(s) = dist(generator);
   // }
   // R
  ridx_R = Rcpp::sample(ridx_Rf,m,true,prob);
  std::vector<int> ret_vec = Rcpp::as<std::vector<int> >(ridx_R);
  ridx = Eigen::Map<Eigen::ArrayXi>(ret_vec.data(),m);
   s_gradient_m = gradient(beta_n,ridx);// m by 1
   //        cout<<"s_gradient_m: "<<s_gradient_m<<endl;
   grad_n.setZero();

   // gradients at m points, adjusted by previous gradients at m points
   for(int s=0;s<m;s++){
       // Rcpp::Rcout<<"sample: "<<ridx[s]<<endl;
       xs = x(ridx(s));
       gs = (s_gradient_m(s)-s_gradient_full(ridx(s)))/samplingProbabilities[ridx(s)];
       //            cout<<"gs:"<<gs<<endl;
       grad_n += gs*xs;
   }
   grad_n/=(m*N);
   grad_n +=grad_full;
   beta_n -= stepSize*grad_n;
   beta_n = SoftThreshold(beta_n, lambdaj*stepSize);
   betadiff=(beta_n-beta_n1).array().abs().maxCoeff();
   beta_n1 = beta_n;
   if(betadiff<eps){converged=true;}
   iter++;
  }
  fVal = objective(beta_n,lambdaj);
  if(trace){fVal_all(iter)= objective(beta_n,lambdaj);}
  grad_full.setZero();
  s_gradient_full = gradient(beta_n,ridx_f);
  for (int s=0; s<N; s++){
      xs = x(s);
      gs = s_gradient_full(s);
      grad_full += gs*xs;
  }
  grad_full/=N;
  subgrad = subgradient(grad_full,beta_n,lambdaj);
  
  return std::make_tuple(beta_n,fVal,subgrad,fVal_all,iter,converged);
}


