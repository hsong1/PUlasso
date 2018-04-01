#include "sgd.h"
using namespace Eigen;

std::tuple<VectorXd,double,VectorXd,VectorXd,int> GD(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, int N, std::function<VectorXd(int)> x, const VectorXd & ibeta, double stepSize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold,std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient, ArrayXd lambdaj, double eps, bool trace)
{
  std::default_random_engine generator;
  VectorXd beta_n=ibeta;
  VectorXd subgrad(ibeta.size()); subgrad.setZero();
  double fVal(0);
  VectorXd fVal_all(nIters+1);
  VectorXd s_gradient(N), grad_n(ibeta.size()),xs;
  ArrayXi ridx(N);
  for(int i=0;i<N;i++){ridx(i)=i;}
  double gs(0);
  int iter(0);
  bool converged(false);
  s_gradient.setZero(); grad_n.setZero();xs.setZero();
  while(iter<nIters&&!converged){
    //        cout<<"iter:"<<iter<<endl;
    if(trace){fVal_all(iter)= objective(beta_n,lambdaj);
      
    }
    s_gradient = gradient(beta_n,ridx);// m by 1
    // Average over N points. grad_n = weighted mean(grad_i)
    grad_n.setZero();
    for(int s=0;s<N;s++){
      xs = x(ridx(s));
      gs = s_gradient(s);
      grad_n += gs*xs;
    }
    grad_n/=N;
    beta_n -= stepSize*grad_n;
    beta_n = SoftThreshold(beta_n, lambdaj*stepSize);
    subgrad = grad_n;
    subgrad = subgradient(grad_n,beta_n,lambdaj);
    if(subgrad.array().abs().maxCoeff()<eps){converged=true;}
    //        cout<<"grad_n:\n"<<grad_n<<endl;
    //        cout<<"stepSize:"<<stepSize(iter)<<endl;
    //        cout<<"beta_n:\n"<<beta_n<<endl;
    iter++;
  }
  //    cout<<"At the end!\n";
  fVal = objective(beta_n,lambdaj);
  if(trace){fVal_all(iter)= objective(beta_n,lambdaj);}
  std::tuple<VectorXd,VectorXd,MatrixXd> result;
  return std::make_tuple(beta_n,fVal,subgrad,fVal_all,iter);
  //    return beta_n;
}

std::tuple<VectorXd,double,VectorXd,VectorXd,int> SGD(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, std::discrete_distribution<> dist,std::function<VectorXd(int)> x, const VectorXd & ibeta, const VectorXd stepSize, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold,std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient,  ArrayXd lambdaj, double eps, bool trace)
{
  std::default_random_engine generator;
  std::vector<double> p(dist.probabilities());
  
  int N(p.size()), m(batchsize), J(lambdaj.size());
  VectorXd beta_n=ibeta;
  VectorXd subgrad(ibeta.size()); subgrad.setZero();//generalized gradient
  double fVal(0);
  VectorXd fVal_all(nIters+1);
  VectorXd s_gradient_full(N),grad_full(ibeta.size()); //Full gradient
  VectorXd s_gradient_m(m), grad_n(ibeta.size()),xs;
  ArrayXi ridx_f(N),ridx(m);
  for (int s=0;s<N;s++){ridx_f(s)=s;}
  double gs(0);//scalar gradient
  s_gradient_m.setZero(); grad_n.setZero();xs.setZero();
  int iter(0);
  bool converged(false);
  
  while(iter<nIters&&!converged){
    if(trace){fVal_all(iter)= objective(beta_n,lambdaj);
      
    }
    // Select random m samples
    for(int s=0;s<m;s++){
      ridx(s) = dist(generator);
    }
    // Calculate scalar part of gradients of m points at beta_n
    // grad_i(beta_n) = gi*(1,xi)^T
    s_gradient_m = gradient(beta_n,ridx);// m by 1
    
    // Average over m points. grad_n = weighted mean(grad_i)
    grad_n.setZero();
    for(int s=0;s<m;s++){
      //            cout<<"sample: "<<ridx[s]<<endl;
      xs = x(ridx(s));
      //            cout<<"gradient_m(s)"<<gradient_m(s)<<endl;
      gs = s_gradient_m(s)/p[ridx(s)];
      grad_n += gs*xs;
    }
    grad_n/=(m*N);
    
    beta_n -= stepSize(iter)*grad_n;
    beta_n = SoftThreshold(beta_n, lambdaj*stepSize(iter));
    //        cout<<"grad_n:\n"<<grad_n<<endl;
    //        cout<<"stepSize:"<<stepSize(iter)<<endl;
    //        cout<<"beta_n:\n"<<beta_n<<endl;
    if(iter%N==0&&iter!=0){
      grad_full.setZero();
      s_gradient_full = gradient(beta_n,ridx_f);
      for (int s=0; s<N; s++){
        xs = x(s);
        gs = s_gradient_full(s);
        grad_full += gs*xs;
      }
      grad_full/=N;
      
      subgrad = grad_full;
      subgrad = subgradient(grad_full,beta_n,lambdaj);
      if(subgrad.array().abs().maxCoeff()<eps){converged=true;}
    }
    iter++;
  }
  fVal = objective(beta_n,lambdaj);
  if(trace){fVal_all(iter)= objective(beta_n,lambdaj);}
  std::tuple<VectorXd,VectorXd,MatrixXd> result;
  return std::make_tuple(beta_n,fVal,subgrad,fVal_all,iter);
  //    return betaMat;
}


std::tuple<VectorXd,double,VectorXd,VectorXd,int> SVRG(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, std::discrete_distribution<> dist,std::function<VectorXd(int)> x, const VectorXd & ibeta, double stepSize, int updateFreq, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold,std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient,  ArrayXd lambdaj, double eps, bool trace)
{
  std::default_random_engine generator;
  std::vector<double> p(dist.probabilities());
  
  int N(p.size()), m(batchsize);
  VectorXd beta_n=ibeta;
  VectorXd subgrad(ibeta.size()); subgrad.setZero();
  double fVal(0);
  VectorXd fVal_all(nIters+1);
  //    MatrixXd betaMat(nIters,ibeta.size());
  VectorXd s_gradient_full(N),grad_full(ibeta.size()); //Full gradient
  VectorXd s_gradient_m(m), grad_n(ibeta.size()); //Each update
  VectorXd xs;
  double gs(0);
  s_gradient_full.setZero();s_gradient_m.setZero();
  xs.setZero();
  ArrayXi ridx_f(N), ridx(m);
  int iter(0);
  bool converged(false);
  
  for (int s=0;s<N;s++){ridx_f(s)=s;}
  
  while(iter<nIters&&!converged){
    if(trace){fVal_all(iter)= objective(beta_n,lambdaj);
      
    }
    if(iter%updateFreq==0)
    {
      //            cout<<"update full gradient"<<endl;
      //            cout<<"beta_n\n"<<beta_n<<endl;
      grad_full.setZero();
      s_gradient_full = gradient(beta_n,ridx_f);
      for (int s=0; s<N; s++){
        xs = x(s);
        gs = s_gradient_full(s);
        grad_full += gs*xs;
      }
      grad_full/=N;
      //            cout<<"grad_full:\n"<<grad_full<<endl;
      subgrad = grad_full;
      subgrad = subgradient(grad_full,beta_n,lambdaj);
      if(subgrad.array().abs().maxCoeff()<eps){converged=true;}
      //            cout<<"grad_full\n"<<grad_full<<endl;
      
    }
    
    // sample m points
    for(int s=0;s<m;s++){
      ridx(s) = dist(generator);
    }
    s_gradient_m = gradient(beta_n,ridx);// m by 1
    //        cout<<"s_gradient_m: "<<s_gradient_m<<endl;
    grad_n.setZero();
    
    // gradients at m points, adjusted by previous gradients at m points
    for(int s=0;s<m;s++){
      //                        cout<<"sample: "<<ridx[s]<<endl;
      xs = x(ridx(s));
      gs = (s_gradient_m(s)-s_gradient_full(ridx(s)))/p[ridx(s)];
      //                        cout<<"x(s):\n"<<xs<<endl;
      grad_n += gs*xs;
    }
    grad_n/=(m*N);
    grad_n +=grad_full;
    beta_n -= stepSize*grad_n;
    beta_n = SoftThreshold(beta_n, lambdaj*stepSize);
    //                cout<<"grad_n:\n"<<grad_n<<endl;
    //                cout<<"stepSize:"<<stepSize<<endl;
    //                cout<<"beta_n:\n"<<beta_n<<endl;
    iter++;
    
  }
  fVal = objective(beta_n,lambdaj);
  if(trace){fVal_all(iter)= objective(beta_n,lambdaj);}
  std::tuple<VectorXd,VectorXd,MatrixXd> result;
  return std::make_tuple(beta_n,fVal,subgrad,fVal_all,iter);
}

std::tuple<VectorXd,double,VectorXd,VectorXd,int> SAG(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, std::discrete_distribution<> dist,std::function<VectorXd(int)> x, const VectorXd & ibeta, double stepSize, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold, std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient, ArrayXd lambdaj, bool sampleSizeAdjustment, double eps, bool trace)
{
  std::default_random_engine generator;
  std::vector<double> p(dist.probabilities());
  
  int N(p.size()), m(batchsize);
  VectorXd beta_n=ibeta;
  VectorXd subgrad(ibeta.size()); subgrad.setZero();
  double fVal(0);
  VectorXd fVal_all(nIters+1);
  //    MatrixXd betaMat(nIters,ibeta.size());
  VectorXd s_gradient_p(N), sum_grad_p(ibeta.size());//gradient evaluated at different theta
  //    VectorXd grad_n(ibeta.size());
  VectorXd s_gradient_full(N),grad_full(ibeta.size()); //Full gradient
  VectorXd s_gradient_m(m), grad_update(ibeta.size());
  VectorXd xs;
  ArrayXi ridx_f(N),ridx(m);
  for (int s=0;s<N;s++){ridx_f(s)=s;}
  ArrayXi sampleIndicator(N); int sampleSize(0);
  double gs(0);
  s_gradient_p.setZero();
  s_gradient_m.setZero();
  sum_grad_p.setZero();
  xs.setZero();
  sampleIndicator.setZero();
  int iter(0);
  bool converged(false);
  
  while(iter<nIters&&!converged){
    if(trace){fVal_all(iter)= objective(beta_n,lambdaj);
      
    }
    
    // sample m points
    for(int s=0;s<m;s++){
      ridx(s) = dist(generator);
    }
    s_gradient_m = gradient(beta_n,ridx);// m by 1
    grad_update.setZero();
    
    // gradients at m points, adjusted by previous gradients at m points
    for(int s=0;s<m;s++){
      //            cout<<"sample: "<<ridx[s]<<endl;
      xs = x(ridx(s));
      gs = (s_gradient_m(s)-s_gradient_p(ridx(s)))/p[ridx(s)];
      //                        cout<<"x(s):\n"<<xs<<endl;
      //            cout<<"s_gradient_m(s): "<<s_gradient_m(s)<<endl;
      //            cout<<"s_gradient_p(s): "<<s_gradient_p(ridx(s))<<endl;
      grad_update += gs*xs;
      s_gradient_p(ridx(s)) = s_gradient_m(s);
      if(sampleSizeAdjustment&&sampleSize<=N){
        sampleIndicator(ridx(s))=1;
        sampleSize = sampleIndicator.sum();
      }
    }
    grad_update/=(m*N);
    //        cout<<"grad_update:\n"<<grad_update<<endl;
    sum_grad_p +=grad_update;//update sum_grad_p
    //        cout<<"mean_grad:\n"<<(sum_grad_p/N)<<endl;
    if(sampleSizeAdjustment){
      beta_n -= stepSize*(sum_grad_p/sampleSize);
    }else{
      beta_n -= stepSize*(sum_grad_p/N);
    }
    
    beta_n = SoftThreshold(beta_n, lambdaj*stepSize);
    //                cout<<"grad_n:\n"<<grad_n<<endl;
    //        cout<<"stepSize:"<<stepSize<<endl;
    //        cout<<"beta_n:\n"<<beta_n<<endl;
    if(iter%N==0&& iter!=0){
      grad_full.setZero();
      s_gradient_full = gradient(beta_n,ridx_f);
      for (int s=0; s<N; s++){
        xs = x(s);
        gs = s_gradient_full(s);
        grad_full += gs*xs;
      }
      grad_full/=N;
      
      subgrad = grad_full;
      subgrad = subgradient(grad_full,beta_n,lambdaj);
      if(subgrad.array().abs().maxCoeff()<eps){converged=true;}
    }
    iter++;
  }
  fVal = objective(beta_n,lambdaj);
  if(trace){fVal_all(iter)= objective(beta_n,lambdaj);}
  std::tuple<VectorXd,VectorXd,MatrixXd> result;
  return std::make_tuple(beta_n,fVal,subgrad,fVal_all,iter);
  //    return betaMat;
}