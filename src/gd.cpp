#include "sgd.h"
using namespace Eigen;

std::tuple<VectorXd,double,VectorXd,VectorXd,int,bool> GD(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, int N, std::function<VectorXd(int)> x, const VectorXd & ibeta, double stepSize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold,std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient, ArrayXd lambdaj, double eps, bool trace)
{
    VectorXd beta_n(ibeta),beta_n1(ibeta);
    VectorXd subgrad(ibeta.size()); subgrad.setZero();
    double fVal(0);
    VectorXd fVal_all(nIters+1);
    VectorXd s_gradient(N), grad_n(ibeta.size()),xs;
    ArrayXi ridx(N);
    for(int i=0;i<N;i++){ridx(i)=i;}
    double gs(0);
    int iter(0);
    double betadiff(1);
    bool converged(false);
    s_gradient.setZero(); grad_n.setZero();xs.setZero();
    while(iter<nIters&&!converged){
        //        cout<<"iter:"<<iter<<endl;
        if(trace){fVal_all(iter)= objective(beta_n,lambdaj);}
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
        betadiff=(beta_n-beta_n1).array().abs().maxCoeff();
        beta_n1 = beta_n;
        if(betadiff<eps){converged=true;}
        //                cout<<"grad_n:\n"<<grad_n<<endl;
        //                cout<<"stepSize:"<<stepSize<<endl;
        //                cout<<"beta_n:\n"<<beta_n<<endl;
        //        cout<<"betadiff:"<<betadiff<<endl;
        iter++;
    }
    //    cout<<"At the end!\n";
    fVal = objective(beta_n,lambdaj);
    if(trace){fVal_all(iter)= objective(beta_n,lambdaj);}
    std::tuple<VectorXd,VectorXd,MatrixXd> result;
    subgrad = subgradient(grad_n,beta_n,lambdaj);
    return std::make_tuple(beta_n,fVal,subgrad,fVal_all,iter,converged);
}

