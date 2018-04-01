#include "pgLUfit.h"
using namespace Eigen;
using namespace std::placeholders;

template <class TX>
pgLUfit<TX>::pgLUfit(TX & X_, VectorXd & z_, VectorXd & icoef_, ArrayXd & gsize_,ArrayXd & pen_,ArrayXd & lambdaseq_,bool isUserLambdaseq_,int pathLength_,double lambdaMinRatio_, double pi_, int maxit_, double tol_,bool verbose_, double stepSize_,int batchSize_, std::vector<double> samplingProbabilities_,std::string method_,bool trace_):
pgGroupLassoFit<TX>(X_,z_,pi_,icoef_,gsize_,pen_,lambdaseq_,isUserLambdaseq_,pathLength_,lambdaMinRatio_,maxit_,tol_,verbose_,trace_),dist(samplingProbabilities_.begin(), samplingProbabilities_.end()),stepSize(stepSize_),batchSize(batchSize_),method(method_)
{

    stepSizeSeq.resize(maxit);
    nUpdates = ArrayXi::Zero(K);
    Deviances = VectorXd::Zero(K);
    fVals = VectorXd::Zero(K);
    subgrads = MatrixXd::Zero(p,K);
    fVals_all = MatrixXd::Zero(maxit+1,K);
    
    VectorXd lpred0(N),beta0(p);
    lpred0 = VectorXd::Ones(N)*std::log(pi/(1-pi));
    beta0 << std::log(pi/(1-pi)),VectorXd::Zero(p-1); //I-only, no need to standardize
    nullDev = evalDev(lpred0);
    
    //Initialize beta
    beta = org_to_std(icoef_);
    if(!beta.segment(1,(p-1)).any())
    {
        beta = beta0 ;
    }

    VectorXd resp0(N);
    VectorXd z0(N);
    z0 =1.0-z_.array();
    resp0 = z_+(z0*pi);
    default_lambdaseq = computeLambdaSequence(resp0);
    if(!isUserLambdaseq){lambdaseq = default_lambdaseq;}
    
};

//Getters
template <typename TX>
ArrayXi pgLUfit<TX>::getnUpdates(){return nUpdates;}
template <typename TX>
double pgLUfit<TX>::getnullDev(){return nullDev;}
template <typename TX>
VectorXd pgLUfit<TX>::getDeviances(){return Deviances;}
template <typename TX>
VectorXd pgLUfit<TX>::getfVals(){return fVals;}
template <typename TX>
MatrixXd pgLUfit<TX>::getfVals_all(){return fVals_all;}
template <typename TX>
MatrixXd pgLUfit<TX>::getSubGradients(){return subgrads;}
//calculate deviance using precalculated lpred
template <class TX>
double pgLUfit<TX>::evalDev(const VectorXd & lpred)
{
    int nl(y.sum());
    int nu = N-nl;
    const double c = std::log(nl/(pi*nu));
    VectorXd pred, logExpLpred, logExpPred,obslogL;
    logExpLpred = (lpred.array().exp().array()+1).array().log();
    pred = c+lpred.array()-logExpLpred.array();
    logExpPred = (1+pred.array().exp()).array().log();
    obslogL = (y.array()*pred.array()-logExpPred.array());//response z_lu = y
    return -2*obslogL.sum();
}

template <class TX>
void pgLUfit<TX>::pgLUfit_main(){
    std::function<double(VectorXd,ArrayXd)> f=std::bind(&pgGroupLassoFit<TX>::evalObjective,this,_1,_2);
    std::function<VectorXd(VectorXd,const ArrayXi &)> g=std::bind(&pgGroupLassoFit<TX>::gradient,this,_1,_2);
    std::function<VectorXd(int)> q = std::bind(&pgGroupLassoFit<TX>::q,this,_1);
    std::function<VectorXd(const VectorXd &, const ArrayXd &)> ST = std::bind(&pgGroupLassoFit<TX>::SoftThreshold,this,_1,_2);
    std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient = std::bind(&pgGroupLassoFit<TX>::subgradient,this,_1,_2,_3);
    ArrayXd lambda_k(K);
    for(int k=0; k<K; k++)
    {
    lambda_k = lambdaseq(k)* pen;
        VectorXd subgrad_k(p);
    double fVal_k;
    VectorXd fVal_all_k;
    int method_int(0);
        if(method=="GD"){
            method_int=1;
        }else if(method=="SGD"){
            method_int=2;
        }else if(method=="SVRG"){
            method_int=3;
        }else if(method=="SAG"){
            method_int=4;
        }else{
            method_int=9;
        }
        
        
        switch(method_int){
            case 1:
                std::tie(beta,fVal_k,subgrad_k,fVal_all_k,iter) = GD(f,g,N,q,beta,stepSize,maxit,ST,subgradient,lambda_k,tol,trace);
                break;
            case 2:
                for(int i=0;i<maxit;i++){stepSizeSeq(i) = stepSize/(1.0+stepSize*lambdaseq(k)*i);}
                std::tie(beta,fVal_k,subgrad_k,fVal_all_k,iter) = SGD(f,g,dist,q,beta,stepSizeSeq,batchSize,maxit,ST,subgradient,lambda_k,tol,trace);
                break;
            case 3:
                 std::tie(beta,fVal_k,subgrad_k,fVal_all_k,iter) = SVRG(f,g,dist,q,beta,stepSize,N,batchSize,maxit,ST,subgradient,lambda_k,tol,trace);
                break;
            case 4:
                std::tie(beta,fVal_k,subgrad_k,fVal_all_k,iter)= SAG(f,g,dist,q,beta,stepSize,batchSize,maxit,ST,subgradient,lambda_k,true,tol,trace);
                break;
            default:
                std::tie(beta,fVal_k,subgrad_k,fVal_all_k,iter) = GD(f,g,N,q,beta,stepSize,maxit,ST,subgradient,lambda_k,tol,trace);
                break;}
        
        Map<MatrixXd> coefficients_k(&coefficients.coeffRef(0, k),p,1);
        Map<MatrixXd> std_coefficients_k(&std_coefficients.coeffRef(0, k),p,1);
        coefficients_k = back_to_org(beta);
        std_coefficients_k = beta;
        nUpdates(k)=iter;
        fVals(k)=fVal_k;
        double penVal(0);
        VectorXd bj;
        for (int j=0;j<J;j++){
            bj=beta.segment(grpSIdx(j)+1,gsize(j));
            penVal+=lambda_k(j)*bj.lpNorm<2>();
        }
        Deviances(k) = (fVal_k-N*penVal)*2;
        subgrads.col(k)=subgrad_k;
        fVals_all.col(k)=fVal_all_k;
        if(subgrad_k.array().abs().maxCoeff()<tol){convFlag(k)=1;}
        
    }
}


////calculate deviance using precalculated lpred
//template <class TX>
//double pgLUfit<TX>::evalObjective(const VectorXd & lpred, const VectorXd & beta)
//{
//    const double c = std::log(nl/(pi*nu));
//    double l12norm(0);
//    VectorXd pred, logExpLpred, logExpPred,obslogL;
//    VectorXd bj;
//    
//    logExpLpred = (lpred.array().exp().array()+1).array().log();
//    pred = c+lpred.array()-logExpLpred.array();
//    logExpPred = (1+pred.array().exp()).array().log();
//    obslogL = (y.array()*pred.array()-logExpPred.array());//response z_lu = y
//    
//    for (int j=0;j<J;j++){
//        bj=beta.segment(grpSIdx(j)+1,gsize(j));
//        l12norm+=bj.lpNorm<2>();
//    }
//    return -2*obslogL.sum()+l12norm;
//}
////Not a member of pgLUfit. Calculate deviance given X,z,pi,coef.
//template <class TX>
//double evalDeviance(const TX & X, const VectorXd & z, const double pi, const VectorXd & coef)
//{
//    int N = X.rows();
//    int p = X.cols()+1;
//    int nl = z.sum();
//    int nu = N-nl;
//    
//    const double c = std::log(nl/(pi*nu));
//    VectorXd lpred(N);
//    lpred.setZero();
//    lpred.setConstant(coef(0));
//    
//    for (int j=1; j<p; ++j)
//    {
//        lpred+=X.block(0,(j-1),N,1)*coef(j);
//    }
//    
//    VectorXd pred, logExpLpred, logExpPred,obslogL;
//    logExpLpred = (lpred.array().exp().array()+1).array().log();
//    pred = c+lpred.array()-logExpLpred.array();
//    logExpPred = (1+pred.array().exp()).array().log();
//    obslogL = (z.array()*pred.array()-logExpPred.array());
//    return -2*obslogL.sum();
//}
//template <class TX>
//VectorXd pgLUfit<TX>::evalObjectiveGrad(const VectorXd & lpred)
//{
//    const double c = std::log(nl/(pi*nu));
//    VectorXd objGrad(p); objGrad.setZero();
//    VectorXd pred, gradpred, logExpLpred, logExpPred,obslogL;
//    logExpLpred = (lpred.array().exp().array()+1).array().log();
//    pred = c+lpred.array()-logExpLpred.array(); //log(nl/pi*nu)+bxi -log(1+exp(bxi))
//    VectorXd exponent1,exponent2,probz,prob1y,gradCoef;
//    exponent1 = (-pred).array().exp();
//    probz=1/(1+exponent1.array()); // 1/(1+exp(-f(b))
//    exponent2 = lpred.array().exp();
//    prob1y = 1/(1+exponent2.array());//1/(1+exp(bxi))
//    gradCoef=(y-probz).array()*prob1y.array();
//    MatrixXd Xcentered_j, Qj;
//    MatrixXd Xdl;
//    
//    for(int j=1;j<J;++j)
//    {
//        Xcentered_j = X.block(0,grpSIdx(j),N,gsize(j));
//        
//        for(int l=0; l<gsize(j); ++l)
//        {
//            Xdl = X.block(0,grpSIdx(j)+l,N,1);
//            Xcentered_j.col(l) = Xdl.array()-Xcenter(grpSIdx(j)+l);
//        }
//        Qj = Xcentered_j*Rinvs[j];
//        
//        for(int l=0; l<gsize(j); ++l){
//            objGrad(j+l)= (gradCoef.array()*Qj.col(l).array()).mean();
//        }
//        objGrad(0) = gradCoef.mean();
//    }
//    return objGrad;
//    
//}
//
////Set up lambda_j for each j in (0,..,J-1) for each lambda(k)
////Return size J
//template <class TX>
//ArrayXd pgLUfit<TX>::lambda_b(int k, const ArrayXd & pen)
//{
//    double lambdak = lambdaseq(k);
//    return lambdak* pen;
//}
//
////BCD
//template <class TX>
//void pgLUfit<TX>::pgLUfit_main()
//{
//    std::random_device rd; // obtain a random number from hardware
//    std::mt19937 rng(rd()); // seed the generator
//    std::uniform_int_distribution<int> uni(0,(N-1)); // integer range 0 to N-1
//    if(!isSPG){sampleSize = N;}
//    ArrayXi ridx(sampleSize);
//    VectorXd lpred(sampleSize),yhat(sampleSize),mustar(sampleSize);
//    VectorXd lpred_N(N);
//    
//    VectorXd beta_old(beta);
//    VectorXd diff(p);
//    double error(1.0);
//    VectorXd lambda_k;
//    bool convergedQ(false); //convergence for a quadratic MM
//    bool converged_lam(false);
//    double TOLERANCE(1e-8);
//    
//    double fval_old(0), fval(0);
//    ArrayXi ridx_N(N);
//    
//    for (int i=0; i<N; i++){
//        ridx_N(i)=i;}
//    
//    for (int k=0;k<K;++k)
//    {
//        if(verbose){cout<<"Fitting "<<k<<"th lambda\n";}
//        iter    = 0;
//        nUpdate = 0;
//        lambda_k = lambda_b(k, pen);// lambda at k, Size J
//        converged_lam = false;
//        //Reset an, b0n, b1n at each new objective problem.
//        an = 0; b0n = 0;
//        b1n.resize(p-1);
//        b1n.setZero();
//        
//        if(lambdaseq(k)>default_lambdaseq(0)-TOLERANCE){
//            if(verbose){cout<<"Analytical Solution\n";}
//            VectorXd beta0(p);
//            beta0 << std::log(pi/(1-pi)),VectorXd::Zero(p-1);
//            beta = beta0 ;
//            converged_lam = true;
//        }
//
//        while(iter<maxit&&!converged_lam)
//        {
//            //take random samples, which define (noisy) quadratic MM
//            //solve quadratic MM
//            //setup
//            for (int i=0; i<sampleSize; i++){
//                if(isSPG){
//                    ridx(i) = uni(rng);
//                }else{ridx(i)=i;}
//            }
////            for(int i=0;i<ridx.size();i++){
////                cout<<"ridx: "<<ridx(i)<<",";
////                cout<<"\n";
////            }
//            
//            lpred = linpred(true,beta,ridx);
//            updateObjective(lpred,ridx,yhat,mustar);//update yhat,mustar
////            if(verbose){cout<<"L(beta_old):"<<fval_old<<endl;}
//            
//            
//            // only for debugging
//            lpred_N = linpred(true,beta,ridx_N);
//            fVals_all(iter,k) = evalObjective(lpred_N,beta);
////            VectorXd objGrad(p),KKTvec(p);
////            objGrad=evalObjectiveGrad(lpred_N);
////            objGrad=objGrad.array().abs();
////            for (int j=0;j<p;j++){
////                KKTvec(j)= (objGrad(j)>lambdaseq(k))?(objGrad(j)-lambdaseq(k)):0;
////            }
////            error = KKTvec.maxCoeff();
//            
//        
////            if(verbose){cout<<"L(beta):"<<fval<<endl;}
////            cout<<"L(beta) new-old:"<<fval-fval_old<<endl;
//            cout<<"fval: "<<fVals_all(iter,k)<<endl;
////            cout<<"KKTvec.max:"<<KKTvec.maxCoeff()<<endl;
////            fval_old=fval;
//            
//            proxGrad(yhat,mustar, ridx,lambda_k, stepSize);
////            cout<<"after prox grad\n";
////            cout<<"beta_old:\n"<<beta_old<<endl;
////            cout<<"beta:\n"<<beta<<endl;
//            diff = beta-beta_old;
//            error = diff.cwiseAbs().maxCoeff();
////            error = KKTvec.maxCoeff();
//            converged_lam = error<tol;
////            cout<<"beta_new:\n"<<beta<<endl;
////            if(verbose){cout<<"nupdate:"<<nUpdate<<endl;}
////            if(verbose){cout<<"error: "<<error<<endl;}
//            
//            //If intercept only model, this is an analytical solution
//            if(!beta.segment(1,(p-1)).any()){
//                VectorXd beta0(p);
//                beta0 << std::log(pi/(1-pi)),VectorXd::Zero(p-1);
//                beta = beta0 ;
//            }
//            nUpdate++;
//            beta_old = beta;
////            cout<<"after prox grad and setting beta_old =beta\n";
////            cout<<"beta_old:\n"<<beta_old<<endl;
////            cout<<"beta:\n"<<beta<<endl;
//        }
//        
//        Map<MatrixXd> coefficients_k(&coefficients.coeffRef(0, k),p,1);
//        Map<MatrixXd> std_coefficients_k(&std_coefficients.coeffRef(0, k),p,1);
//        coefficients_k = back_to_org(beta);
//        std_coefficients_k = beta;
//        iters(k) = iter;
//        nUpdates(k) = nUpdate;
//        lpred_N = linpred(true,beta,ridx_N);
//        Deviances(k) = evalDev(lpred_N);
//        fVals(k) = evalObjective(lpred_N,beta);
////        VectorXd objGrad(p),KKTvec(p);
////        objGrad=evalObjectiveGrad(lpred_N);
////        for (int j=0;j<p;j++){
////            KKTvec(j)= (objGrad.array().abs().coeff(j)>lambdaseq(k))?objGrad.array().abs().coeff(j):0;
////        }
//        
////        cout<<"fval:"<<evalObjective(lpred_N, beta)<<endl;
////        cout<<"objGrad:"<<objGrad<<endl;
////        cout<<"lambda:"<<lambdaseq(k)<<endl;
////        cout<<"|objGrad|<lambda:\n"<<KKTvec<<endl;
////        if(verbose&&converged_lam)
//        if(converged_lam)
//        {cout<<"converged at "<<iter<<"th iterations\n";}
//        if(!converged_lam){convFlag(k)=1;}
//        
//    }
//}
//
////The explicit instantiation part
template class pgLUfit<MatrixXd>;
template class pgLUfit<SparseMatrix<double> >;
template class pgLUfit<Map<MatrixXd> >;
//
//template double evalDeviance<MatrixXd>(const MatrixXd & X, const VectorXd & z, const double pi, const VectorXd & coef);
//template double evalDeviance<Map<MatrixXd> >(const Map<MatrixXd> & X, const VectorXd & z, const double pi, const VectorXd & coef);
//template double evalDeviance<SparseMatrix<double> >(const SparseMatrix<double> & X, const VectorXd & z, const double pi, const VectorXd & coef);
//

