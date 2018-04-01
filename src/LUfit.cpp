#include "LUfit.h"
using namespace Rcpp;

template <class TX>
LUfit<TX>::LUfit(TX & X_, VectorXd & z_, VectorXd & icoef_, ArrayXd & gsize_,ArrayXd & pen_,ArrayXd & lambdaseq_,bool isUserLambdaseq_,int pathLength_,double lambdaMinRatio_, double pi_, int maxit_, double tol_,double inner_tol_,bool useStrongSet_,bool verbose_,bool trace_)
:groupLassoFit<TX>(X_,z_,icoef_,gsize_,pen_,lambdaseq_,isUserLambdaseq_,pathLength_,lambdaMinRatio_, maxit_,tol_,verbose_,trace_),t(0.25),lresp(z_),pi(pi_),nUpdate(0),inner_tol(inner_tol_),useStrongSet(useStrongSet_)
{
    //Initialize LU parameters
    nl = y.sum();
    nu = N-nl;
    if(nl==0||nu==0){throw std::invalid_argument("Response can't be all zero or one");}
    bias = std::log((nl+nu*pi)/(pi*nu));
    
    nUpdates = ArrayXi::Zero(K);
    Deviances = VectorXd::Zero(K);
    fVals = VectorXd::Zero(K);
    g_grads = MatrixXd::Zero(p,K);
    fVals_all = MatrixXd::Zero(maxit,K);
    
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
    //update mu, lresp, resid at initial beta
    VectorXd lpred = linpred(beta);
    lpred_old = lpred;
    updateObjFunc(lpred);//update mu, lresp, resid at new beta
    
    //lambdaseq, in default, lammax is calculated where lresp = [1,p,1-p]
    VectorXd lresp0(N),mu0(p);
    VectorXd exponent1;
    lresp0 = lresp;
    exponent1 = (-lpred0).array().exp();
    mu0 =1/(1+exponent1.array());
    lresp0.segment(nl,nu) = mu0.segment(nl, nu);
    
    default_lambdaseq = computeLambdaSequence(lresp0);
    if(!isUserLambdaseq){lambdaseq = default_lambdaseq;}
    
    //update inactiveSet1,2
    //If useStrongSet=False, inactiveSet1 = {}, inactiveSet2=inactiveSet
    setupinactiveSets(0, resid, default_lambdaseq[0], lambdaseq, useStrongSet);
    
};

//Getters
template <typename TX>
ArrayXi LUfit<TX>::getnUpdates(){return nUpdates;}
template <typename TX>
double LUfit<TX>::getnullDev(){return nullDev;}
template <typename TX>
VectorXd LUfit<TX>::getDeviances(){return Deviances;}
template <typename TX>
VectorXd LUfit<TX>::getfVals(){return fVals;}
template <typename TX>
MatrixXd LUfit<TX>::getGeneralizedGradients(){return g_grads;}
template <typename TX>
MatrixXd LUfit<TX>::getfVals_all(){return fVals_all.topRows(nUpdate);}

template <typename TX>
void LUfit<TX>::setupinactiveSets(int k, const VectorXd & resid, double lam_max, const ArrayXd & lambdaseq, bool useStrongSet)
{
    inactiveSet1.clear();
    inactiveSet2 = inactiveSet;
    
    if(useStrongSet)
    {
        double cutoff;
        double TOLERANCE = 1e-8;
        VectorXd gj;
        double gjnorm;
        int sind;
        
        //for (int j : inactiveSet)
        for (std::set<int>::const_iterator it = inactiveSet.begin();it!=inactiveSet.end();it++)
        {
            if (k != 0){cutoff = sqrt(pen(*it)) * (2 * lambdaseq(k) - lambdaseq(k-1));}
            else
            {
                if (lam_max > 0){cutoff = sqrt(pen(*it)) * (2 * lambdaseq(k) - lam_max);}
                else cutoff = 0;
            }
            
            sind = grpSIdx(*it);
            g[*it] = Rinvs[*it].adjoint()*((X.block(0,sind,N,gsize(*it)).adjoint()*resid))/N;
            gjnorm = g[*it].norm();
            
            if (gjnorm + TOLERANCE > cutoff)
            {
                inactiveSet1.insert(*it);
                inactiveSet2.erase(*it);
            }
        }
    }
}


template <class TX>
void LUfit<TX>::compute_mu_mustar(const VectorXd & lpred, VectorXd & mu, VectorXd & mustar)
{
    VectorXd exponent1, exponent2;
    exponent1 = (-lpred).array().exp();
    exponent2 = exponent1*std::exp(-bias);
    mu =1/(1+exponent1.array());
    mustar = 1/(1+exponent2.array());
}


//update mu, lresp, resid at new beta
template <class TX>
void LUfit<TX>::updateObjFunc(VectorXd & lpred)
{
    VectorXd mustar;
    compute_mu_mustar(lpred, mu, mustar);
    lresp.segment(nl,nu) = mu.segment(nl, nu);
    resid = (lresp-mustar)/t;
    resid_old = resid;
    coordinateDescent_0(resid);
}
//calculate deviance using precalculated lpred
template <class TX>
double LUfit<TX>::evalDev(const VectorXd & lpred)
{
    const double c = std::log(nl/(pi*nu));
    VectorXd pred, logExpLpred, logExpPred,obslogL;
    logExpLpred = (lpred.array().exp().array()+1).array().log();
    pred = c+lpred.array()-logExpLpred.array();
    logExpPred = (1+pred.array().exp()).array().log();
    obslogL = (y.array()*pred.array()-logExpPred.array());//response z_lu = y
    
    return -2*obslogL.sum();
}
template <class TX>
double LUfit<TX>::evalObjective(const VectorXd & lpred, const VectorXd & beta, const ArrayXd & lambda)
{
    const double c = std::log(nl/(pi*nu));
    double penVal(0);
    VectorXd pred, logExpLpred, logExpPred,obslogL;
    VectorXd bj;
    
    logExpLpred = (lpred.array().exp().array()+1).array().log();
    pred = c+lpred.array()-logExpLpred.array();
    logExpPred = (1+pred.array().exp()).array().log();
    obslogL = (y.array()*pred.array()-logExpPred.array());//response z_lu = y
    
    for (int j=0;j<J;j++){
        bj=beta.segment(grpSIdx(j)+1,gsize(j));
        penVal+=lambda(j)*bj.lpNorm<2>();
    }
    return -obslogL.sum()+N*penVal;
}
//Not a member of LUfit. Calculate deviance given X,z,pi,coef.
template <class TX>
double evalDeviance(const TX & X, const VectorXd & z, const double pi, const VectorXd & coef)
{
    int N = X.rows();
    int p = X.cols()+1;
    int nl = z.sum();
    int nu = N-nl;
    
    const double c = std::log(nl/(pi*nu));
    VectorXd lpred(N);
    lpred.setZero();
    lpred.setConstant(coef(0));
    
    for (int j=1; j<p; ++j)
    {
        lpred+=X.block(0,(j-1),N,1)*coef(j);
    }
    
    VectorXd pred, logExpLpred, logExpPred,obslogL;
    logExpLpred = (lpred.array().exp().array()+1).array().log();
    pred = c+lpred.array()-logExpLpred.array();
    logExpPred = (1+pred.array().exp()).array().log();
    obslogL = (z.array()*pred.array()-logExpPred.array());
    
    return -2*obslogL.sum();
}
template <class TX>
VectorXd LUfit<TX>::evalObjectiveGrad(const VectorXd & lpred)
{
    const double c = std::log(nl/(pi*nu));
    VectorXd objGrad(p); objGrad.setZero();
    VectorXd pred, gradpred, logExpLpred, logExpPred,obslogL;
    logExpLpred = (lpred.array().exp().array()+1).array().log();
    pred = c+lpred.array()-logExpLpred.array(); //log(nl/pi*nu)+bxi -log(1+exp(bxi))
    VectorXd exponent1,exponent2,probz,prob1y,gradCoef;
    exponent1 = (-pred).array().exp();
    probz=1/(1+exponent1.array()); // 1/(1+exp(-f(b))
    exponent2 = lpred.array().exp();
    prob1y = 1/(1+exponent2.array());//1/(1+exp(bxi))
    gradCoef=(y-probz).array()*prob1y.array();
    MatrixXd Xcentered_j, Qj;
    MatrixXd Xdl;
    
    for(int j=1;j<J;++j)
    {
        Xcentered_j = X.block(0,grpSIdx(j),N,gsize(j));
        
        for(int l=0; l<gsize(j); ++l)
        {
            Xdl = X.block(0,grpSIdx(j)+l,N,1);
            Xcentered_j.col(l) = Xdl.array()-Xcenter(grpSIdx(j)+l);
        }
        Qj = Xcentered_j*Rinvs[j];
        
        for(int l=0; l<gsize(j); ++l){
            objGrad(j+l)= (gradCoef.array()*Qj.col(l).array()).mean();
        }
        objGrad(0) = gradCoef.mean();
    }
    return objGrad;
    
}

//Set up lambda_j for each j in (0,..,J-1) for each lambda(k)
//Return size J
template <class TX>
ArrayXd LUfit<TX>::lambda_b(int k, const ArrayXd & pen)
{
    double lambdak = lambdaseq(k);
    return lambdak* pen/t;
}

//BCD
template <class TX>
void LUfit<TX>::LUfit_main()
{
    
    VectorXd beta_old(beta);
    VectorXd diff(p);
    double error(1.0);
    bool converged_lam(false);
    ArrayXd lambda_k;
    bool convergedQ(false);
    VectorXd objGrad(p),KKTvec(p);
    
    for (int k=0;k<K;++k)
    {
        if(verbose){Rcout<<"Fitting "<<k<<"th lambda\n";}
        iter    = 0;
        nUpdate = 0;
        lambda_k = lambda_b(k, pen);
        
        converged_lam = false;
        
        while(iter<maxit&&!converged_lam){
            convergedQ= quadraticBCD(resid, lambda_k,inner_tol);
            diff = beta-beta_old;
            error = diff.cwiseAbs().maxCoeff();
            
            converged_lam = convergedQ&&(error<tol);
            
            //Majorization at current beta
            //VectorXd lpred = linpred(beta);
            
            VectorXd lpred = linpred_update(resid,resid_old,lpred_old);
            //If intercept only model, this is an analytical solution
            if(!beta.segment(1,(p-1)).any())
            {
                VectorXd beta0(p);
                beta0 << std::log(pi/(1-pi)),VectorXd::Zero(p-1);
                beta = beta0 ;
                lpred = linpred(beta);
            }
            if(trace){fVals_all(nUpdate,k) = evalObjective(lpred,beta,(lambda_k*t));}
            
            converged_lam = convergedQ&&(error<tol);
            lpred_old = lpred;
            updateObjFunc(lpred);
            nUpdate++;
            
            if(k!=0){setupinactiveSets(k,resid,default_lambdaseq[0],lambdaseq, useStrongSet);};
            beta_old = beta;
            
        }
        
        Map<MatrixXd> coefficients_k(&coefficients.coeffRef(0, k),p,1);
        Map<MatrixXd> std_coefficients_k(&std_coefficients.coeffRef(0, k),p,1);
        coefficients_k = back_to_org(beta);
        std_coefficients_k = beta;
        iters(k) = iter;
        nUpdates(k)=nUpdate;
        Deviances(k) = evalDev(lpred_old);
        fVals(k) = evalObjective(lpred_old,beta,(lambda_k*t));
        
        VectorXd objGrad(p),KKTvec(p),gj(p);
        objGrad=evalObjectiveGrad(lpred_old);
        double gjnorm;
        KKTvec(0) = objGrad(0);
        for (int j=1;j<J;j++){
            Map<VectorXd> gj(&objGrad.coeffRef(grpSIdx(j)+1),gsize(j));
            gjnorm=gj.norm();
            KKTvec.segment(grpSIdx(j)+1,gsize(j))= ((gjnorm>lambda_k(j))?(1-(lambda_k(j)/gjnorm)):0)*gj;
        }
        g_grads.col(k) = KKTvec;
//        cout<<"fval:"<<evalObjective(lpred_old, beta)<<endl;
//        cout<<"objGrad:\n"<<objGrad<<endl;
//        cout<<"lambda:"<<lambdaseq(k)<<endl;
//        cout<<"|objGrad|<ambda:\n"<<KKTvec<<endl;
        if(verbose&&converged_lam)
        {Rcout<<"converged at "<<nUpdate<<"th iterations\n";}
        if(!converged_lam){convFlag(k)=1;}
        
    }
}

//The explicit instantiation part
template class LUfit<MatrixXd>;
template class LUfit<SparseMatrix<double> >;
template class LUfit<Map<MatrixXd> >;

template double evalDeviance<MatrixXd>(const MatrixXd & X, const VectorXd & z, const double pi, const VectorXd & coef);
template double evalDeviance<Map<MatrixXd> >(const Map<MatrixXd> & X, const VectorXd & z, const double pi, const VectorXd & coef);
template double evalDeviance<SparseMatrix<double> >(const SparseMatrix<double> & X, const VectorXd & z, const double pi, const VectorXd & coef);

