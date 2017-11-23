#ifndef LUfit_h
#define LUfit_h
#include "groupLasso.h"

using namespace Eigen;

template <class TX>
class LUfit : public groupLassoFit<TX>
{
protected:
    using groupLassoFit<TX>::X;// with intercept, N by p matrix, p = 1+k1+..+k(J-1)
    using groupLassoFit<TX>::y;// size N
    using groupLassoFit<TX>::beta;// size p
    using groupLassoFit<TX>::gsize;// size J, first group = intercept
    using groupLassoFit<TX>::pen; // size J, first element = 0;
    using groupLassoFit<TX>::lambdaseq; //size k, default 100
    using groupLassoFit<TX>::isUserLambdaseq;
    using groupLassoFit<TX>::pathLength;
    using groupLassoFit<TX>::lambdaMinRatio;
    using groupLassoFit<TX>::maxit;
    using groupLassoFit<TX>::tol;
    using groupLassoFit<TX>::verbose;
    
    //Definition Inside
    using groupLassoFit<TX>::default_lambdaseq;
    using groupLassoFit<TX>::resid;
    using groupLassoFit<TX>::grpSIdx;//size J
    using groupLassoFit<TX>::iters;
    using groupLassoFit<TX>::Rinvs;
    using groupLassoFit<TX>::coefficients; //size p*k
    using groupLassoFit<TX>::std_coefficients; //size p*k
    using groupLassoFit<TX>::iter;// current iterations
    //using groupLassoFit<TX>::intercept_set;
    using groupLassoFit<TX>::activeSet;
    using groupLassoFit<TX>::inactiveSet;
    using groupLassoFit<TX>::inactiveSet1;
    using groupLassoFit<TX>::inactiveSet2;
    using groupLassoFit<TX>::g;
    using groupLassoFit<TX>::convFlag;
    
    //Dimension Information
    using groupLassoFit<TX>::N;
    using groupLassoFit<TX>::J;
    using groupLassoFit<TX>::p;
    using groupLassoFit<TX>::K;
    
    using groupLassoFit<TX>::linpred;
    using groupLassoFit<TX>::linpred_update;
    using groupLassoFit<TX>::coordinateDescent_0;
    using groupLassoFit<TX>::quadraticBCD;
    //using groupLassoFit<TX>::setupinactiveSets;
    
    ///////////////////////////////////////////
    
    const double t; //Hessian bound
    VectorXd lresp;//latent response
    double pi;
    int nUpdate;
    double inner_tol;
    bool useStrongSet;
    
    //
    int nl;
    int nu;
    double bias;
    ArrayXi nUpdates;
    
    VectorXd lpred_old;
    VectorXd mu;// mu = 1/(1+exp(-Qbeta))
    VectorXd resid_old;
    VectorXd Deviances;
    double nullDev;
    
    //private functions
    VectorXd convert_mu(const VectorXd & beta);
    VectorXd convert_mustar(const VectorXd & beta);
    void compute_mu_mustar(const VectorXd & lpred, VectorXd & mu, VectorXd & mustar);
    
    void updateObjFunc(VectorXd & lpred); //update mu, lresp, resid at new Qbeta
    void setupinactiveSets(int k, const VectorXd & resid, double lam_max,
                           const ArrayXd & lambdaseq,bool useStrongSet);
    double evalDev(const VectorXd & lpred, const VectorXd & beta);
    ArrayXd lambda_b(int k, const ArrayXd & pen);
    
public:
    LUfit(const TX & X_, VectorXd & z_, VectorXd & icoef_, ArrayXd & gsize_,ArrayXd & pen_,ArrayXd & lambdaseq_, bool isUserLambdaseq_,int pathLength_,double lambdaMinRatio_,double pi_, int maxit_, double tol_, double inner_tol_,bool useStrongSet_,bool verbose_);
    
    void LUfit_main();
    using groupLassoFit<TX>::computeLambdaSequence;
    using groupLassoFit<TX>::getCoefficients;
    using groupLassoFit<TX>::getStdCoefficients;
    using groupLassoFit<TX>::getIters;
    using groupLassoFit<TX>::getconvFlag;
    ArrayXi getnUpdates();
    double getnullDev();
    VectorXd getDeviances();
    //double deviance(const VectorXd & coef);
    using groupLassoFit<TX>::back_to_org;
    using groupLassoFit<TX>::org_to_std;
    
    
};

template <class TX>
double evalDeviance(const TX & X, const VectorXd & z, const double pi, const VectorXd & coef);


#endif /* LUfit_h */
