#ifndef pgLUfit_h
#define pgLUfit_h
#include "pgGroupLasso.h"
#include "sgd.h"
#include <random>
using namespace Eigen;
using namespace std;
template <class TX>
class pgLUfit : public pgGroupLassoFit<TX>
{
protected:
//    pgGroupLassoFit<TX> pgGrpLasso;
    std::discrete_distribution<> dist;
//    using pgGroupLassoFit<TX>::X;// with intercept, N by p matrix, p = 1+k1+..+k(J-1)
    using pgGroupLassoFit<TX>::y;// size N
    using pgGroupLassoFit<TX>::beta;// size p
    using pgGroupLassoFit<TX>::pi;
    using pgGroupLassoFit<TX>::gsize;// size J, first group = intercept
    using pgGroupLassoFit<TX>::pen; // size J, first element = 0;
    using pgGroupLassoFit<TX>::lambdaseq; //size k, default 100
    using pgGroupLassoFit<TX>::isUserLambdaseq;
    using pgGroupLassoFit<TX>::pathLength;
    using pgGroupLassoFit<TX>::lambdaMinRatio;
    using pgGroupLassoFit<TX>::maxit;
    using pgGroupLassoFit<TX>::tol;
    using pgGroupLassoFit<TX>::verbose;
    using pgGroupLassoFit<TX>::trace;

//
    //Definition Inside
    VectorXd stepSizeSeq;
    using pgGroupLassoFit<TX>::default_lambdaseq;
    using pgGroupLassoFit<TX>::grpSIdx;//size J
//    using pgGroupLassoFit<TX>::iters;
//    using pgGroupLassoFit<TX>::Rinvs;
    using pgGroupLassoFit<TX>::coefficients; //size p*k
    using pgGroupLassoFit<TX>::std_coefficients; //size p*k
//    using pgGroupLassoFit<TX>::Xcenter;
    using pgGroupLassoFit<TX>::iter;// current iterations
    using pgGroupLassoFit<TX>::convFlag;
//
    //Dimension Information
    using pgGroupLassoFit<TX>::N;
    using pgGroupLassoFit<TX>::J;
    using pgGroupLassoFit<TX>::p;
    using pgGroupLassoFit<TX>::K;
//
//    //function
//    using pgGroupLassoFit<TX>::linpred;
//
//    ///////////////////////////////////////////
//
    double stepSize;
    int batchSize;
    std::string method;
    ArrayXi nUpdates;
    VectorXd Deviances;
    double nullDev;
    VectorXd fVals;
    MatrixXd subgrads;
    MatrixXd fVals_all;
//
//    // proximal gradient functions
//    void updateObjective(const VectorXd & lpred, const ArrayXi & rind, VectorXd & yhat, VectorXd & mustar);
//
//    void proxGrad_0(const VectorXd & yhat, const VectorXd & mustar);
//    void proxGrad_1(const VectorXd & yhat, const VectorXd & mustar, const ArrayXi & ridx, ArrayXd lambda, double stepSize);
//    void proxGrad(const VectorXd & yhat, const VectorXd & mustar, const ArrayXi & ridx, ArrayXd lambda, double stepSize);
//
//    //other functions
//    //lambda sequence for 1<=k<=lambda.size()
//    ArrayXd lambda_b(int k, const ArrayXd & pen);
    double evalDev(const VectorXd & lpred);
//    double evalObjective(const VectorXd & lpred, const VectorXd & beta);
//    VectorXd evalObjectiveGrad(const VectorXd & lpred);
    
public:
    pgLUfit(TX & X_, VectorXd & z_, VectorXd & icoef_, ArrayXd & gsize_,ArrayXd & pen_,ArrayXd & lambdaseq_,bool isUserLambdaseq_,int pathLength_,double lambdaMinRatio_, double pi_, int maxit_, double tol_,bool verbose_, double stepSize_,int sampleSize_, std::vector<double> samplingProbabilities_,std::string method_,bool trace_);
   
    void pgLUfit_main();
    using pgGroupLassoFit<TX>::computeLambdaSequence;
    using pgGroupLassoFit<TX>::getCoefficients;
    using pgGroupLassoFit<TX>::getStdCoefficients;
//    using pgGroupLassoFit<TX>::getIters;
    using pgGroupLassoFit<TX>::getconvFlag;
    ArrayXi getnUpdates();
    double getnullDev();
    VectorXd getDeviances();
    VectorXd getfVals();
    MatrixXd getfVals_all();
    MatrixXd getSubGradients();
    using pgGroupLassoFit<TX>::back_to_org;
    using pgGroupLassoFit<TX>::org_to_std;
    using pgGroupLassoFit<TX>::evalObjective;
    using pgGroupLassoFit<TX>::standardizeX;
    using pgGroupLassoFit<TX>::destandardizeX;
    
//    using pgGroupLassoFit<TX>::decenterX;
    
};

#endif /* pgLUfit_h */

