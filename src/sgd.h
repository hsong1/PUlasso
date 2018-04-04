#ifndef sgd_h
#define sgd_h
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
// #include <iostream>
// #include <random>
# include "Rcpp.h"
#include <tuple>
using namespace Eigen;
using namespace std;
std::tuple<VectorXd,double,VectorXd,VectorXd,int,bool> GD(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, int N, std::function<VectorXd(int)> x, const VectorXd & ibeta, double stepSize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold, std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient, ArrayXd lambdaj, double eps, bool trace);

std::tuple<VectorXd,double,VectorXd,VectorXd,int,bool> SGD(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient ,std::function<VectorXd(int)> x, const VectorXd & ibeta, std::vector<double> samplingProbabilities, const VectorXd stepSize, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold, std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient, ArrayXd lambdaj,double eps, bool trace);

std::tuple<VectorXd,double,VectorXd,VectorXd,int,bool> SVRG(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient,std::function<VectorXd(int)> x, const VectorXd & ibeta, std::vector<double> samplingProbabilities,double stepSize, int updateFreq, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold,std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient,  ArrayXd lambdaj,double eps, bool trace);
// 
std::tuple<VectorXd,double,VectorXd,VectorXd,int,bool> SAG(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, std::function<VectorXd(int)> x, const VectorXd & ibeta, std::vector<double> samplingProbabilities, double stepSize, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold, std::function<VectorXd(VectorXd &, VectorXd &, ArrayXd &)> subgradient,  ArrayXd lambdaj, bool sampleSizeAdjustment,double eps, bool trace);


#endif /* sgd_h */
