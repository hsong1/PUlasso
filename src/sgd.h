#ifndef sgd_h
#define sgd_h
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>
using namespace Eigen;
using namespace std;
std::tuple<VectorXd,double,VectorXd,VectorXd,int> GD(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, int N, std::function<VectorXd(int)> x, const VectorXd & ibeta, double stepSize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold, ArrayXd lambdaj, double eps, bool trace);

std::tuple<VectorXd,double,VectorXd,VectorXd,int> SGD(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, std::discrete_distribution<> dist,std::function<VectorXd(int)> x, const VectorXd & ibeta, const VectorXd stepSize, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold, ArrayXd lambdaj,double eps, bool trace);

std::tuple<VectorXd,double,VectorXd,VectorXd,int> SVRG(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, std::discrete_distribution<> dist,std::function<VectorXd(int)> x, const VectorXd & ibeta, double stepSize, int updateFreq, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold, ArrayXd lambdaj,double eps, bool trace);

std::tuple<VectorXd,double,VectorXd,VectorXd,int> SAG(std::function<double(const VectorXd &, const ArrayXd &)> objective,std::function<VectorXd(const VectorXd &,const ArrayXi &)> gradient, std::discrete_distribution<> dist,std::function<VectorXd(int)> x, const VectorXd & ibeta, double stepSize, int batchsize, int nIters, std::function<VectorXd(const VectorXd &, const ArrayXd &)> SoftThreshold, ArrayXd lambdaj, bool sampleSizeAdjustment,double eps, bool trace);


#endif /* sgd_h */
