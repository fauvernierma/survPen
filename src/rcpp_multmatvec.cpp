
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;                       // 'maps' rather than copies
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision

//' Matrix multiplication between two matrices
//'
//' @param Mat1 a matrix.
//' @param Mat2 another matrix.
//' @return prod the product Mat1%*%Mat2
//' @export
// [[Rcpp::export(name="'%mult%'")]]
MatrixXd multmat(const Map<MatrixXd> Mat1, const Map<MatrixXd> Mat2){
 MatrixXd prod = Mat1*Mat2;
 return(prod);
}

//' Matrix multiplication between a matrix and a vector
//'
//' @param Mat a matrix.
//' @param vec a vector.
//' @return prod the product Mat%*%vec
//' @export
// [[Rcpp::export(name="'%vec%'")]]
VectorXd multvec(const Map<MatrixXd> Mat, const Map<VectorXd> vec){
 VectorXd prod = Mat*vec;
 return(prod);
}
