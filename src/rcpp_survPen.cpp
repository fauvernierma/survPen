
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
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


//' Matrix cross-multiplication between two matrices
//'
//' @param Mat1 a matrix.
//' @param Mat2 another matrix.
//' @return prod the product t(Mat1)%*%Mat2
//' @export
// [[Rcpp::export(name="'%cross%'")]]
MatrixXd multcross(const Map<MatrixXd> Mat1, const Map<MatrixXd> Mat2){
 MatrixXd prod = Mat1.transpose()*Mat2;
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




//' colSums of a matrix
//'
//' @param Mat a matrix.
//' @return colSums(Mat)
//' @export
// [[Rcpp::export]]
VectorXd colSums2(Map<MatrixXd> Mat) {
  return(Mat.colwise().sum());
}


//' Derivative of a Choleski factor
//'
//' @param deriv_Vp derivatives of the Bayesian covariance matrix wrt rho (log smoothing parameters).
//' @param p number of regression parameters
//' @param R1 Choleski factor of Vp
//' @return a list containing the derivatives of R1 wrt rho (log smoothing parameters)
//' @export
// [[Rcpp::export]]
List deriv_R(const List deriv_Vp, const int p, const Map<MatrixXd> R1){
	
	const int nb_smooth = deriv_Vp.size();
	
	List deriv_R1(nb_smooth);
	
	for (int m = 0; m < nb_smooth; m++){

		deriv_R1[m] = MatrixXd::Zero(p,p);

		MatrixXd B = - as<Map<MatrixXd> >(deriv_Vp[m]);

		for (int i = 0 ; i < p; i++){
			for (int j = i ; j < p ; j++){// R and its derivatives are triangular superior

				if(i>0)
				{

					for (int k = 0 ; k < i ; k++){

						B(i,j) -= as<Map<MatrixXd> >(deriv_R1[m])(k,i)*R1(k,j)+R1(k,i)*as<Map<MatrixXd> >(deriv_R1[m])(k,j);

					}

				}

				if(i==j)
				{

					as<Map<MatrixXd> >(deriv_R1[m])(i,j) = 0.5*B(i,j)/R1(i,j);

				}
				else
				{

					as<Map<MatrixXd> >(deriv_R1[m])(i,j) = 1/R1(i,i)*(B(i,j)-R1(i,j)*as<Map<MatrixXd> >(deriv_R1[m])(i,i));
				
				}

			}

		}
		
		
	}

	return(deriv_R1);

}



//' Gradient vector of LCV and LAML wrt rho (log smoothing parameters)
//'
//' @param X_GL list of matrices (\code{length(X.GL)=n.legendre}) for Gauss-Legendre quadrature
//' @param GL_temp list of vectors used to make intermediate calculations and save computation time
//' @param haz_GL list of all the matrix-vector multiplications X.GL[[i]]\%*\%beta for Gauss Legendre integration in order to save computation time
//' @param deriv_rho_beta firt derivative of beta wrt rho (implicit differentiation)
//' @param weights vector of weights for Gauss-Legendre integration on [-1;1]
//' @param tm vector of midpoints times for Gauss-Legendre integration; tm = 0.5*(t1 - t0)
//' @param nb_smooth number of smoothing parameters
//' @param p number of regression parameters
//' @param n_legendre number of nodes for Gauss-Legendre quadrature
//' @param S_list List of all the rescaled penalty matrices multiplied by their associated smoothing parameters
//' @param temp_LAML temporary matrix used when method="LAML" to save computation time
//' @param Vp Bayesian covariance matrix
//' @param S_beta List such that S_beta[[i]]=S_list[[i]]\%*\%beta
//' @param beta vector of estimated regression parameters
//' @param inverse_new_S inverse of the penalty matrix
//' @param X design matrix for the model
//' @param temp_deriv3 temporary matrix for third derivatives calculation when type="net" to save computation time
//' @param event vector of right-censoring indicators
//' @param expected vector of expected hazard rates
//' @param type "net" or "overall"
//' @param Ve frequentist covariance matrix
//' @param mat_temp temporary matrix used when method="LCV" to save computation time
//' @param method criterion used to select the smoothing parameters. Should be "LAML" or "LCV"; default is "LAML"
//' @return List of objects with the following items:
//' \item{grad_rho}{gradient vector of LCV or LAML}
//' \item{deriv_rho_inv_Hess_beta}{List of first derivatives of Vp wrt rho}
//' \item{deriv_rho_Hess_unpen_beta}{List of first derivatives of the Hessian of the unpenalized log-likelihood wrt rho}
//' @export
// [[Rcpp::export]]		
List grad_rho(const List X_GL, const List GL_temp, const List haz_GL, 
const Map<MatrixXd> deriv_rho_beta, const Map<VectorXd> weights,
const Map<VectorXd> tm, const int nb_smooth, const int p, const int n_legendre,
const List S_list, const List temp_LAML, const Map<MatrixXd> Vp, const List S_beta, 
const Map<VectorXd> beta, const Map<MatrixXd> inverse_new_S,
const Map<MatrixXd> X, const Map<MatrixXd> temp_deriv3,
const Map<VectorXd> event, const Map<VectorXd> expected, const String type,
const Map<MatrixXd> Ve, const Map<MatrixXd> mat_temp, const String method) {
		
	VectorXd grad_rho = VectorXd::Zero(nb_smooth);
	
	List deriv_rho_Hess_unpen_beta(nb_smooth);
	
	List deriv_rho_inv_Hess_beta(nb_smooth);
	
	for (int j = 0; j < nb_smooth; j++)
     {
		 
				deriv_rho_Hess_unpen_beta[j] = MatrixXd::Zero(p,p);	
				
				for (int i = 0; i < n_legendre; i++)
				{

					as<Map<MatrixXd> >(deriv_rho_Hess_unpen_beta[j]) -= as<Map<MatrixXd> >(X_GL[i]).transpose() * ((as<Map<MatrixXd> >(X_GL[i]).array().colwise() * 
					(as<Map<VectorXd> >(as<List> (GL_temp[j])[i]).array()*tm.array()*weights(i))).matrix()) ;
			
			
				}
				
		
				if (type=="net"){

					as<Map<MatrixXd> >(deriv_rho_Hess_unpen_beta[j]) += X.transpose() * ((X.array().colwise() * 
					((temp_deriv3 * deriv_rho_beta.row(j).transpose()).array()*event.array()*expected.array())).matrix()) ;
						
					}
				
				deriv_rho_inv_Hess_beta[j] = -Vp * (as<Map<MatrixXd> >(deriv_rho_Hess_unpen_beta[j]) - as<Map<MatrixXd> >(S_list[j])) * Vp ;

				if (method=="LAML"){
					
					grad_rho(j) = 0.5*(as<Map<VectorXd> >(S_beta[j]).array() * beta.array()).sum() // grad_rho_ll
				
					- 0.5*(inverse_new_S.array()*as<Map<MatrixXd> >(temp_LAML[j]).array()).sum() // grad.rho.log.abs.S
				
					+ 0.5*(Vp.array()*(as<Map<MatrixXd> >(S_list[j]) - as<Map<MatrixXd> >(deriv_rho_Hess_unpen_beta[j])).array()).sum() ; //grad.rho.log.det.Hess.beta
				}
				
				if (method=="LCV"){
					
					grad_rho(j) = (mat_temp.array()*as<Map<MatrixXd> >(deriv_rho_Hess_unpen_beta[j]).array()).sum() + 
					
					(-Ve.array()*as<Map<MatrixXd> >(S_list[j]).array()).sum() ;
	 
				}
	 
	 }
	 

	return List::create( 
			_["grad_rho"]  = grad_rho, 
			_["deriv_rho_inv_Hess_beta"]  = deriv_rho_inv_Hess_beta, 
			_["deriv_rho_Hess_unpen_beta"] = deriv_rho_Hess_unpen_beta
	) ;
}
	

//' Hessian matrix of LCV and LAML wrt rho (log smoothing parameters)
//'
//' @param X_GL list of matrices (\code{length(X.GL)=n.legendre}) for Gauss-Legendre quadrature
//' @param X_GL_Q list of transformed matrices from X_GL in order to calculate only the diagonal of the fourth derivative of the likelihood
//' @param GL_temp list of vectors used to make intermediate calculations and save computation time
//' @param haz_GL list of all the matrix-vector multiplications X.GL[[i]]\%*\%beta for Gauss Legendre integration in order to save computation time
//' @param deriv2_rho_beta second derivatives of beta wrt rho (implicit differentiation)
//' @param deriv_rho_beta firt derivatives of beta wrt rho (implicit differentiation)
//' @param weights vector of weights for Gauss-Legendre integration on [-1;1]
//' @param tm vector of midpoints times for Gauss-Legendre integration; tm = 0.5*(t1 - t0)
//' @param nb_smooth number of smoothing parameters
//' @param p number of regression parameters
//' @param n_legendre number of nodes for Gauss-Legendre quadrature
//' @param deriv_rho_inv_Hess_beta list of first derivatives of Vp wrt rho
//' @param deriv_rho_Hess_unpen_beta list of first derivatives of Hessian of unpenalized log likelihood wrt rho
//' @param S_list List of all the rescaled penalty matrices multiplied by their associated smoothing parameters
//' @param minus_eigen_inv_Hess_beta vector of eigenvalues of Vp
//' @param temp_LAML temporary matrix used when method="LAML" to save computation time
//' @param temp_LAML2 temporary matrix used when method="LAML" to save computation time
//' @param Vp Bayesian covariance matrix
//' @param S_beta List such that S_beta[[i]]=S_list[[i]]\%*\%beta
//' @param beta vector of estimated regression parameters
//' @param inverse_new_S inverse of the penalty matrix
//' @param X design matrix for the model
//' @param X_Q transformed design matrix in order to calculate only the diagonal of the fourth derivative of the likelihood
//' @param temp_deriv3 temporary matrix for third derivatives calculation when type="net" to save computation time
//' @param temp_deriv4 temporary matrix for fourth derivatives calculation when type="net" to save computation time
//' @param event vector of right-censoring indicators
//' @param expected vector of expected hazard rates
//' @param type "net" or "overall"
//' @param Ve frequentist covariance matrix
//' @param deriv_rho_Ve list of derivatives of Ve wrt rho
//' @param mat_temp temporary matrix used when method="LCV" to save computation time
//' @param deriv_mat_temp list of derivatives of mat_temp wrt rho
//' @param eigen_mat_temp vector of eigenvalues of mat_temp
//' @param method criterion used to select the smoothing parameters. Should be "LAML" or "LCV"; default is "LAML"
//' @return Hessian matrix of LCV or LAML wrt rho
//' @export
// [[Rcpp::export]]
MatrixXd Hess_rho(const List X_GL, const List X_GL_Q, const List GL_temp, const List haz_GL, 
const List deriv2_rho_beta, const Map<MatrixXd> deriv_rho_beta, const Map<VectorXd> weights,
const Map<VectorXd> tm, const int nb_smooth, const int p, const int n_legendre,
const List deriv_rho_inv_Hess_beta, const List deriv_rho_Hess_unpen_beta, const List S_list, 
const Map<VectorXd> minus_eigen_inv_Hess_beta, const List temp_LAML, const List temp_LAML2,
const Map<MatrixXd> Vp, const List S_beta, const Map<VectorXd> beta, const Map<MatrixXd> inverse_new_S,
const Map<MatrixXd> X, const Map<MatrixXd> X_Q, const Map<MatrixXd> temp_deriv3, const Map<MatrixXd> temp_deriv4,
const Map<VectorXd> event, const Map<VectorXd> expected, const String type,
const Map<MatrixXd> Ve, const List deriv_rho_Ve, const Map<MatrixXd> mat_temp, const List deriv_mat_temp, 
const Map<VectorXd> eigen_mat_temp, const String method) {
		
	MatrixXd Hess_rho = MatrixXd::Zero(nb_smooth,nb_smooth);
	
	for (int j = 0; j < nb_smooth; j++)
     {
		 for (int j2 = 0; j2 < nb_smooth; j2++)
			{
				
				VectorXd  diag_deriv2_Hess_unpen_beta = VectorXd::Zero(p);	
				
				for (int i = 0; i < n_legendre; i++)
				{
				
					MatrixXd mat1 = (as<Map<MatrixXd> >(X_GL[i]).array().colwise() * as<Map<VectorXd> >(as<List> (GL_temp[j2])[i]).array()).matrix()  ;
					
					VectorXd vec1 = deriv_rho_beta.row(j);
		
		
					MatrixXd mat2 = (as<Map<MatrixXd> >(X_GL[i]).array().colwise() * as<Map<VectorXd> >(haz_GL[i]).array()).matrix() ;
		
					VectorXd vec2 = as<Map<MatrixXd> >(deriv2_rho_beta[j2]).row(j) ;
								
		
					VectorXd vec_temp = mat1 * vec1 + mat2 * vec2;
		
				
					diag_deriv2_Hess_unpen_beta -= (as<Map<MatrixXd> >(X_GL_Q[i]).array().square().colwise()*(
					vec_temp.array()*tm.array()*weights(i))).matrix().colwise().sum();
			
			
				}
				
				
				if (type=="net"){

						MatrixXd mat3 = (X.array().colwise()*(temp_deriv4*deriv_rho_beta.row(j2).transpose()).array()).matrix() ;

						diag_deriv2_Hess_unpen_beta += (X_Q.array().square().colwise() * (
						event.array()*expected.array()*
						((mat3*deriv_rho_beta.row(j).transpose())+
						temp_deriv3*as<Map<MatrixXd> >(deriv2_rho_beta[j2]).row(j).transpose()).array()
						)).matrix().colwise().sum() ;
							
						
					}
				
			
				if (method=="LAML"){
				
					Hess_rho(j,j2) = 0.5*(-as<Map<MatrixXd> >(deriv_rho_inv_Hess_beta[j2]).array()*
					(-as<Map<MatrixXd> >(deriv_rho_Hess_unpen_beta[j])+ as<Map<MatrixXd> >(S_list[j])).array()).sum() +
				
					0.5*(-minus_eigen_inv_Hess_beta.array()*diag_deriv2_Hess_unpen_beta.array()).sum() // Hess_rho_log_det_Hess_beta
					
					-0.5*(as<Map<MatrixXd> >(temp_LAML2[j2]).array()*as<Map<MatrixXd> >(temp_LAML[j]).array()).sum() + //Hess_rho_log_abs_S
					
					(as<Map<VectorXd> >(S_beta[j]).array() * deriv_rho_beta.row(j2).transpose().array() ).sum(); // Hess_rho_ll1
					
					
					if (j==j2){

						Hess_rho(j,j2) += 0.5*((Vp.array()*as<Map<MatrixXd> >(S_list[j2]).array()).sum() // Hess_rho_log_det_Hess_beta
						
						-(inverse_new_S.array()*as<Map<MatrixXd> >(temp_LAML[j2]).array()).sum() //-0.5*Hess_rho_log_abs_S
						
						+(as<Map<VectorXd> >(S_beta[j2]).array() * beta.array()).sum()); // Hess_rho_ll1

					}
				
				}
				
				
				if (method=="LCV"){
					
					Hess_rho(j,j2) = (eigen_mat_temp.array()*diag_deriv2_Hess_unpen_beta.array()).sum() +
					(as<Map<MatrixXd> >(deriv_mat_temp[j2]).array()*as<Map<MatrixXd> >(deriv_rho_Hess_unpen_beta[j]).array()).sum() +
					(-as<Map<MatrixXd> >(deriv_rho_Ve[j2]).array()*as<Map<MatrixXd> >(S_list[j]).array()).sum() ;

					if (j==j2){

						Hess_rho(j,j2) += (-Ve.array()*as<Map<MatrixXd> >(S_list[j2]).array()).sum() ;

					}
					
				}	
				
				
			}
	 
	 }
	 
	return(Hess_rho) ;
	
}





