// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// matrix_to_array_c
arma::cube matrix_to_array_c(arma::mat matrix, arma::uword times, arma::uword features);
RcppExport SEXP _aifeducation_matrix_to_array_c(SEXP matrixSEXP, SEXP timesSEXP, SEXP featuresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type matrix(matrixSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type times(timesSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type features(featuresSEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_to_array_c(matrix, times, features));
    return rcpp_result_gen;
END_RCPP
}
// to_categorical_c
arma::mat to_categorical_c(arma::vec class_vector, arma::uword n_classes);
RcppExport SEXP _aifeducation_to_categorical_c(SEXP class_vectorSEXP, SEXP n_classesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type class_vector(class_vectorSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type n_classes(n_classesSEXP);
    rcpp_result_gen = Rcpp::wrap(to_categorical_c(class_vector, n_classes));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_aifeducation_matrix_to_array_c", (DL_FUNC) &_aifeducation_matrix_to_array_c, 3},
    {"_aifeducation_to_categorical_c", (DL_FUNC) &_aifeducation_to_categorical_c, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_aifeducation(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
