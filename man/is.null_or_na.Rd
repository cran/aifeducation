% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/aux_fct.R
\name{is.null_or_na}
\alias{is.null_or_na}
\title{Check if NULL or NA}
\usage{
is.null_or_na(object)
}
\arguments{
\item{object}{An object to test.}
}
\value{
Returns \code{FALSE} if the object is not \code{NULL} and not \code{NA}.
Returns \code{TRUE} in all other cases.
}
\description{
Function for checking if an object is \code{NULL} or \code{NA}
}
\seealso{
Other Auxiliary Functions: 
\code{\link{array_to_matrix}()},
\code{\link{calc_standard_classification_measures}()},
\code{\link{check_embedding_models}()},
\code{\link{clean_pytorch_log_transformers}()},
\code{\link{create_iota2_mean_object}()},
\code{\link{create_synthetic_units}()},
\code{\link{generate_id}()},
\code{\link{get_coder_metrics}()},
\code{\link{get_folds}()},
\code{\link{get_n_chunks}()},
\code{\link{get_stratified_train_test_split}()},
\code{\link{get_synthetic_cases}()},
\code{\link{get_train_test_split}()},
\code{\link{matrix_to_array_c}()},
\code{\link{split_labeled_unlabeled}()},
\code{\link{summarize_tracked_sustainability}()},
\code{\link{to_categorical_c}()}
}
\concept{Auxiliary Functions}
\keyword{internal}
