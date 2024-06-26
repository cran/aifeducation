% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{matrix_to_array_c}
\alias{matrix_to_array_c}
\title{Reshape matrix to array}
\usage{
matrix_to_array_c(matrix, times, features)
}
\arguments{
\item{matrix}{\code{matrix} containing the sequential data.}

\item{times}{\code{uword} Number of sequences.}

\item{features}{\code{uword} Number of features within each sequence.}
}
\value{
Returns an array. The first dimension corresponds to the cases,
the second to the times, and the third to the features.
}
\description{
Function written in C++ for reshaping a matrix containing sequential data into
an array for use with keras.
}
\examples{
#matrix has shape (batch,times*features)
matrix<-matrix(data=c(1,1,1,2,2,2,
                     2,2,2,3,3,3,
                     1,1,1,1,1,1),
              nrow=3,
              byrow=TRUE)
matrix

#Transform matrix to a array
#array has shape (batch,times*features)
matrix_to_array_c(matrix=matrix,times=2,features=3)
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
\code{\link{is.null_or_na}()},
\code{\link{split_labeled_unlabeled}()},
\code{\link{summarize_tracked_sustainability}()},
\code{\link{to_categorical_c}()}
}
\concept{Auxiliary Functions}
