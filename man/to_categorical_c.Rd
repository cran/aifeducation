% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{to_categorical_c}
\alias{to_categorical_c}
\title{Transforming classes to one-hot encoding}
\usage{
to_categorical_c(class_vector, n_classes)
}
\arguments{
\item{class_vector}{\code{vector} containing integers for every class. The
integers must range from 0 to n_classes-1.}

\item{n_classes}{\code{int} Total number of classes.}
}
\value{
Returns a \code{matrix} containing the binary representation for
every class.
}
\description{
Function written in C++ transforming a vector of classes (int) into
a binary class matrix.
}
\seealso{
Other Auxiliary Functions: 
\code{\link{get_alpha_3_codes}()},
\code{\link{matrix_to_array_c}()},
\code{\link{summarize_tracked_sustainability}()}
}
\concept{Auxiliary Functions}
