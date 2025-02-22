% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils_general.R
\name{print_message}
\alias{print_message}
\title{Print message (\code{message()})}
\usage{
print_message(msg, trace)
}
\arguments{
\item{msg}{\code{string} Message that should be printed.}

\item{trace}{\code{bool} Silent printing (\code{FALSE}) or not (\code{TRUE}).}
}
\value{
This function returns nothing.
}
\description{
Prints a message \code{msg} if \code{trace} parameter is \code{TRUE} with current date with \code{message()} function.
}
\seealso{
Other Utils: 
\code{\link{auto_n_cores}()},
\code{\link{clean_pytorch_log_transformers}()},
\code{\link{create_config_state}()},
\code{\link{create_dir}()},
\code{\link{generate_id}()},
\code{\link{get_file_extension}()},
\code{\link{get_py_package_versions}()},
\code{\link{is.null_or_na}()},
\code{\link{output_message}()},
\code{\link{run_py_file}()}
}
\concept{Utils}
