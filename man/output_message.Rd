% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils_general.R
\name{output_message}
\alias{output_message}
\title{Print message}
\usage{
output_message(msg, trace, msg_fun)
}
\arguments{
\item{msg}{\code{string} Message that should be printed.}

\item{trace}{\code{bool} Silent printing (\code{FALSE}) or not (\code{TRUE}).}

\item{msg_fun}{\code{bool} value that determines what function should be used. \code{TRUE} for \code{message()}, \code{FALSE} for
\code{cat()}.}
}
\value{
This function returns nothing.
}
\description{
Prints a message \code{msg} if \code{trace} parameter is \code{TRUE} with current date with \code{message()} or \code{cat()}
function.
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
\code{\link{print_message}()},
\code{\link{run_py_file}()}
}
\concept{Utils}
