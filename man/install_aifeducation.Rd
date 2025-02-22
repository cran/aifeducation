% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/install_and_config.R
\name{install_aifeducation}
\alias{install_aifeducation}
\title{Install aifeducation on a machine}
\usage{
install_aifeducation(install_aifeducation_studio = TRUE)
}
\arguments{
\item{install_aifeducation_studio}{\code{bool} If \code{TRUE} all necessary R packages are installed for using AI for Education
Studio.}
}
\value{
Function does nothing return. It installs python, optional R packages, and necessary 'python' packages on a
machine.
}
\description{
Function for installing 'aifeducation' on a machine. This functions assumes that not 'python' and no
'miniconda' is installed. Only'pytorch' is installed.
}
\seealso{
Other Installation and Configuration: 
\code{\link{check_aif_py_modules}()},
\code{\link{install_py_modules}()},
\code{\link{set_transformers_logger}()}
}
\concept{Installation and Configuration}
