% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/saving_and_loading.R
\name{load_ai_model}
\alias{load_ai_model}
\title{Loading models created with 'aifeducation'}
\usage{
load_ai_model(model_dir, ml_framework = aifeducation_config$get_framework())
}
\arguments{
\item{model_dir}{Path to the directory where the model is stored.}

\item{ml_framework}{\code{string} Determines the machine learning framework
for using the model. Possible are \code{ml_framework="pytorch"} for 'pytorch',
\code{ml_framework="tensorflow"} for 'tensorflow', and \code{ml_framework="auto"}.
for using the framework used when saving the model.}
}
\value{
Returns an object of class \link{TextEmbeddingClassifierNeuralNet} or
\link{TextEmbeddingModel}.
}
\description{
Function for loading models created with 'aifeducation'.
}
\seealso{
Other Saving and Loading: 
\code{\link{save_ai_model}()}
}
\concept{Saving and Loading}
