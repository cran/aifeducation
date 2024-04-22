## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  install.packages("aifeducation")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  reticulate::install_python()

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  reticulate::py_available(initialize = TRUE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  reticulate::install_miniconda()

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  #For Linux
#  aifeducation::install_py_modules(envname="aifeducation",
#                                   install="all",
#                                   remove_first=FALSE,
#                                   tf_version="<=2.15",
#                                   pytorch_cuda_version = "12.1"
#                                   cpu_only=FALSE)
#  
#  #For Windows and MacOS
#  aifeducation::install_py_modules(envname="aifeducation",
#                                   install="all",
#                                   remove_first=FALSE,
#                                   tf_version="<=2.15",
#                                   pytorch_cuda_version = "12.1"
#                                   cpu_only=TRUE)
#  

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  aifeducation::check_aif_py_modules(print=TRUE,
#                                     check="pytorch")
#  
#  aifeducation::check_aif_py_modules(print=TRUE,
#                                     check="tensorflow")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  aifeducation::set_config_cpu_only()

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  aifeducation::set_config_gpu_low_memory()

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  aifeducation::set_config_tf_logger()

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  reticulate::use_condaenv(condaenv = "aifeducation")
#  library(aifeducation)
#  set_transformers_logger("ERROR")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  #For tensorflow
#  aifeducation_config$set_global_ml_backend("tensorflow")
#  
#  #For PyTorch
#  aifeducation_config$set_global_ml_backend("pytorch")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  #if you would like to use only cpus
#  set_config_cpu_only()
#  
#  #if you have a graphic device with low memory
#  set_config_gpu_low_memory()
#  
#  #if you would like to reduce the tensorflow output to errors
#  set_config_os_environ_logger(level = "ERROR")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  #For Linux
#  aifeducation::install_py_modules(envname="aifeducation",
#                                   install="all",
#                                   remove_first=TRUE,
#                                   tf_version="<=2.14",
#                                   pytorch_cuda_version = "12.1"
#                                   cpu_only=FALSE)
#  
#  #For Windows with gpu support
#  aifeducation::install_py_modules(envname="aifeducation",
#                                   install="all",
#                                   remove_first=TRUE,
#                                   tf_version="<=2.10",
#                                   pytorch_cuda_version = "12.1"
#                                   cpu_only=FALSE)
#  #For Windows without gpu support
#  aifeducation::install_py_modules(envname="aifeducation",
#                                   install="all",
#                                   remove_first=TRUE,
#                                   tf_version="<=2.14",
#                                   pytorch_cuda_version = "12.1"
#                                   cpu_only=TRUE)
#  
#  #For MacOS
#  aifeducation::install_py_modules(envname="aifeducation",
#                                   install="all",
#                                   remove_first=TRUE,
#                                   tf_version="<=2.14",
#                                   pytorch_cuda_version = "12.1"
#                                   cpu_only=TRUE)

