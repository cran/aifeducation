## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  # install.packages("devtools")
#  devtools::install_github("FBerding/aifeducation",
#                           dependencies = TRUE)

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  reticulate::install_python()

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  reticulate::py_available(initialize = TRUE)

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  reticulate::install_miniconda()

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  install_py_modules(envname="aifeducation")

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  aifeducation::check_aif_py_modules(print=TRUE)

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  aifeducation::set_config_cpu_only()

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  aifeducation::set_config_gpu_low_memory()

## ---- include = TRUE, eval=FALSE----------------------------------------------
#  aifeducation::set_config_tf_logger()

