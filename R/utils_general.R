# This file is part of the R package "aifeducation".
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

#' @title Get file extension
#' @description Function for requesting the file extension
#'
#' @param file_path `string` Path to a file.
#' @return Returns the extension of a file as a string.
#'
#' @family Utils
#' @export
get_file_extension<-function(file_path){
  extension <- stringi::stri_split_fixed(file_path, pattern = ".")[[1]]
  extension <- stringi::stri_trans_tolower(extension[[length(extension)]])
  return(extension)
}

#' @title Check if NULL or NA
#' @description Function for checking if an object is `NULL` or .
#'
#' @param object An object to test.
#' @return Returns `FALSE` if the object is not `NULL` and not `NA`. Returns `TRUE` in all other cases.
#'
#' @family Utils
#' @export
is.null_or_na <- function(object) {
  if(is.null(object)){
    return(TRUE)
  } else {
    if(sum(is.na(object))==length(object)){
      return(TRUE)
    } else {
      return(FALSE)
    }
  }
}

#' @title Clean pytorch log of transformers
#' @description Function for preparing and cleaning the log created by an object of class Trainer from the python
#'   library 'transformer's.
#'
#' @param log `data.frame` containing the log.
#' @return Returns a `data.frame` containing epochs, loss, and val_loss.
#'
#' @family Utils
#' @export
clean_pytorch_log_transformers <- function(log) {
  max_epochs <- max(log$epoch)

  cols <- c("epoch", "loss", "val_loss")

  cleaned_log <- matrix(
    data = NA,
    nrow = max_epochs,
    ncol = length(cols)
  )
  colnames(cleaned_log) <- cols
  for (i in 1:max_epochs) {
    cleaned_log[i, "epoch"] <- i

    tmp_loss <- subset(log, log$epoch == i & is.na(log$loss) == FALSE)
    tmp_loss <- tmp_loss[1, "loss"]
    cleaned_log[i, "loss"] <- tmp_loss

    tmp_val_loss <- subset(log, log$epoch == i & is.na(log$eval_loss) == FALSE)
    tmp_val_loss <- tmp_val_loss[1, "eval_loss"]
    cleaned_log[i, "val_loss"] <- tmp_val_loss
  }
  return(as.data.frame(cleaned_log))
}

#' @title Generate ID suffix for objects
#' @description Function for generating an ID suffix for objects of class [TextEmbeddingModel],
#'   [TEClassifierRegular], and [TEClassifierProtoNet].
#'
#' @param length `int` determining the length of the id suffix.
#' @return Returns a `string` of the requested length.
#' @family Utils
#' @export
generate_id <- function(length = 16) {
  id_suffix <- NULL
  sample_values <- c(
    "a", "A",
    "b", "B",
    "c", "C",
    "d", "D",
    "e", "E",
    "f", "F",
    "g", "G",
    "h", "H",
    "i", "I",
    "j", "J",
    "k", "K",
    "l", "L",
    "m", "M",
    "n", "N",
    "o", "O",
    "p", "P",
    "q", "Q",
    "r", "R",
    "s", "S",
    "t", "T",
    "u", "U",
    "v", "V",
    "w", "W",
    "x", "X",
    "y", "Y",
    "z", "Z",
    seq(from = 0, to = 9, by = 1)
  )


  id_suffix <- sample(
    x = sample_values,
    size = length,
    replace = TRUE
  )
  id_suffix <- paste(id_suffix, collapse = "")
  return(id_suffix)
}

#' @title Print message
#' @description Prints a message `msg` if `trace` parameter is `TRUE` with current date with `message()` or `cat()`
#'   function.
#'
#' @param msg `string` Message that should be printed.
#' @param trace `bool` Silent printing (`FALSE`) or not (`TRUE`).
#' @param msg_fun `bool` value that determines what function should be used. `TRUE` for `message()`, `FALSE` for
#'   `cat()`.
#'
#' @return This function returns nothing.
#' @family Utils
#' @export
output_message <- function(msg, trace, msg_fun) {
  fun <- ifelse(msg_fun, message, cat)
  if (trace) fun(paste(date(), msg))
}

#' @title Print message (`message()`)
#' @description Prints a message `msg` if `trace` parameter is `TRUE` with current date with `message()` function.
#'
#' @param msg `string` Message that should be printed.
#' @param trace `bool` Silent printing (`FALSE`) or not (`TRUE`).
#'
#' @return This function returns nothing.
#' @family Utils
#' @export
print_message <- function(msg, trace) {
  output_message(msg, trace, TRUE)
}

#' @title Print message  (`cat()`)
#' @description Prints a message `msg` if `trace` parameter is `TRUE` with current date with `cat()` function.
#'
#' @param msg `string` Message that should be printed.
#' @param trace `bool` Silent printing (`FALSE`) or not (`TRUE`).
#'
#' @return This function returns nothing.
#' @family Utils
#' @keywords internal
#' @noRd
cat_message <- function(msg, trace) {
  output_message(msg, trace, FALSE)
}

#' @title Create directory if not exists
#' @description Check whether the passed `dir_path` directory exists. If not, creates a new directory and prints a `msg`
#'   message if `trace` is `TRUE`.
#'
#' @param dir_path `string` A new directory path that should be created.
#' @param trace `bool` Whether a `msg` message should be printed.
#' @param msg `string` A message that should be printed if `trace` is `TRUE`.
#' @param msg_fun `func` Function used for printing the message.
#'
#' @return `TRUE` or `FALSE` depending on whether the shiny app is active.
#' @family Utils
#' @export
create_dir <- function(dir_path, trace, msg = "Creating Directory", msg_fun = TRUE) {
  if (!dir.exists(dir_path)) {
    output_message(msg, trace, msg_fun)
    dir.create(dir_path, recursive = TRUE)
  }
}

#' @title Run python file
#' @description Used to run python files with `reticulate::py_run_file()` from folder `python`.
#'
#' @param py_file_name `string` Name of a python file to run. The file must be in the `python` folder of `aifeducation`
#'   package.
#' @return This function returns nothing.
#'
#' @importFrom reticulate py_run_file
#'
#' @family Utils
#' @export
run_py_file <- function(py_file_name) {
  reticulate::py_run_file(system.file("python", py_file_name, package = "aifeducation"))
}

#' @title Number of cores for multiple tasks
#' @description Function for getting the number of cores that should be used
#' for parallel processing of tasks. The number of cores is set to 75 % of the
#' available cores. If the environment variable `CI` is set to `"true"` or if the
#' process is running on cran `2` is returned.
#'
#' @importFrom parallel detectCores
#'
#' @return Returns `int` as the number of cores.
#'
#' @family Utils
#' @export
auto_n_cores<-function(){
  if(Sys.getenv("CI")=="true"|
     Sys.getenv("NOT_CRAN")=="true"|
     Sys.getenv("_R_CHECK_LIMIT_CORES_")=="true"){
    n_cores=min(2,parallel::detectCores())
  } else {
    n_cores=floor(parallel::detectCores()*0.75)
  }
return(n_cores=max(1,n_cores))
}
