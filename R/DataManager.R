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

#' @title Data manager for classification tasks
#'
#' @description Abstract class for managing the data and samples during training a classifier. DataManagerClassifier is
#'   used with [TEClassifierRegular] and [TEClassifierProtoNet].
#'
#' @return Objects of this class are used for ensuring the correct data management for training different types of
#'   classifiers. Objects of this class are also used for data augmentation by creating synthetic cases with different
#'   techniques.
#' @family Data Management
#' @export
DataManagerClassifier <- R6::R6Class(
  classname = "DataManagerClassifier",
  public = list(
    #' @field config ('list')\cr
    #'   Field for storing configuration of the [DataManagerClassifier].
    config = NULL,

    #' @field state ('list')\cr
    #'   Field for storing the current state of the [DataManagerClassifier].
    state = list(
      iteration = NULL,
      step = NULL
    ),

    #' @field datasets ('list')\cr
    #'   Field for storing the data sets used during training. All elements of the list are data sets of class
    #'   `datasets.arrow_dataset.Dataset`. The following data sets are available:
    #'
    #' * data_labeled: all cases which have a label.
    #' * data_unlabeled: all cases which have no label.
    #' * data_labeled_synthetic: all synthetic cases with their corresponding labels.
    #' * data_labeled_pseudo: subset of data_unlabeled if pseudo labels were estimated by a classifier.
    #'
    datasets = NULL,

    #' @field name_idx ('named vector')\cr
    #'   Field for storing the pairs of indexes and names of every case. The pairs for labeled and unlabeled data are
    #'   separated.
    name_idx = NULL,

    #' @field samples ('list')\cr
    #'   Field for storing the assignment of every cases to a train, validation or test data set depending on the
    #'   concrete fold. Only the indexes and not the names are stored. In addition, the list contains the assignment for
    #'   the final training which excludes a test data set. If the [DataManagerClassifier] uses `i` folds the sample for
    #'   the final training can be requested with `i+1`.
    samples = NULL,

    #' @description Creating a new instance of this class.
    #' @param data_embeddings Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] from which the
    #'   [DataManagerClassifier] should be created.
    #' @param data_targets `factor` containing the labels for cases stored in `data_embeddings`. Factor must be named
    #'   and has to use the same names used in `data_embeddings`. Missing values are supported and should be supplied
    #'   (e.g., for pseudo labeling).
    #' @param folds `int` determining the number of cross-fold samples. Value must be at least 2.
    #' @param val_size `double` between 0 and 1, indicating the proportion of cases of each class which should be used
    #'   for the validation sample. The remaining cases are part of the training data.
    #' @param class_levels `vector` containing the possible levels of the labels.
    #' @param one_hot_encoding `bool` If `TRUE` all labels are converted to one hot encoding.
    #' @param add_matrix_map `bool` If `TRUE` all embeddings are transformed into a two dimensional matrix. The number
    #'   of rows equals the number of cases. The number of columns equals `times*features`.
    #' @param sc_methods `string` determining the technique used for creating synthetic cases.
    #' @param sc_min_k `int` determining the minimal number of neighbors during the creating of synthetic cases.
    #' @param sc_max_k `int` determining the minimal number of neighbors during the creating of synthetic cases.
    #' @param trace `bool` If `TRUE` information on the process are printed to the console.
    #' @param n_cores `int` Number of cores which should be used during the calculation of synthetic cases.
    #' @return Method returns an initialized object of class [DataManagerClassifier].
    initialize = function(data_embeddings,
                          data_targets,
                          folds = 5,
                          val_size = 0.25,
                          class_levels,
                          one_hot_encoding = TRUE,
                          add_matrix_map = TRUE,
                          sc_methods = "dbsmote",
                          sc_min_k = 1,
                          sc_max_k = 10,
                          trace = TRUE,
                          n_cores=auto_n_cores()) {
      # Checking Prerequisites---------------------------------------------------
      check_class(data_embeddings, c("EmbeddedText", "LargeDataSetForTextEmbeddings"), FALSE)
      check_class(data_targets, c("factor"), FALSE)
      if (is.null(names(data_targets))) {
        stop("data_targets must be a named factor.")
      }
      check_type(folds, "int")
      if (folds < 2) {
        stop("folds must be at least 2.")
      }
      check_type(val_size, "double")
      check_type(class_levels, "vector")
      check_type(one_hot_encoding, "bool")
      check_type(add_matrix_map, "bool")
      check_type(sc_methods, "string")
      check_type(sc_min_k, "int")
      check_type(sc_max_k, "int")
      check_type(trace, "bool")
      check_type(n_cores, "int",FALSE)

      # Create Dataset-------------------------------------------------------
      private$prepare_datasets(
        data_embeddings = data_embeddings,
        data_targets = data_targets,
        trace = trace
      )

      # Create indices name mapping for labeled and unlabeled data--------------
      private$create_indices_name_map()

      # Check for valid number of folds------------------------------------------
      fin_k_folds <- private$check_and_calculate_number_folds(folds)

      # Saving Configuration-----------------------------------------------------
      self$config$n_folds <- fin_k_folds
      self$config$val_size <- val_size
      self$config$one_hot_encoding <- one_hot_encoding
      self$config$add_matrix_map <- add_matrix_map
      self$config$class_levels <- class_levels
      self$config$n_classes <- length(class_levels)
      self$config$features <- data_embeddings$get_features()
      self$config$times <- data_embeddings$get_times()
      self$config$sc$methods <- sc_methods
      self$config$sc$max_k <- sc_max_k
      self$config$sc$min_k <- sc_min_k
      self$config$n_cores<-n_cores

      # Add one hot encoding if necessary
      if (self$config$one_hot_encoding == TRUE) {
        self$datasets$data_labeled <- private$add_one_hot_encoding(self$datasets$data_labeled)
      }

      # Add matrix map if necessary
      if (self$config$add_matrix_map == TRUE) {
        self$datasets$data_labeled <- private$add_matrix_form(self$datasets$data_labeled)
        self$datasets$data_unlabeled <- private$add_matrix_form(self$datasets$data_unlabeled)
      }

      # create folds
      private$create_folds()

      # create sample for final training
      private$create_final_sample()
    },

    #' @description Method for requesting the configuration of the [DataManagerClassifier].
    #' @return Returns a `list` storing the configuration of the [DataManagerClassifier].
    get_config = function() {
      return(self$config)
    },

    #' @description Method for requesting the complete labeled data set.
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing all cases with labels.
    get_labeled_data = function() {
      return(self$datasets$data_labeled)
    },

    #' @description Method for requesting the complete unlabeled data set.
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing all cases without labels.
    get_unlabeled_data = function() {
      return(self$datasets$data_unlabeled)
    },

    #' @description Method for requesting the assignments to train, validation, and test data sets for every fold and
    #'   the final training.
    #' @return Returns a `list` storing the assignments to a train, validation, and test data set for every fold. In the
    #'   case of the sample for the final training the test data set is always empty (`NULL`).
    get_samples = function() {
      return(self$samples)
    },

    #' @description Method for setting the current state of the [DataManagerClassifier].
    #' @param iteration `int` determining the current iteration of the training. That is iteration determines the fold
    #'   to use for training, validation, and testing. If *i* is the number of fold *i+1* request the sample for the
    #'   final training. For requesting the sample for the final training iteration can take a string `"final"`.
    #' @param step `int` determining the step for estimating and using pseudo labels during training. Only relevant if
    #'   training is requested with pseudo labels.
    #' @return Method does not return anything. It is used for setting the internal state of the DataManager.
    set_state = function(iteration, step = NULL) {
      check_type(step, "int", TRUE)

      if (is.numeric(iteration) == FALSE) {
        if (iteration == "final") {
          iteration <- self$config$n_folds + 1
        } else {
          stop(
            paste(
              iteration,
              "is not a valid input for iteration"
            )
          )
        }
      }

      self$state$iteration <- iteration
      self$state$step <- step
    },

    #' @description Method for requesting the number of folds the [DataManagerClassifier] can use with the current data.
    #' @return Returns the number of folds the [DataManagerClassifier] uses.
    get_n_folds = function() {
      return(self$config$n_folds)
    },

    #' @description Method for requesting the number of classes.
    #' @return Returns the number classes.
    get_n_classes = function() {
      return(self$config$n_classes)
    },

    #' @description Method for requesting descriptive sample statistics.
    #' @return Returns a table describing the absolute frequencies of the labeled and unlabeled data. The rows contain
    #'   the length of the sequences while the columns contain the labels.
    get_statistics = function() {
      if (!is.null(self$datasets)) {
        if (!is.null(self$datasets$data_labeled)) {
          self$datasets$data_labeled$set_format("np")
          length_labeled <- self$datasets$data_labeled["length"]
          labels_labeled <- self$datasets$data_labeled["labels"]

          if (!is.null(self$datasets$data_unlabeled)) {
            self$datasets$data_unlabeled$set_format("np")
            length_unlabeled <- self$datasets$data_unlabeled["length"]
            labels_unlabeled <- rep(NA, times = length(length_unlabeled))
          } else {
            length_unlabeled <- NULL
            labels_unlabeled <- NULL
          }

          statistics <- table(c(length_labeled, length_unlabeled),
                              c(labels_labeled, labels_unlabeled),
                              useNA = "ifany"
          )
          return(statistics)
        } else {
          stop("Labeled data not set.")
        }
      } else {
        stop("Data not set.")
      }
    },

    #' @description Method for requesting a data set for training depending in the current state of the
    #'   DataManagerClassifier.
    #' @param inc_labeled `bool` If `TRUE` the data set includes all cases which have labels.
    #' @param inc_unlabeled `bool` If `TRUE` the data set includes all cases which have no labels.
    #' @param inc_synthetic `bool` If `TRUE` the data set includes all synthetic cases with their corresponding labels.
    #' @param inc_pseudo_data `bool` If `TRUE` the data set includes all cases which have pseudo labels.
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing the requested kind of data along
    #'   with all requested transformations for training. Please note that this method returns a data sets that is
    #'   designed for training only. The corresponding validation data set is requested with `get_val_dataset` and the
    #'   corresponding test data set with `get_test_dataset`.
    get_dataset = function(inc_labeled = TRUE,
                           inc_unlabeled = FALSE,
                           inc_synthetic = FALSE,
                           inc_pseudo_data = FALSE) {
      # Checks
      check_type(inc_labeled, "bool", FALSE)
      check_type(inc_unlabeled, "bool", FALSE)
      check_type(inc_synthetic, "bool", FALSE)
      check_type(inc_pseudo_data, "bool", FALSE)

      data <- NULL

      requested_datasets <- list()

      if (inc_labeled == TRUE) {
        self$datasets$data_labeled$set_format("np")
        requested_datasets[length(requested_datasets) + 1] <- list(self$datasets$data_labeled$select(
          as.integer(self$samples[[self$state$iteration]]$train)
        ))
      }

      if (inc_synthetic == TRUE) {
        if (!is.null(self$datasets$data_labeled_synthetic)) {
          self$datasets$data_labeled_synthetic$set_format("np")
          requested_datasets[length(requested_datasets) + 1] <- list(self$datasets$data_labeled_synthetic)
        }
      }

      if (inc_pseudo_data == TRUE) {
        if (!is.null(self$datasets$data_labeled_pseudo)) {
        self$datasets$data_labeled_pseudo$set_format("np")
        requested_datasets[length(requested_datasets) + 1] <- list(self$datasets$data_labeled_pseudo)
        }
      }
      if (inc_unlabeled == TRUE) {
        if (!is.null(self$datasets$data_unlabeled)) {
        self$datasets$data_unlabeled$set_format("np")
        requested_datasets[length(requested_datasets) + 1] <- list(self$datasets$data_unlabeled)
        }
      }


      if (length(requested_datasets) < 1) {
        data <- NULL
      } else if (length(requested_datasets) == 1) {
        data <- requested_datasets[[1]]
      } else {
        data <- datasets$concatenate_datasets(dsets = requested_datasets, axis = 0L)
      }
      return(data)
    },

    #' @description Method for requesting a data set for validation depending in the current state of the
    #'   [DataManagerClassifier].
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing the requested kind of data along
    #'   with all requested transformations for validation. The corresponding data set for training can be requested
    #'   with `get_dataset` and the corresponding data set for testing with `get_test_dataset`.
    get_val_dataset = function() {
      return(self$datasets$data_labeled$select(as.integer(self$samples[[self$state$iteration]]$val)))
    },

    #' @description Method for requesting a data set for testing depending in the current state of the
    #'   DataManagerClassifier.
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing the requested kind of data along
    #'   with all requested transformations for validation. The corresponding data set for training can be requested
    #'   with `get_dataset` and the corresponding data set for validation with `get_val_dataset`.
    get_test_dataset = function() {
      if (self$state$iteration <= self$get_n_folds()) {
        return(self$datasets$data_labeled$select(as.integer(self$samples[[self$state$iteration]]$test)))
      } else {
        return(NULL)
      }
    },

    #' @description Method for generating synthetic data used during training. The process uses all labeled data
    #'   belonging to the current state of the DataManagerClassifier.
    #' @param  trace `bool` If `TRUE` information on the process are printed to the console.
    #' @param inc_pseudo_data `bool` If `TRUE` data with pseudo labels are used in addition to the labeled data for
    #'   generating synthetic cases.
    #' @return This method does nothing return. It generates a new data set for synthetic cases which are stored as an
    #'   object of class `datasets.arrow_dataset.Dataset` in the field `datasets$data_labeled_synthetic`. Please note
    #'   that a call of this method will override an existing data set in the corresponding field.
    create_synthetic = function(trace = TRUE,
                                inc_pseudo_data = FALSE) {
      # checks
      check_type(trace, "bool", FALSE)
      check_type(inc_pseudo_data, "bool", FALSE)

      # Print status message to console
      if (trace == TRUE) {
        if (self$state$iteration <= self$config$n_folds) {
          message(paste(
            date(),
            "|", "Iteration", self$state$iteration, "from", self$config$n_folds,
            "|", "Generating synthetic cases"
          ))
        } else {
          message(paste(
            date(),
            "|", "Final training",
            "|", "Generating synthetic cases"
          ))
        }
      }

      data <- self$get_dataset(
        inc_labeled = TRUE,
        inc_unlabeled = FALSE,
        inc_synthetic = FALSE,
        inc_pseudo_data = inc_pseudo_data
      )

      # Set up parallel processing
      requireNamespace(package = "foreach", quietly = TRUE)
      cl <- parallel::makeCluster(self$config$n_cores)
      doParallel::registerDoParallel(cl)

      # Create Synthetic Cases
      data$set_format("np")
      syn_cases <- get_synthetic_cases_from_matrix(
        matrix_form = data["matrix_form"],
        target = data["labels"],
        sequence_length = data["length"],
        method = self$config$sc$methods,
        min_k = self$config$sc$min_k,
        max_k = self$config$sc$max_k,
        times = self$config$times,
        features = self$config$features
      )

      # Unload cluster for parallel processing
      parallel::stopCluster(cl)

      # Post-process synthetic cases
      embeddings_syntehtic <- syn_cases$syntetic_embeddings
      targets_synthetic <- as.numeric(as.character(syn_cases$syntetic_targets))

      # Check if cases with zero length are included
      length <- get_n_chunks(
        text_embeddings = embeddings_syntehtic,
        features = self$config$features,
        times = self$config$times
      )
      idx_non_zero_length <- which(x = (length > 0))

      # exclude cases with zero length
      embeddings_syntehtic <- embeddings_syntehtic[idx_non_zero_length, , , drop = FALSE]
      targets_synthetic <- targets_synthetic[idx_non_zero_length]

      if (dim(embeddings_syntehtic)[1] > 1) {
        # Add or replace current dataset for synthetic cases
        self$datasets$data_labeled_synthetic <- datasets$Dataset$from_dict(
          reticulate::dict(
            list(
              id = rownames(embeddings_syntehtic),
              input = prepare_r_array_for_dataset(embeddings_syntehtic),
              labels = targets_synthetic,
              length = get_n_chunks(
                text_embeddings = embeddings_syntehtic,
                features = self$config$features,
                times = self$config$times
              )
            ),
            convert = FALSE
          )
        )


        if (self$config$one_hot_encoding == TRUE) {
          self$datasets$data_labeled_synthetic <- private$add_one_hot_encoding(self$datasets$data_labeled_synthetic)
        }

        if (self$config$add_matrix_map == TRUE) {
          self$datasets$data_labeled_synthetic <- private$add_matrix_form(self$datasets$data_labeled_synthetic)
        }
      } else {
        self$datasets$data_labeled_synthetic <- NULL
      }

      if (trace == TRUE) {
        if (self$state$iteration <= self$config$n_folds) {
          message(paste(
            date(),
            "|", "Iteration", self$state$iteration, "from", self$config$n_folds,
            "|", "Generating synthetic cases done"
          ))
        } else {
          message(paste(
            date(),
            "|", "Final training",
            "|", "Generating synthetic cases done"
          ))
        }
      }
    },

    #' @description Method for adding data with pseudo labels generated by a classifier
    #' @param inputs `array` or `matrix` representing the input data.
    #' @param labels `factor` containing the corresponding pseudo labels.
    #' @return This method does nothing return. It generates a new data set for synthetic cases which are stored as an
    #'   object of class `datasets.arrow_dataset.Dataset` in the field `datasets$data_labeled_pseudo`. Please note that
    #'   a call of this method will override an existing data set in the corresponding field.
    add_replace_pseudo_data = function(inputs,
                                       labels) {
      # checks
      check_class(inputs, c("array", "matrix"), FALSE)
      check_class(labels, c("factor"), FALSE)

      private$check_labels(labels)
      # Add or replace current dataset for pseudo data
      self$datasets$data_labeled_pseudo <- datasets$Dataset$from_dict(
        reticulate::dict(
          list(
            id = rownames(inputs),
            input = prepare_r_array_for_dataset(inputs),
            labels = as.numeric(labels) - 1,
            length = get_n_chunks(text_embeddings = inputs, features = self$config$features, times = self$config$times)
          ),
          convert = FALSE
        )
      )

      if (self$config$one_hot_encoding == TRUE) {
        self$datasets$data_labeled_pseudo <- private$add_one_hot_encoding(self$datasets$data_labeled_pseudo)
      }

      if (self$config$add_matrix_map == TRUE) {
        self$datasets$data_labeled_pseudo <- private$add_matrix_form(self$datasets$data_labeled_pseudo)
      }
    }
    #-----------------------------------------------------------------------------
  ),
  private = list(
    prepare_datasets = function(data_embeddings, data_targets, trace) {
      if ("EmbeddedText" %in% class(data_embeddings)) {
        data_set_embeddings <- data_embeddings$convert_to_LargeDataSetForTextEmbeddings()
        data_set_embeddings <- data_set_embeddings$get_dataset()
      } else {
        data_set_embeddings <- data_embeddings$get_dataset()
      }

      # Reduce embeddings to unique ids
      n_init_cases <- data_set_embeddings$num_rows
      data_set_embeddings <- reduce_to_unique(data_set_embeddings, "id")
      n_final_cases <- data_set_embeddings$num_rows

      # Prepare classes for join
      class_vector <- vector(length = n_final_cases)
      class_vector[] <- NA
      names(class_vector) <- data_set_embeddings["id"]

      # Convert labels
      data_targets <- na.omit(data_targets)
      data_targets_names <- names(data_targets)
      data_targets <- as.numeric(data_targets) - 1
      names(data_targets) <- data_targets_names

      # Estimate all targets which have an embedding input
      relevant_target_names <- intersect(names(class_vector), y = names(data_targets))

      # Add relevant data to the class vector
      class_vector[relevant_target_names] <- data_targets[relevant_target_names]

      # Sort class vector according to the input ids
      class_vector <- class_vector[data_set_embeddings["id"]]

      # Get indices for labeled-unlabeled split
      indices_unlabeled <- which(is.na(class_vector)) - 1
      indices_labeled <- which(!is.na(class_vector)) - 1

      # Add labels to data set
      data_set_embeddings <- data_set_embeddings$add_column("labels", np$array(class_vector))

      # Split data
      self$datasets$data_labeled <- data_set_embeddings$select(as.integer(indices_labeled))
      if (length(indices_unlabeled) != 0) {
        self$datasets$data_unlabeled <- data_set_embeddings$select(as.integer(indices_unlabeled))
      } else {
        self$datasets$data_unlabeled <- NULL
      }


      # Report numbers
      if (trace == TRUE) {
        message(paste(
          date(),
          "Total Cases:", n_init_cases,
          "Unique Cases:", n_final_cases,
          "Labeled Cases:", length(indices_labeled)
        ))
      }
    },
    #--------------------------------------------------------------------------
    get_all_labels=function(){
      self$datasets$data_labeled$set_format("np")
      self$datasets$data_labeled["labels"]
    },
    #-------------------------------------------------------------------------
    check_and_calculate_number_folds = function(folds) {
      sample_target <- private$get_all_labels()
      freq_cat <- table(sample_target)
      min_freq <- min(freq_cat)
      if (min_freq < 6) {
        stop(paste("Frequency of the smallest category/class is", min_freq, ". At least
                   6 cases are necessary. Consider to remove this category/class."))
      } else {
        if (min_freq / folds < 3) {
          fin_k_folds <- min_freq
          warning(paste("Frequency of the smallest category/class is not sufficent to ensure
                    at least 3 cases per fold. Adjusting number of folds from ", folds, "to", fin_k_folds, "."))
        } else {
          fin_k_folds <- folds
        }
      }
      return(fin_k_folds)
    },
    add_matrix_form = function(dataset) {
      if (!is.null(dataset)) {
        private$load_reload_python_scripts()
        dataset <- dataset$map(py$map_input_to_matrix_form, fn_kwargs = list(
          times = as.integer(self$config$times),
          features = as.integer(self$config$features)
        ))
        return(dataset)
      } else {
        return(NULL)
      }
    },
    add_one_hot_encoding = function(dataset) {
      if (!is.null(dataset)) {
        private$load_reload_python_scripts()
        dataset <- dataset$map(py$map_labels_to_one_hot,
          fn_kwargs = reticulate::dict(list(num_classes = as.integer(self$config$n_classes)))
        )
        return(dataset)
      } else {
        return(NULL)
      }
    },
    load_reload_python_scripts = function() {
      reticulate::py_run_file(system.file("python/py_functions.py",
        package = "aifeducation"
      ))
    },
    create_indices_name_map = function() {
      self$name_idx$labeled_data <- seq.int(from = 0, to = (length(self$datasets$data_labeled["id"])) - 1)
      names(self$name_idx$labeled_data) <- self$datasets$data_labeled["id"]

      if (!is.null(self$datasets$data_unlabeled)) {
        self$name_idx$unlabeled_data <- seq.int(from = 0, to = (length(self$datasets$data_unlabeled["id"])) - 1)
        names(self$name_idx$unlabeled_data) <- self$datasets$data_unlabeled["id"]
      } else {
        self$name_idx$unlabeled_data <- NULL
      }
    },
    create_folds = function() {
      # Create Train, Test, and Validation Samples--------------------------------
      # Check maximal number of folds, adjust, and create folds
      data_targets <- factor(x = self$datasets$data_labeled["labels"])
      names(data_targets) <- self$datasets$data_labeled["id"]

      folds <- get_folds(
        target = data_targets,
        k_folds = self$config$n_folds
      )

      self$samples <- NULL
      for (i in 1:self$config$n_folds) {
        train_val_split <- get_train_test_split(
          embedding = NULL,
          target = data_targets[folds$train_sample[[i]]],
          val_size = self$config$val_size
        )

        fold <- list(
          train = self$name_idx$labeled_data[names(train_val_split$target_train)],
          val = self$name_idx$labeled_data[names(train_val_split$target_test)],
          test = self$name_idx$labeled_data[folds$val_sample[[i]]]
        )

        self$samples[i] <- list(fold)
      }
    },
    create_final_sample = function(data_targets) {
      data_targets <- factor(x = self$datasets$data_labeled["labels"])
      names(data_targets) <- self$datasets$data_labeled["id"]

      names_final_split <- get_stratified_train_test_split(
        targets = data_targets,
        val_size = self$config$val_size
      )
      final_split <- list(
        train = self$name_idx$labeled_data[names_final_split$train_sample],
        val = self$name_idx$labeled_data[names_final_split$test_sample],
        test = NULL
      )
      self$samples[length(self$samples) + 1] <- list(final_split)
    },
    check_labels = function(labels) {
      if (is.factor(labels) == FALSE) {
        stop("labels must be an object of class factor.")
      } else {
        levels_identical <- sum(levels(labels) == self$config$class_levels)
        if (levels_identical != self$config$n_classes) {
          stop(paste(
            "Levels of the labels are not identical with the levels of the classifier.",
            "Necessary levels:", self$config$class_levels,
            "Provided levels":levels(labels)
          ))
        }
      }
    }
  )
)
