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

#' @title Embedded text
#' @description Object of class `R6` which stores the text embeddings generated by an object of class
#'   [TextEmbeddingModel]. The text embeddings are stored within memory/RAM. In the case of a high number of documents
#'   the data may not fit into memory/RAM. Thus, please use this object only for a small sample of texts. In general, it
#'   is recommended to use an object of class [LargeDataSetForTextEmbeddings] which can deal with any number of texts.
#' @return Returns an object of class [EmbeddedText]. These objects are used for storing and managing the text
#'   embeddings created with objects of class [TextEmbeddingModel]. Objects of class [EmbeddedText] serve as input for
#'   objects of class [TEClassifierRegular], [TEClassifierProtoNet], and [TEFeatureExtractor]. The main aim of this
#'   class is to provide a structured link between embedding models and classifiers. Since objects of this class save
#'   information on the text embedding model that created the text embedding it ensures that only embedding generated
#'   with same embedding model are combined. Furthermore, the stored information allows objects to check if embeddings
#'   of the correct text embedding model are used for training and predicting.
#' @family Data Management
#' @export
EmbeddedText <- R6::R6Class(
  classname = "EmbeddedText",
  private = list(

    # model_name string Name of the model that generates this embedding.
    model_name = NA,


    # Label of the model that generates this embedding.
    model_label = NA,


    # Date when the embedding generating model was created.
    model_date = NA,


    # Method of the underlying embedding model
    model_method = NA,


    # Version of the model that generated this embedding.
    model_version = NA,


    # Language of the model that generated this embedding.
    model_language = NA,


    # Maximal number of tokens that processes the generating model for a chunk.
    param_seq_length = NA,


    # Number of tokens that were added at the beginning of the sequence for the next chunk
    # by this model.
    param_overlap = NA,


    # Maximal number of chunks which are supported by the generating model.
    param_chunks = NA,

    # Features of the embeddings
    param_features = NA,

    # Minimal layer to be included in the creation of embeddings.
    param_emb_layer_min = NA,

    # Maximal layer to be included in the creation of embeddings.
    param_emb_layer_max = NA,

    # Type of pooling tokens embeddings within each layer.
    param_emb_pool_type = NA,


    # Aggregation method of the hidden states. Deprecated. Included for backward
    # compatibility.
    param_aggregation = NA,

    # List containing information on the feature extractor if the embeddings
    # are compressed.
    feature_extractor = list(),

    # Variable for checking if the object is successfully configured. Only is
    # this is TRUE the object can be used
    configured = FALSE,

    # Method for setting configured to TRUE
    set_configuration_to_TRUE = function() {
      private$configured <- TRUE
    },

    # Method for checking if the configuration is done successfully
    check_config_for_TRUE = function() {
      if (private$configured == FALSE) {
        stop("The object is not configured. Please call the method configure.")
      }
    }
  ),
  public = list(
    #' @field embeddings ('data.frame()')\cr
    #'   data.frame containing the text embeddings for all chunks. Documents are in the rows. Embedding dimensions are
    #'   in the columns.
    embeddings = NA,

    #' @description Creates a new object representing text embeddings.
    #' @param model_name `string` Name of the model that generates this embedding.
    #' @param model_label `string` Label of the model that generates this embedding.
    #' @param model_date `string` Date when the embedding generating model was created.
    #' @param model_method `string` Method of the underlying embedding model.
    #' @param model_version `string` Version of the model that generated this embedding.
    #' @param model_language `string` Language of the model that generated this embedding.
    #' @param param_seq_length `int` Maximum number of tokens that processes the generating model for a chunk.
    #' @param param_chunks `int` Maximum number of chunks which are supported by the generating model.
    #' @param param_features `int` Number of dimensions of the text embeddings.
    #' @param param_overlap `int` Number of tokens that were added at the beginning of the sequence for the next chunk
    #'   by this model.    #'
    #' @param param_emb_layer_min `int` or `string` determining the first layer to be included in the creation of
    #'   embeddings.
    #' @param param_emb_layer_max `int` or `string` determining the last layer to be included in the creation of
    #'   embeddings.
    #' @param param_emb_pool_type `string` determining the method for pooling the token embeddings within each layer.
    #' @param param_aggregation `string` Aggregation method of the hidden states. Deprecated. Only included for backward
    #'   compatibility.
    #' @param embeddings `data.frame` containing the text embeddings.
    #' @return Returns an object of class [EmbeddedText] which stores the text embeddings produced by an objects of
    #'   class [TextEmbeddingModel].
    configure = function(model_name = NA,
                         model_label = NA,
                         model_date = NA,
                         model_method = NA,
                         model_version = NA,
                         model_language = NA,
                         param_seq_length = NA,
                         param_chunks = NULL,
                         param_features = NULL,
                         param_overlap = NULL,
                         param_emb_layer_min = NULL,
                         param_emb_layer_max = NULL,
                         param_emb_pool_type = NULL,
                         param_aggregation = NULL,
                         embeddings) {
      private$model_name <- model_name
      private$model_label <- model_label
      private$model_date <- model_date
      private$model_method <- model_method
      private$model_version <- model_version
      private$model_language <- model_language
      private$param_seq_length <- param_seq_length
      private$param_chunks <- param_chunks
      private$param_features <- param_features
      private$param_overlap <- param_overlap


      private$param_emb_layer_min <- param_emb_layer_min
      private$param_emb_layer_max <- param_emb_layer_max
      private$param_emb_pool_type <- param_emb_pool_type

      private$param_aggregation <- param_aggregation

      self$embeddings <- embeddings
    },
    #--------------------------------------------------------------------------
    #' @description Saves a data set to disk.
    #' @param dir_path Path where to store the data set.
    #' @param folder_name `string` Name of the folder for storing the data set.
    #' @param create_dir `bool` If `True` the directory will be created if it does not exist.
    #' @return Method does not return anything. It write the data set to disk.
    save = function(dir_path, folder_name, create_dir = TRUE) {
      # Create directory
      if (dir.exists(dir_path) == FALSE) {
        if (create_dir == TRUE) {
          dir.create(dir_path)
        } else {
          stop("Directory does not exist.")
        }
      }

      #save date
      data_embeddings=self$embeddings

      save(
        data_embeddings,
        file = paste0(dir_path,"/",folder_name,"/data.rda")
        )
    },
    #-------------------------------------------------------------------------
    #' @description Method for checking if the model was successfully configured. An object can only be used if this
    #'   value is `TRUE`.
    #' @return `bool` `TRUE` if the model is fully configured. `FALSE` if not.
    is_configured = function() {
      return(private$configured)
    },
    #--------------------------------------------------------------------------
    #' @description loads an object of class [EmbeddedText] from disk and updates the object to the current version of
    #'   the package.
    #' @param dir_path Path where the data set set is stored.
    #' @return Method does not return anything. It loads an object from disk.
    load_from_disk = function(dir_path) {
      if (self$is_configured() == TRUE) {
        stop("The object has already been configured.")
      }

      # Load R file
      config_file <- load_R_config_state(dir_path)

      # Set configuration
      self$configure(
        model_name = config_file$private$model_name,
        model_label = config_file$private$model_label,
        model_date = config_file$private$model_date,
        model_method = config_file$private$model_method,
        model_version = config_file$private$model_version,
        model_language = config_file$private$model_language,
        param_seq_length = config_file$private$param_seq_length,
        param_chunks = config_file$private$param_chunks,
        param_features = config_file$private$param_features,
        param_overlap = config_file$private$param_overlap,
        param_emb_layer_min = config_file$private$param_emb_layer_min,
        param_emb_layer_max = config_file$private$param_emb_layer_max,
        param_emb_pool_type = config_file$private$param_emb_pool_type,
        param_aggregation = config_file$private$param_aggregation,
        embeddings = NULL
      )

      # Check for feature extractor and add information
      if (is.null_or_na(config_file$private$feature_extractor$model_name) == FALSE) {
        self$add_feature_extractor_info(
          model_name = config_file$private$feature_extractor$model_name,
          model_label = config_file$private$feature_extractor$model_label,
          features = config_file$private$feature_extractor$features,
          method = config_file$private$feature_extractor$method,
          noise_factor = config_file$private$feature_extractor$noise_factor,
          optimizer = config_file$private$feature_extractor$optimizer
        )
      }

      #Load data
      data=load(paste0(dir_path,"/","data.rda"))
      data=get(data)
      self$embeddings=data
    },
    #--------------------------------------------------------------------------
    #' @description Method for retrieving information about the model that generated this embedding.
    #' @return `list` contains all saved information about the underlying text embedding model.
    get_model_info = function() {
      tmp <- list(
        model_name = private$model_name,
        model_label = private$model_label,
        model_date = private$model_date,
        model_method = private$model_method,
        model_version = private$model_version,
        model_language = private$model_language,
        param_seq_length = private$param_seq_length,
        param_chunks = private$param_chunks,
        param_features = private$param_features,
        param_overlap = private$param_overlap,
        param_emb_layer_min = private$param_emb_layer_min,
        param_emb_layer_max = private$param_emb_layer_max,
        param_emb_pool_type = private$param_emb_pool_type,
        param_aggregation = private$param_aggregation
      )
      return(tmp)
    },

    #--------------------------------------------------------------------------
    #' @description Method for retrieving the label of the model that generated this embedding.
    #' @return `string` Label of the corresponding text embedding model
    get_model_label = function() {
      return(private$transformer_components$ml_framework)
    },

    #--------------------------------------------------------------------------
    #' @description Number of chunks/times of the text embeddings.
    #' @return Returns an `int` describing the number of chunks/times of the text embeddings.
    get_times = function() {
      return(private$param_chunks)
    },

    #--------------------------------------------------------------------------
    #' @description Number of actual features/dimensions of the text embeddings.In the case a
    #'   [feature extractor][TEFeatureExtractor] was used the number of features is smaller as the original number of
    #'   features. To receive the original number of features (the number of features before applying a
    #'   [feature extractor][TEFeatureExtractor]) you can use the method `get_original_features` of this class.
    #' @return Returns an `int` describing the number of features/dimensions of the text embeddings.
    get_features = function() {
      if (self$is_compressed() == TRUE) {
        return(private$feature_extractor$features)
      } else {
        return(private$param_features)
      }
    },

    #--------------------------------------------------------------------------
    #' @description Number of original features/dimensions of the text embeddings.
    #' @return Returns an `int` describing the number of features/dimensions if no
    #'   [feature extractor][TEFeatureExtractor]) is used or before a [feature extractor][TEFeatureExtractor]) is
    #'   applied.
    get_original_features = function() {
      return(private$param_features)
    },

    #--------------------------------------------------------------------------
    #' @description Checks if the text embedding were reduced by a [feature extractor][TEFeatureExtractor].
    #' @return Returns `TRUE` if the number of dimensions was reduced by a [feature extractor][TEFeatureExtractor]. If
    #'   not return `FALSE`.
    is_compressed = function() {
      if (is.null_or_na(private$feature_extractor$model_name)) {
        return(FALSE)
      } else {
        return(TRUE)
      }
    },

    #--------------------------------------------------------------------------
    #' @description Method setting information on the [feature extractor][TEFeatureExtractor] that was used to reduce
    #'   the number of dimensions of the text embeddings. This information should only be used if a
    #'   [feature extractor][TEFeatureExtractor] was applied.
    #' @param model_name `string` Name of the underlying [TextEmbeddingModel].
    #' @param model_label `string` Label of the underlying [TextEmbeddingModel].
    #' @param features `int` Number of dimension (features) for the **compressed** text embeddings.
    #' @param method `string` Method that the [TEFeatureExtractor] applies for genereating the compressed text
    #'   embeddings.
    #' @param noise_factor `double` Noise factor of the [TEFeatureExtractor].
    #' @param optimizer `string` Optimizer used during training the [TEFeatureExtractor].
    #' @return Method does nothing return. It sets information on a [feature extractor][TEFeatureExtractor].
    add_feature_extractor_info = function(model_name,
                                          model_label = NA,
                                          features = NA,
                                          method = NA,
                                          noise_factor = NA,
                                          optimizer = NA) {
      private$feature_extractor <- list(
        model_name = model_name,
        model_label = model_label,
        features = features,
        method = method,
        noise_factor = noise_factor,
        optimizer = optimizer
      )
    },
    #--------------------------------------------------------------------------
    #' @description Method for receiving information on the [feature extractor][TEFeatureExtractor] that was used to
    #'   reduce the number of dimensions of the text embeddings.
    #' @return Returns a `list` with information on the [feature extractor][TEFeatureExtractor]. If no
    #'   [feature extractor][TEFeatureExtractor] was used it returns `NULL`.
    get_feature_extractor_info = function() {
      if (is.null_or_na(private$feature_extractor$model_name)) {
        return(NULL)
      } else {
        return(private$feature_extractor)
      }
    },

    #--------------------------------------------------------------------------
    #' @description Method for converting this object to an object of class [LargeDataSetForTextEmbeddings].
    #' @return Returns an object of class [LargeDataSetForTextEmbeddings] which uses memory mapping allowing to work
    #'   with large data sets.
    convert_to_LargeDataSetForTextEmbeddings = function() {
      new_data_set <- LargeDataSetForTextEmbeddings$new()
      new_data_set$configure(
        model_name = private$model_name,
        model_label = private$model_label,
        model_date = private$model_date,
        model_method = private$model_method,
        model_version = private$model_version,
        model_language = private$model_language,
        param_seq_length = private$param_seq_length,
        param_chunks = private$param_chunks,
        param_features = private$param_features,
        param_overlap = private$param_overlap,
        param_emb_layer_min = private$param_emb_layer_min,
        param_emb_layer_max = private$param_emb_layer_max,
        param_emb_pool_type = private$param_emb_pool_type,
        param_aggregation = private$param_aggregation
      )

      if (self$is_compressed() == TRUE) {
        new_data_set$add_feature_extractor_info(
          model_name = private$feature_extractor$model_name,
          model_label = private$feature_extractor$model_label,
          features = private$feature_extractor$features,
          method = private$feature_extractor$method,
          noise_factor = private$feature_extractor$noise_factor,
          optimizer = private$feature_extractor$optimizer
        )
      }

      new_data_set$add_embeddings_from_array(self$embeddings)
      return(new_data_set)
    },
    #--------------------------------------------------------------------------
    #' @description Number of rows.
    #' @return Returns the number of rows of the text embeddings which represent the number of cases.
    n_rows = function() {
      return(dim(self$embeddings)[1])
    },
    #--------------------------------------------------------------------------
    #' @description Return all fields.
    #' @return Method returns a `list` containing all public and private fields
    #' of the object.
    get_all_fields = function() {
      public_list <- NULL
      private_list <- NULL

      for (entry in names(self)) {
        if (is.function(self[[entry]]) == FALSE & is.environment(self[[entry]]) == FALSE) {
          public_list[entry] <- list(self[[entry]])
        }
      }

      for (entry in names(private)) {
        if (is.function(private[[entry]]) == FALSE & is.environment(private[[entry]]) == FALSE) {
          private_list[entry] <- list(private[[entry]])
        }
      }

      return(
        list(
          public = public_list,
          private = private_list
        )
      )
    }
  )
)
