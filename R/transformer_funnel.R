#'Function for creating a new transformer based on Funnel Transformer
#'
#'This function creates a transformer configuration based on the Funnel Transformer
#'base architecture
#'and a vocabulary based on WordPiece by using
#'the python libraries 'transformers' and 'tokenizers'.
#'
#'@param ml_framework \code{string} Framework to use for training and inference.
#'\code{ml_framework="tensorflow"} for 'tensorflow' and \code{ml_framework="pytorch"}
#'for 'pytorch'.
#'@param model_dir \code{string} Path to the directory where the model should be saved.
#'@param vocab_raw_texts \code{vector} containing the raw texts for creating the
#'vocabulary.
#'@param vocab_size \code{int} Size of the vocabulary.
#'@param vocab_do_lower_case \code{bool} \code{TRUE} if all words/tokens should be lower case.
#'@param max_position_embeddings \code{int} Number of maximal position embeddings. This parameter
#'also determines the maximum length of a sequence which can be processed with the model.
#'@param block_sizes \code{vector} of \code{int} determining the number and sizes
#'of each block.
#'@param hidden_size \code{int} Initial number of neurons in each layer.
#'@param target_hidden_size \code{int} Number of neurons in the final layer.
#'This parameter determines the dimensionality of the resulting text embedding.
#'@param num_attention_heads \code{int} Number of attention heads.
#'@param intermediate_size \code{int} Number of neurons in the intermediate layer of
#'the attention mechanism.
#'@param num_decoder_layers \code{int} Number of decoding layers.
#'@param hidden_act \code{string} name of the activation function.
#'@param hidden_dropout_prob \code{double} Ratio of dropout.
#'@param attention_probs_dropout_prob \code{double} Ratio of dropout for attention
#'probabilities.
#'@param activation_dropout \code{float} Dropout probability between the layers of
#'the feed-forward blocks.
#'@param pooling_type \code{string} \code{"mean"} for pooling with mean and \code{"max"}
#'for pooling with maximum values.
#'
#'@param sustain_track \code{bool} If \code{TRUE} energy consumption is tracked
#'during training via the python library codecarbon.
#'@param sustain_iso_code \code{string} ISO code (Alpha-3-Code) for the country. This variable
#'must be set if sustainability should be tracked. A list can be found on
#'Wikipedia: \url{https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes}.
#'@param sustain_region Region within a country. Only available for USA and
#'Canada See the documentation of codecarbon for more information.
#'\url{https://mlco2.github.io/codecarbon/parameters.html}
#'@param sustain_interval \code{integer} Interval in seconds for measuring power
#'usage.
#'
#'@param trace \code{bool} \code{TRUE} if information about the progress should be
#'printed to the console.
#'@return This function does not return an object. Instead the configuration
#'and the vocabulary of the new model are saved on disk.
#'
#'@note The model uses a configuration with \code{truncate_seq=TRUE} to avoid
#'implementation problems with tensorflow.
#'@note To train the model, pass the directory of the model to the function
#'\link{train_tune_funnel_model}.
#'@note Model is created with \code{separete_cls=TRUE},\code{truncate_seq=TRUE}, and
#' \code{pool_q_only=TRUE}.
#'@note This models uses a WordPiece Tokenizer like BERT and can be trained with
#'whole word masking. Transformer library may show a warning which can be ignored.
#'
#'@references
#'Dai, Z., Lai, G., Yang, Y. & Le, Q. V. (2020). Funnel-Transformer: Filtering
#'out Sequential Redundancy for Efficient Language Processing.
#'\doi{10.48550/arXiv.2006.03236}
#'
#'@references Hugging Face documentation
#'\url{https://huggingface.co/docs/transformers/model_doc/funnel#funnel-transformer}
#'
#'@family Transformer
#'@export
create_funnel_model<-function(
    ml_framework=aifeducation_config$get_framework()$TextEmbeddingFramework,
    model_dir,
    vocab_raw_texts=NULL,
    vocab_size=30522,
    vocab_do_lower_case=FALSE,
    max_position_embeddings=512,
    hidden_size=768,
    target_hidden_size=64,
    block_sizes=c(4,4,4),
    num_attention_heads=12,
    intermediate_size=3072,
    num_decoder_layers=2,
    pooling_type="mean",
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    activation_dropout=0.0,
    sustain_track=TRUE,
    sustain_iso_code=NULL,
    sustain_region=NULL,
    sustain_interval=15,
    trace=TRUE){

  #argument checking-----------------------------------------------------------
  #if(max_position_embeddings>512){
  #  warning("Due to a quadratic increase in memory requirments it is not
  #          recommended to set max_position_embeddings above 512.
  #          If you want to analyse long documents please split your document
  #          into several chunks with an object of class TextEmbedding Model or
  #          use another transformer (e.g. longformer).")
  #}

  if((hidden_act %in% c("gelu", "relu", "silu","gelu_new"))==FALSE){
    stop("hidden_act must be gelu, relu, silu or gelu_new")
  }

  if((ml_framework %in%c("pytorch","tensorflow","not_specified"))==FALSE){
    stop("ml_framework must be 'tensorflow' or 'pytorch'.")
  }

  if(ml_framework=="not_specified"){
    stop("The global machine learning framework is not set. Please use
             aifeducation_config$set_global_ml_backend() directly after loading
             the library to set the global framework. ")
  }

  #Start Sustainability Tracking
  if(sustain_track==TRUE){
    if(is.null(sustain_iso_code)==TRUE){
      stop("Sustainability tracking is activated but iso code for the
               country is missing. Add iso code or deactivate tracking.")
    }
    sustainability_tracker<-codecarbon$OfflineEmissionsTracker(
      country_iso_code=sustain_iso_code,
      region=sustain_region,
      tracking_mode="machine",
      log_level="warning",
      measure_power_secs=sustain_interval,
      save_to_file=FALSE,
      save_to_api=FALSE
    )
    sustainability_tracker$start()
  }

  #Creating a new Tokenizer for Computing Vocabulary
  special_tokens=c("<cls>","<sep>","<pad>","<unk>","<mask>")
  tok_new<-tok$Tokenizer(tok$models$WordPiece())
  tok_new$normalizer=tok$normalizers$BertNormalizer(
    lowercase=vocab_do_lower_case,
    clean_text = TRUE,
    handle_chinese_chars = TRUE,
    strip_accents = vocab_do_lower_case)
  tok_new$pre_tokenizer=tok$pre_tokenizers$BertPreTokenizer()
  tok_new$post_processor<-tok$processors$BertProcessing(
    sep=reticulate::tuple(list("<sep>",as.integer(1))),
    cls=reticulate::tuple(list("<cls>",as.integer(0)))
  )

  tok_new$decode=tok$decoders$WordPiece()

  trainer<-tok$trainers$WordPieceTrainer(
    vocab_size=as.integer(vocab_size),
    special_tokens = special_tokens,
    show_progress=trace)

  #Calculating Vocabulary
  if(trace==TRUE){
    cat(paste(date(),
              "Start Computing Vocabulary","\n"))
  }
  tok_new$train_from_iterator(vocab_raw_texts,trainer=trainer)
  if(trace==TRUE){
    cat(paste(date(),
              "Start Computing Vocabulary - Done","\n"))
  }

  if(dir.exists(model_dir)==FALSE){
    cat(paste(date(),"Creating Directory","\n"))
    dir.create(model_dir)
  }

  write(c(special_tokens,names(tok_new$get_vocab())),
        file=paste0(model_dir,"/","vocab.txt"))

  if(trace==TRUE){
    cat(paste(date(),
              "Creating Tokenizer","\n"))
  }

  tokenizer=transformers$PreTrainedTokenizerFast(
    tokenizer_object=tok_new,
    unk_token="<unk>",
    sep_token="<sep>",
    pad_token="<pad>",
    cls_token="<cls>",
    mask_token="<mask>",
    bos_token = "<cls>",
    eos_token = "<sep>")

  if(trace==TRUE){
    cat(paste(date(),
              "Creating Tokenizer - Done","\n"))
  }

  configuration=transformers$FunnelConfig(
    vocab_size=as.integer(length(tokenizer$get_vocab())),
    block_sizes =as.integer(block_sizes),
    block_repeats=NULL,
    num_decoder_layers=as.integer(num_decoder_layers),
    d_model=as.integer(hidden_size),
    n_head=as.integer(num_attention_heads),
    d_head=as.integer(target_hidden_size),
    d_inner=as.integer(intermediate_size),
    hidden_act=hidden_act,
    hidden_dropout_prob=hidden_dropout_prob,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    activation_dropout=as.integer(activation_dropout),
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pooling_type =pooling_type,
    attention_type="relative_shift",
    separate_cls=TRUE,
    truncate_seq=TRUE,
    pool_q_only=TRUE,
    max_position_embeddings=as.integer(max_position_embeddings),
  )

  if(ml_framework=="tensorflow"){
    model=transformers$TFFunnelModel(configuration)
  } else {
    model=transformers$FunnelModel(configuration)
  }

  if(trace==TRUE){
    cat(paste(date(),
              "Saving Funnel Transformer Model","\n"))
  }
  model$save_pretrained(save_directory=model_dir)

  if(trace==TRUE){
    cat(paste(date(),
              "Saving Tokenizer Model","\n"))
  }
  tokenizer$save_pretrained(model_dir)

  #Stop Sustainability Tracking if requested
  if(sustain_track==TRUE){
    sustainability_tracker$stop()
    sustainability_data<-summarize_tracked_sustainability(sustainability_tracker)
    sustain_matrix=t(as.matrix(unlist(sustainability_data)))

    if(trace==TRUE){
      cat(paste(date(),
                "Saving Sustainability Data","\n"))
    }

    write.csv(
      x=sustain_matrix,
      file=paste0(model_dir,"/sustainability.csv"),
      row.names = FALSE)
  }

  if(trace==TRUE){
    cat(paste(date(),
              "Done","\n"))
  }
}



#'Function for training and fine-tuning a Funnel Transformer model
#'
#'This function can be used to train or fine-tune a transformer
#'based on Funnel Transformer architecture with the help of the python libraries 'transformers',
#''datasets', and 'tokenizers'.
#'
#'@param ml_framework \code{string} Framework to use for training and inference.
#'\code{ml_framework="tensorflow"} for 'tensorflow' and \code{ml_framework="pytorch"}
#'for 'pytorch'.
#'@param output_dir \code{string} Path to the directory where the final model
#'should be saved. If the directory does not exist, it will be created.
#'@param model_dir_path \code{string} Path to the directory where the original
#'model is stored.
#'@param raw_texts \code{vector} containing the raw texts for training.
#'@param p_mask \code{double} Ratio determining the number of words/tokens for masking.
#'@param val_size \code{double} Ratio determining the amount of token chunks used for
#'validation.
#'@param n_epoch \code{int} Number of epochs for training.
#'@param batch_size \code{int} Size of batches.
#'@param chunk_size \code{int} Size of every chunk for training.
#'@param min_seq_len \code{int} Only relevant if \code{full_sequences_only=FALSE}.
#'Value determines the minimal sequence length for inclusion in training process.
#'@param full_sequences_only \code{bool} \code{TRUE} if only token sequences with
#'a length equal to \code{chunk_size} should be used for training.
#'@param learning_rate \code{double} Learning rate for adam optimizer.
#'@param n_workers \code{int} Number of workers.
#'@param multi_process \code{bool} \code{TRUE} if multiple processes should be activated.
#'
#'@param sustain_track \code{bool} If \code{TRUE} energy consumption is tracked
#'during training via the python library codecarbon.
#'@param sustain_iso_code \code{string} ISO code (Alpha-3-Code) for the country. This variable
#'must be set if sustainability should be tracked. A list can be found on
#'Wikipedia: \url{https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes}.
#'@param sustain_region Region within a country. Only available for USA and
#'Canada See the documentation of codecarbon for more information.
#'\url{https://mlco2.github.io/codecarbon/parameters.html}
#'@param sustain_interval \code{integer} Interval in seconds for measuring power
#'usage.
#'
#'@param trace \code{bool} \code{TRUE} if information on the progress should be printed
#'to the console.
#'@param keras_trace \code{int} \code{keras_trace=0} does not print any
#'information about the training process from keras on the console.
#'\code{keras_trace=1} prints a progress bar. \code{keras_trace=2} prints
#'one line of information for every epoch.
#'@return This function does not return an object. Instead the trained or fine-tuned
#'model is saved to disk.
#'@note if \code{aug_vocab_by > 0} the raw text is used for training a WordPiece
#'tokenizer. At the end of this process, additional entries are added to the vocabulary
#'that are not part of the original vocabulary. This is in an experimental state.
#'@note Pre-Trained models which can be fine-tuned with this function are available
#'at \url{https://huggingface.co/}.
#'@note New models can be created via the function \link{create_funnel_model}.
#'@note Training of the model makes use of dynamic masking.
#'@references
#'Dai, Z., Lai, G., Yang, Y. & Le, Q. V. (2020). Funnel-Transformer: Filtering
#'out Sequential Redundancy for Efficient Language Processing.
#'\doi{10.48550/arXiv.2006.03236}
#'
#'@references Hugging Face documentation
#'\url{https://huggingface.co/docs/transformers/model_doc/funnel#funnel-transformer}
#'
#'@family Transformer
#'
#'@importFrom utils write.csv
#'@importFrom utils read.csv
#'
#'@export
train_tune_funnel_model=function(ml_framework=aifeducation_config$get_framework()$TextEmbeddingFramework,
                                 output_dir,
                                 model_dir_path,
                                 raw_texts,
                                 p_mask=0.15,
                                 val_size=0.1,
                                 n_epoch=1,
                                 batch_size=12,
                                 chunk_size=250,
                                 min_seq_len=50,
                                 full_sequences_only=FALSE,
                                 learning_rate=3e-3,
                                 n_workers=1,
                                 multi_process=FALSE,
                                 sustain_track=TRUE,
                                 sustain_iso_code=NULL,
                                 sustain_region=NULL,
                                 sustain_interval=15,
                                 trace=TRUE,
                                 keras_trace=1){

  if((ml_framework %in%c("pytorch","tensorflow","not_specified"))==FALSE){
    stop("ml_framework must be 'tensorflow' or 'pytorch'.")
  }

  if(ml_framework=="not_specified"){
    stop("The global machine learning framework is not set. Please use
             aifeducation_config$set_global_ml_backend() directly after loading
             the library to set the global framework. ")
  }

  #Start Sustainability Tracking
  if(sustain_track==TRUE){
    if(is.null(sustain_iso_code)==TRUE){
      stop("Sustainability tracking is activated but iso code for the
               country is missing. Add iso code or deactivate tracking.")
    }
    sustainability_tracker<-codecarbon$OfflineEmissionsTracker(
      country_iso_code=sustain_iso_code,
      region=sustain_region,
      tracking_mode="machine",
      log_level="warning",
      measure_power_secs=sustain_interval,
      save_to_file=FALSE,
      save_to_api=FALSE
    )
    sustainability_tracker$start()
  }

  if(ml_framework=="tensorflow"){
    if(file.exists(paste0(model_dir_path,"/tf_model.h5"))){
      from_pt=FALSE
    } else if (file.exists(paste0(model_dir_path,"/pytorch_model.bin"))){
      from_pt=TRUE
    } else {
      stop("Directory does not contain a tf_model.h5 or pytorch_model.bin file.")
    }
  } else {
    if(file.exists(paste0(model_dir_path,"/pytorch_model.bin"))){
      from_tf=FALSE
    } else if (file.exists(paste0(model_dir_path,"/tf_model.h5"))){
      from_tf=TRUE
    } else {
      stop("Directory does not contain a tf_model.h5 or pytorch_model.bin file.")
    }
  }

  if(ml_framework=="tensorflow"){
    mlm_model=transformers$TFFunnelForMaskedLM$from_pretrained(model_dir_path, from_pt=from_pt)
  } else {
    mlm_model=transformers$FunnelForMaskedLM$from_pretrained(model_dir_path,from_tf=from_tf)
  }

  tokenizer<-transformers$AutoTokenizer$from_pretrained(model_dir_path)

  if(trace==TRUE){
    cat(paste(date(),"Tokenize Raw Texts","\n"))
  }

  if(trace==TRUE){
    cat(paste(date(),"Creating Sequence Chunks For Training","\n"))
  }

  if(full_sequences_only==FALSE){
    tokenized_texts=tokenizer(raw_texts,
                              truncation =TRUE,
                              padding= FALSE,
                              max_length=as.integer(chunk_size),
                              return_overflowing_tokens = TRUE,
                              return_length = TRUE,
                              return_special_tokens_mask=TRUE,
                              return_offsets_mapping = FALSE,
                              return_attention_mask = TRUE,
                              return_tensors="np")
    tokenized_dataset=datasets$Dataset$from_dict(tokenized_texts)
    relevant_indices=which(tokenized_dataset["length"]<=chunk_size & tokenized_dataset["length"]>=min_seq_len)


    tokenized_texts=tokenizer(raw_texts,
                              truncation =TRUE,
                              padding= TRUE,
                              max_length=as.integer(chunk_size),
                              return_overflowing_tokens = TRUE,
                              return_length = TRUE,
                              return_special_tokens_mask=TRUE,
                              return_offsets_mapping = FALSE,
                              return_attention_mask = TRUE,
                              return_tensors="np")
    tokenized_dataset=datasets$Dataset$from_dict(tokenized_texts)
    tokenized_dataset=tokenized_dataset$select(as.integer(relevant_indices-1))

  } else {
    tokenized_texts=tokenizer(raw_texts,
                              truncation =TRUE,
                              padding= FALSE,
                              max_length=as.integer(chunk_size),
                              return_overflowing_tokens = TRUE,
                              return_length = TRUE,
                              return_special_tokens_mask=TRUE,
                              return_offsets_mapping = FALSE,
                              return_attention_mask = TRUE,
                              return_tensors="np")
    tokenized_dataset=datasets$Dataset$from_dict(tokenized_texts)
    relevant_indices=which(tokenized_dataset["length"]==chunk_size)
    tokenized_dataset=tokenized_dataset$select(as.integer(relevant_indices-1))
  }

  n_chunks=tokenized_dataset$num_rows

  if(trace==TRUE){
    cat(paste(date(),n_chunks,"Chunks Created","\n"))
  }

  if(dir.exists(paste0(output_dir))==FALSE){
    if(trace==TRUE){
      cat(paste(date(),"Creating Output Directory","\n"))
    }
    dir.create(paste0(output_dir))
  }

  if(dir.exists(paste0(output_dir,"/checkpoints"))==FALSE){
    if(trace==TRUE){
      cat(paste(date(),"Creating Checkpoint Directory","\n"))
    }
    dir.create(paste0(output_dir,"/checkpoints"))
  }

  if(ml_framework=="tensorflow"){


    if(trace==TRUE){
      cat(paste(date(),"Using Token Masking","\n"))
    }
    data_collator=transformers$DataCollatorForLanguageModeling(
      tokenizer = tokenizer,
      mlm = TRUE,
      mlm_probability = p_mask,
      return_tensors = "tf"
    )

    tokenized_dataset=tokenized_dataset$add_column(name="labels",column=tokenized_dataset["input_ids"])
    tokenized_dataset$set_format(type="tensorflow")

    tokenized_dataset=tokenized_dataset$train_test_split(test_size=val_size)

    tf_train_dataset=mlm_model$prepare_tf_dataset(
      dataset = tokenized_dataset$train,
      batch_size = as.integer(batch_size),
      collate_fn = data_collator,
      shuffle = TRUE
    )
    tf_test_dataset=mlm_model$prepare_tf_dataset(
      dataset = tokenized_dataset$test,
      batch_size = as.integer(batch_size),
      collate_fn = data_collator,
      shuffle = TRUE
    )

    if(trace==TRUE){
      cat(paste(date(),"Preparing Training of the Model","\n"))
    }
    adam<-tf$keras$optimizers$Adam


    callback_checkpoint=tf$keras$callbacks$ModelCheckpoint(
      filepath = paste0(output_dir,"/checkpoints/best_weights.h5"),
      monitor="val_loss",
      verbose = as.integer(min(keras_trace,1)),
      mode="auto",
      save_best_only=TRUE,
      save_freq="epoch",
      save_weights_only= TRUE
    )

    if(trace==TRUE){
      cat(paste(date(),"Compile Model","\n"))
    }
    mlm_model$compile(optimizer=adam(learning_rate),
                      loss="auto")

    #Clear session to provide enough resources for computations
    tf$keras$backend$clear_session()

    if(trace==TRUE){
      cat(paste(date(),"Start Fine Tuning","\n"))
    }
    mlm_model$fit(x=tf_train_dataset,
                  validation_data=tf_test_dataset,
                  epochs=as.integer(n_epoch),
                  workers=as.integer(n_workers),
                  use_multiprocessing=multi_process,
                  callbacks=list(callback_checkpoint),
                  verbose=as.integer(keras_trace))

    if(trace==TRUE){
      cat(paste(date(),"Load Weights From Best Checkpoint","\n"))
    }
    mlm_model$load_weights(paste0(output_dir,"/checkpoints/best_weights.h5"))
  } else {


    if(trace==TRUE){
      cat(paste(date(),"Using Token Masking","\n"))
    }
    data_collator=transformers$DataCollatorForLanguageModeling(
      tokenizer = tokenizer,
      mlm = TRUE,
      mlm_probability = p_mask,
      return_tensors = "pt"
    )


    tokenized_dataset=tokenized_dataset$add_column(name="labels",column=tokenized_dataset["input_ids"])
    tokenized_dataset$set_format(type="torch")

    tokenized_dataset=tokenized_dataset$train_test_split(test_size=val_size)

    training_args=transformers$TrainingArguments(
      output_dir = paste0(output_dir,"/checkpoints"),
      overwrite_output_dir=TRUE,
      evaluation_strategy = "epoch",
      num_train_epochs = as.integer(n_epoch),
      logging_strategy="epoch",
      save_strategy ="epoch",
      save_total_limit=as.integer(1),
      load_best_model_at_end=TRUE,
      optim = "adamw_torch",
      learning_rate = learning_rate,
      per_device_train_batch_size = as.integer(batch_size),
      per_device_eval_batch_size = as.integer(batch_size),
      save_safetensors=TRUE,
      auto_find_batch_size=FALSE,
      report_to="none"
    )

    trainer=transformers$Trainer(
      model=mlm_model,
      train_dataset = tokenized_dataset$train,
      eval_dataset = tokenized_dataset$test,
      args = training_args,
      data_collator = data_collator,
      tokenizer = tokenizer
    )
    trainer$remove_callback(transformers$integrations$CodeCarbonCallback)

    trainer$train()
  }

  if(trace==TRUE){
    cat(paste(date(),"Saving Funnel Model","\n"))
  }
  mlm_model$save_pretrained(save_directory=output_dir)

  if(trace==TRUE){
    cat(paste(date(),"Saving Tokenizer","\n"))
  }
  tokenizer$save_pretrained(output_dir)

  #Stop Sustainability Tracking if requested
  if(sustain_track==TRUE){
    sustainability_tracker$stop()
    sustainability_data<-summarize_tracked_sustainability(sustainability_tracker)
    sustain_matrix=t(as.matrix(unlist(sustainability_data)))

    if(trace==TRUE){
      cat(paste(date(),
                "Saving Sustainability Data","\n"))
    }

    sustainability_data_file_path_input=paste0(model_dir_path,"/sustainability.csv")
    sustainability_data_file_path=paste0(output_dir,"/sustainability.csv")

    if(file.exists(sustainability_data_file_path_input)){
      sustainability_data_chronic<-as.matrix(read.csv(sustainability_data_file_path_input))
      sustainability_data_chronic<-rbind(
        sustainability_data_chronic,
        sustain_matrix
      )

      write.csv(
        x=sustainability_data_chronic,
        file=sustainability_data_file_path,
        row.names = FALSE)

    } else {
      write.csv(
        x=sustain_matrix,
        file=sustainability_data_file_path,
        row.names = FALSE)
    }
  }

  if(trace==TRUE){
    cat(paste(date(),"Done","\n"))
  }
}


