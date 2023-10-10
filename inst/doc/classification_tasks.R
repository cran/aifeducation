## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
library(aifeducation)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  reticulate::use_condaenv(condaenv = "aifeducation")
#  library(aifeducation)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  #For tensorflow
#  aifeducation_config$set_global_ml_backend("tensorflow")
#  set_transformers_logger("ERROR")
#  
#  #For PyTorch
#  aifeducation_config$set_global_ml_backend("pytorch")
#  set_transformers_logger("ERROR")

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
#  install.packages("readtext")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  #for excel files
#  textual_data<-readtext::readtext(
#    file="text_data.xlsx",
#    text_field = "texts",
#    docid_field = "id"
#  )

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  #read all files with the extension .txt in the directory data
#  textual_data<-readtext::readtext(
#    file="data/*.txt"
#  )
#  
#  #read all files with the extension .pdf in the directory data
#  textual_data<-readtext::readtext(
#    file="data/*.pdf"
#  )

## ----include = TRUE, eval=TRUE------------------------------------------------
example_data<-data.frame(
  id=quanteda::docvars(quanteda.textmodels::data_corpus_moviereviews)$id2,
  label=quanteda::docvars(quanteda.textmodels::data_corpus_moviereviews)$sentiment)
example_data$text<-as.character(quanteda.textmodels::data_corpus_moviereviews)

table(example_data$label)

## ----include = TRUE, eval=TRUE------------------------------------------------
example_data$label[c(1:500,1001:1500)]=NA
summary(example_data$label)

## ----include = TRUE, eval=TRUE------------------------------------------------
example_data$label[1501:1750]=NA
summary(example_data$label)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  vocab_draft<-bow_pp_create_vocab_draft(
#    path_language_model="language_model/english-gum-ud-2.5-191206.udpipe",
#    data=example_data$text,
#    upos=c("NOUN", "ADJ","VERB"),
#    label_language_model="english-gum-ud-2.5-191206",
#    language="english",
#    trace=TRUE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  basic_text_rep<-bow_pp_create_basic_text_rep(
#    data = example_data$text,
#    vocab_draft = vocab_draft,
#    remove_punct = TRUE,
#    remove_symbols = TRUE,
#    remove_numbers = TRUE,
#    remove_url = TRUE,
#    remove_separators = TRUE,
#    split_hyphens = FALSE,
#    split_tags = FALSE,
#    language_stopwords="eng",
#    use_lemmata = FALSE,
#    to_lower=FALSE,
#    min_termfreq = NULL,
#    min_docfreq= NULL,
#    max_docfreq=NULL,
#    window = 5,
#    weights = 1 / (1:5),
#    trace=TRUE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  basic_text_rep<-bow_pp_create_basic_text_rep(
#  create_bert_model(
#      model_dir = "my_own_transformer",
#      vocab_raw_texts=example_data$text,
#      vocab_size=30522,
#      vocab_do_lower_case=FALSE,
#      max_position_embeddings=512,
#      hidden_size=768,
#      num_hidden_layer=12,
#      num_attention_heads=12,
#      intermediate_size=3072,
#      hidden_act="gelu",
#      hidden_dropout_prob=0.1,
#      sustain_track=TRUE,
#      sustain_iso_code="DEU",
#      sustain_region=NULL,
#      sustain_interval=15,
#      trace=TRUE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  train_tune_bert_model(
#    output_dir = "my_own_transformer_trained",
#    bert_model_dir_path = "my_own_transformer",
#    raw_texts = example_data$text,
#    p_mask=0.15,
#    whole_word=TRUE,
#    val_size=0.1,
#    n_epoch=1,
#    batch_size=12,
#    chunk_size=250,
#    n_workers=1,
#    multi_process=FALSE,
#    sustain_track=TRUE,
#    sustain_iso_code="DEU",
#    sustain_region=NULL,
#    sustain_interval=15,
#    trace=TRUE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  topic_modeling<-TextEmbeddingModel$new(
#    model_name="topic_model_embedding",
#    model_label="Text Embedding via Topic Modeling",
#    model_version="0.0.1",
#    model_language="english",
#    method="lda",
#    bow_basic_text_rep=basic_text_rep,
#    bow_n_dim=12,
#    bow_max_iter=500,
#    bow_cr_criterion=1e-8,
#    trace=TRUE
#  )

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  global_vector_clusters_modeling<-TextEmbeddingModel$new(
#    model_name="global_vector_clusters_embedding",
#    model_label="Text Embedding via Clusters of GlobalVectors",
#    model_version="0.0.1",
#    model_language="english",
#    method="glove_cluster",
#    bow_basic_text_rep=basic_text_rep,
#    bow_n_dim=96,
#    bow_n_cluster=384,
#    bow_max_iter=500,
#    bow_max_iter_cluster=500,
#    bow_cr_criterion=1e-8,
#    trace=TRUE
#  )

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  bert_modeling<-TextEmbeddingModel$new(
#    model_name="bert_embedding",
#    model_label="Text Embedding via BERT",
#    model_version="0.0.1",
#    model_language="english",
#    method = "bert",
#    max_length = 512,
#    chunks=4,
#    overlap=30,
#    aggregation="last",
#    model_dir="my_own_transformer_trained"
#    )

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  topic_embeddings<-topic_modeling$embed(
#    raw_text=example_data$text,
#    doc_id=example_data$id,
#    trace = TRUE)
#  
#  cluster_embeddings<-global_vector_clusters_modeling$embed(
#    raw_text=example_data$text,
#    doc_id=example_data$id,
#    trace = TRUE)
#  
#  bert_embeddings<-bert_modeling$embed(
#    raw_text=example_data$text,
#    doc_id=example_data$id,
#    trace = TRUE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  save_ai_model(
#    model=topic_modeling,
#    model_dir="text_embedding_models",
#    append_ID=FALSE)
#  
#  save_ai_model(
#    model=global_vector_clusters_modeling,
#    model_dir="text_embedding_models",
#    append_ID=FALSE)
#  
#  save_ai_model(
#    model=bert_modeling,
#    model_dir="text_embedding_models",
#    append_ID=FALSE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  topic_modeling<-load_ai_model(
#    model_dir="text_embedding_models/topic_model_embedding")
#  
#  global_vector_clusters_modeling<-load_ai_model(
#    model_dir="text_embedding_models/global_vector_clusters_embedding")
#  
#  bert_modeling<-load_ai_model(
#    model_dir="text_embedding_models/bert_embedding")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_targets<-as.factor(example_data$label)
#  names(example_targets)=example_data$id
#  
#  classifier<-TextEmbeddingClassifierNeuralNet$new(
#    name="movie_review_classifier",
#    label="Classifier for Estimating a Postive or Negative Rating of Movie Reviews",
#    text_embeddings=bert_embeddings,
#    targets=example_targets,
#    hidden=NULL,
#    rec=c(256),
#    self_attention_heads = 2,
#    dropout=0.4,
#    recurrent_dropout=0.4,
#    l2_regularizer=0.01,
#    optimizer="adam",
#    act_fct="gelu",
#    rec_act_fct="tanh")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_targets<-as.factor(example_data$label)
#  names(example_targets)=example_data$id
#  
#  classifier$train(
#     data_embeddings = bert_embeddings,
#     data_targets = example_targets,
#     data_n_test_samples=5,
#     use_baseline=TRUE,
#     bsl_val_size=0.33,
#     use_bsc=TRUE,
#     bsc_methods=c("dbsmote"),
#     bsc_max_k=10,
#     bsc_val_size=0.25,
#     use_bpl=TRUE,
#     bpl_max_steps=5,
#     bpl_epochs_per_step=30,
#     bpl_dynamic_inc=TRUE,
#     bpl_balance=FALSE,
#     bpl_max=1.00,
#     bpl_anchor=1.00,
#     bpl_min=0.00,
#     bpl_weight_inc=0.00,
#     bpl_weight_start=1.00,
#     bpl_model_reset=TRUE,
#     epochs=30,
#     batch_size=8,
#     sustain_track=TRUE,
#     sustain_iso_code="DEU",
#     sustain_region=NULL,
#     sustain_interval=15,
#     trace=TRUE,
#     view_metrics=FALSE,
#     keras_trace=0,
#     n_cores=2,
#     dir_checkpoint="training/classifier")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  classifier$reliability$test_metric_mean

## ----include = TRUE, eval=TRUE------------------------------------------------
test_metric_mean

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  library(iotarelr)
#  iotarelr::plot_iota2_alluvial(test_classifier$reliability$iota_object_end_free)

## ----include = FALSE, eval=TRUE-----------------------------------------------
sustainability_data<-test_classifier_sustainability

## ----include = TRUE, eval=TRUE------------------------------------------------
sustainability_data

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  save_ai_model(
#    model=classifier,
#    model_dir="classifiers",
#    save_format = "keras",
#    append_ID=FALSE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  classifier<-load_ai_model(
#    model_dir="classifiers/movie_review_classifier")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  # If our mode is not loaded
#  bert_modeling<-load_ai_model(
#    model_dir="text_embedding_models/bert_embedding")
#  
#  # Create a numerical representation of the text
#  text_embeddings<-bert_modeling$embed(
#    raw_text = textual_data$texts,
#    doc_id = textual_data$doc_id,
#    batch_size=8,
#    trace=TRUE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  # If your classifier is not loaded
#  classifier<-load_ai_model(
#    model_dir="classifiers/movie_review_classifier")
#  
#  # Predict the classes of new texts
#  predicted_categories<-classifier$predict(
#    newdata = text_embeddings,
#    batch_size=8,
#    verbose=0)

