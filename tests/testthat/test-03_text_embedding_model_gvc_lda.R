testthat::skip_on_os("windows")

testthat::skip_if_not_installed(pkg="quanteda")
testthat::skip_if_not_installed(pkg="topicmodels")
testthat::skip_if_not_installed(pkg="text2vec")
testthat::skip_if_not_installed(pkg="tidytext")

path="test_data/gvc_lda/basic_text_rep_movie_reviews.rda"
testthat::skip_if_not(condition=file.exists(testthat::test_path(path)),
                      message  = "Necessary dataset not available")

if(dir.exists(testthat::test_path("test_artefacts"))==FALSE){
  dir.create(testthat::test_path("test_artefacts"))
}

if(dir.exists(testthat::test_path("test_artefacts/tmp_full_models"))==FALSE){
  dir.create(testthat::test_path("test_artefacts/tmp_full_models"))
}


#------------------------------------------------------------------------------
load(testthat::test_path(path))

example_data<-imdb_movie_reviews
#------------------------------------------------------------------------------

global_vector_clusters_modeling<-TextEmbeddingModel$new(
  model_name="global_vector_clusters_embedding",
  model_label="Text Embedding via Clusters of GlobalVectors",
  model_version="0.0.1",
  model_language="english",
  method="glove_cluster",
  bow_basic_text_rep=basic_text_rep_movie_reviews,
  bow_n_dim=25,
  bow_n_cluster=100,
  bow_max_iter=10,
  bow_max_iter_cluster=400,
  bow_cr_criterion=1e-8,
  trace=FALSE)

test_that("creation_GlobalVectorCluster", {
  expect_s3_class(global_vector_clusters_modeling,
                  class="TextEmbeddingModel")
})

model_name=global_vector_clusters_modeling$get_model_info()$model_name

test_that("GlobalVectorClusters Save Total Model H5", {
  expect_no_error(
    save_ai_model(model=global_vector_clusters_modeling,
                  model_dir = testthat::test_path("test_artefacts/tmp_full_models"),
                  save_format = "H5")
  )
})

test_that("GlobalVectorClusters Load Total Model H5", {
  global_vector_clusters_modeling<-NULL
  global_vector_clusters_modeling<-load_ai_model(
    model_dir = testthat::test_path(paste0("test_artefacts/tmp_full_models/",model_name)),
    ml_framework="auto"
  )
  expect_s3_class(global_vector_clusters_modeling,
                  class="TextEmbeddingModel")
})

test_that("GlobalVectorClusters Save Total Model TF", {
  expect_no_error(
    save_ai_model(model=global_vector_clusters_modeling,
                  model_dir = testthat::test_path("test_artefacts/tmp_full_models"),
                  save_format = "tf")
  )
})

test_that("GlobalVectorClusters Load Total Model TF", {
  global_vector_clusters_modeling<-NULL
  global_vector_clusters_modeling<-load_ai_model(
    model_dir = testthat::test_path(paste0("test_artefacts/tmp_full_models/",model_name)),
    ml_framework="auto"
  )
  expect_s3_class(global_vector_clusters_modeling,
                  class="TextEmbeddingModel")
})


test_that("embedding_GlobalVectorCluster", {
  embeddings<-global_vector_clusters_modeling$embed(raw_text = example_data$text[1:10],
                                        doc_id = 1:10)
  expect_s3_class(embeddings, class="EmbeddedText")

  embeddings<-NULL
  embeddings<-global_vector_clusters_modeling$embed(raw_text = example_data$text[1:1],
                                                    doc_id = 1:1)
  expect_s3_class(embeddings, class="EmbeddedText")

})

test_that("encoding_GlobalVectorCluster", {
  encodings<-global_vector_clusters_modeling$encode(raw_text = example_data$text[1:10])
  expect_length(encodings,10)
  expect_type(encodings,type="list")
})

test_that("decoding_GlobalVectorCluster", {
  encodings<-global_vector_clusters_modeling$encode(raw_text = example_data$text[1:10])
  decodings<-global_vector_clusters_modeling$decode(encodings)
  expect_length(decodings,10)
  expect_type(decodings,type="list")
})


#------------------------------------------------------------------------------
topic_modeling<-TextEmbeddingModel$new(
  model_name="topic_model_embedding",
  model_label="Text Embedding via Topic Modeling",
  model_version="0.0.1",
  model_language="english",
  method="lda",
  bow_basic_text_rep=basic_text_rep_movie_reviews,
  bow_n_dim=2,
  bow_max_iter=10,
  bow_cr_criterion=1e-8,
  trace=FALSE)


test_that("creation_topic_modeling", {
  expect_s3_class(topic_modeling,
                  class="TextEmbeddingModel")
})

model_name=topic_modeling$get_model_info()$model_name

test_that("TopicModeling Save Total Model H5", {
  expect_no_error(
    save_ai_model(model=topic_modeling,
                  model_dir = testthat::test_path("test_artefacts/tmp_full_models"),
                  save_format = "H5")
  )
})

test_that("TopicModeling Load Total Model H5", {
  topic_modeling<-NULL
  topic_modeling<-load_ai_model(
    model_dir = testthat::test_path(paste0("test_artefacts/tmp_full_models/",model_name)),
    ml_framework="auto"
  )
  expect_s3_class(topic_modeling,
                  class="TextEmbeddingModel")
})

test_that("TopicModeling Save Total Model TF", {
  expect_no_error(
    save_ai_model(model=topic_modeling,
                  model_dir = testthat::test_path("test_artefacts/tmp_full_models"),
                  save_format = "tf")
  )
})

test_that("TopicModeling Load Total Model TF", {
  topic_modeling<-NULL
  topic_modeling<-load_ai_model(
    model_dir = testthat::test_path(paste0("test_artefacts/tmp_full_models/",model_name)),
    ml_framework="auto"
  )
  expect_s3_class(topic_modeling,
                  class="TextEmbeddingModel")
})


test_that("embedding_topic_modeling", {
  embeddings<-topic_modeling$embed(raw_text = example_data$text[1:10],
                                   doc_id = 1:10)
  expect_s3_class(embeddings, class="EmbeddedText")

  embeddings<-NULL
  embeddings<-topic_modeling$embed(raw_text = example_data$text[1:1],
                                   doc_id = 1:1)
  expect_s3_class(embeddings, class="EmbeddedText")

})

test_that("encoding_topic_modeling", {
  encodings<-topic_modeling$encode(raw_text = example_data$text[1:10])

  expect_length(encodings,10)
  expect_type(encodings,type="list")
})

test_that("decoding_topic_modeling", {
  encodings<-topic_modeling$encode(raw_text = example_data$text[1:10])
  decodings<-topic_modeling$decode(encodings)

  expect_length(decodings,10)
  expect_type(decodings,type="list")
})



#-------------------------------------------------------------------------------
#test check_embedding for embeddings and models
embeddings_topic<-topic_modeling$embed(raw_text = example_data$text[1:10],
                                 doc_id = 1:10)
embeddings_gvc<-global_vector_clusters_modeling$embed(raw_text = example_data$text[1:10],
                                                  doc_id = 1:10)

test_that("check_embeddings_models", {
  expect_false(
    check_embedding_models(
      object_list = list(topic_modeling,global_vector_clusters_modeling),
      same_class = FALSE))

  expect_false(
    check_embedding_models(
      object_list = list(embeddings_topic,global_vector_clusters_modeling),
      same_class = FALSE))

  expect_false(
    check_embedding_models(
      object_list = list(embeddings_gvc,global_vector_clusters_modeling),
      same_class = TRUE))

  expect_false(
    check_embedding_models(
      object_list = list(embeddings_topic,topic_modeling),
      same_class = TRUE))

  expect_false(
    check_embedding_models(
      object_list = list(embeddings_topic,embeddings_gvc),
      same_class = TRUE))

  expect_false(
    check_embedding_models(
      object_list = list(embeddings_topic,embeddings_gvc),
      same_class = FALSE))

  expect_true(
    check_embedding_models(
      object_list = list(embeddings_topic,topic_modeling),
      same_class = FALSE))

  expect_true(
    check_embedding_models(
      object_list = list(embeddings_gvc,global_vector_clusters_modeling),
      same_class = FALSE))

  expect_true(
    check_embedding_models(
      object_list = list(embeddings_gvc,embeddings_gvc),
      same_class = FALSE))

  expect_true(
    check_embedding_models(
      object_list = list(embeddings_gvc,embeddings_gvc),
      same_class = TRUE))

  expect_true(
    check_embedding_models(
      object_list = list(embeddings_topic,embeddings_topic),
      same_class = FALSE))

  expect_true(
    check_embedding_models(
      object_list = list(embeddings_topic,embeddings_topic),
      same_class = TRUE))
})
#------------------------------------------------------------------------------
#check_combine_embedded_texts
test_that("check_combine_embedded_texts",{
  expect_error(
    combine_embeddings(
      embeddings_list = list(embeddings_topic,embeddings_gvc)))
  expect_error(
    combine_embeddings(
      embeddings_list = list(embeddings_topic,embeddings_topic)))
  expect_error(
    combine_embeddings(
      embeddings_list = list(embeddings_gvc,embeddings_gvc)))

  tmp_embedding1<-embeddings_gvc$clone(deep = TRUE)
  tmp_embedding1$embeddings<-tmp_embedding1$embeddings[1:5,,,drop=FALSE]
  tmp_embedding2<-embeddings_gvc$clone(deep = TRUE)
  tmp_embedding2$embeddings<-tmp_embedding2$embeddings[6:10,,,drop=FALSE]

  expect_s3_class(combine_embeddings(embeddings_list = list(tmp_embedding1,tmp_embedding2)),
                  class="EmbeddedText")
  expect_equal(nrow(combine_embeddings(embeddings_list = list(tmp_embedding1,tmp_embedding2))$embeddings),10)

  tmp_embedding1<-embeddings_topic$clone(deep = TRUE)
  tmp_embedding1$embeddings<-tmp_embedding1$embeddings[1:5,,,drop=FALSE]
  tmp_embedding2<-embeddings_topic$clone(deep = TRUE)
  tmp_embedding2$embeddings<-tmp_embedding2$embeddings[6:10,,,drop=FALSE]

  expect_s3_class(combine_embeddings(embeddings_list = list(tmp_embedding1,tmp_embedding2)),
                  class="EmbeddedText")
  expect_equal(nrow(combine_embeddings(embeddings_list = list(tmp_embedding1,tmp_embedding2))$embeddings),10)

})


#------------------------------------------------------------------------------
#check_combine_embedded_texts_with_only_1_case
test_that("check_single_case",{
  tmp_embedding1<-embeddings_gvc$clone(deep = TRUE)
  tmp_embedding1$embeddings<-tmp_embedding1$embeddings[1:1,,,drop=FALSE]
  tmp_embedding2<-embeddings_gvc$clone(deep = TRUE)
  tmp_embedding2$embeddings<-tmp_embedding2$embeddings[2:2,,,drop=FALSE]

  expect_s3_class(combine_embeddings(embeddings_list = list(tmp_embedding1,tmp_embedding2)),
                  class="EmbeddedText")
  expect_equal(nrow(combine_embeddings(embeddings_list = list(tmp_embedding1,tmp_embedding2))$embeddings),2)

  tmp_embedding1<-embeddings_topic$clone(deep = TRUE)
  tmp_embedding1$embeddings<-tmp_embedding1$embeddings[1:1,,,drop=FALSE]
  tmp_embedding2<-embeddings_topic$clone(deep = TRUE)
  tmp_embedding2$embeddings<-tmp_embedding2$embeddings[2:2,,,drop=FALSE]

  expect_s3_class(combine_embeddings(embeddings_list = list(tmp_embedding1,tmp_embedding2)),
                  class="EmbeddedText")
  expect_equal(nrow(combine_embeddings(embeddings_list = list(tmp_embedding1,tmp_embedding2))$embeddings),2)

})

