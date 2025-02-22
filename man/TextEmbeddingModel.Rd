% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/TextEmbeddingModel.R
\name{TextEmbeddingModel}
\alias{TextEmbeddingModel}
\title{Text embedding model}
\value{
Objects of class \link{TextEmbeddingModel} transform raw texts into numerical representations which can be used
for downstream tasks. For this aim objects of this class allow to tokenize raw texts, to encode tokens to sequences
of integers, and to decode sequences of integers back to tokens.
}
\description{
This \code{R6} class stores a text embedding model which can be used to tokenize, encode, decode, and embed
raw texts. The object provides a unique interface for different text processing methods.
}
\seealso{
Other Text Embedding: 
\code{\link{TEFeatureExtractor}}
}
\concept{Text Embedding}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{last_training}}{('list()')\cr
List for storing the history and the results of the last training. This
information will be overwritten if a new training is started.}

\item{\code{tokenizer_statistics}}{('matrix()')\cr
Matrix containing the tokenizer statistics for the creation of the tokenizer
and all training runs according to Kaya & Tantuğ (2024).

Kaya, Y. B., & Tantuğ, A. C. (2024). Effect of tokenization granularity for Turkish
large language models. Intelligent Systems with Applications, 21, 200335.
https://doi.org/10.1016/j.iswa.2024.200335}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-TextEmbeddingModel-configure}{\code{TextEmbeddingModel$configure()}}
\item \href{#method-TextEmbeddingModel-load_from_disk}{\code{TextEmbeddingModel$load_from_disk()}}
\item \href{#method-TextEmbeddingModel-load}{\code{TextEmbeddingModel$load()}}
\item \href{#method-TextEmbeddingModel-save}{\code{TextEmbeddingModel$save()}}
\item \href{#method-TextEmbeddingModel-encode}{\code{TextEmbeddingModel$encode()}}
\item \href{#method-TextEmbeddingModel-decode}{\code{TextEmbeddingModel$decode()}}
\item \href{#method-TextEmbeddingModel-get_special_tokens}{\code{TextEmbeddingModel$get_special_tokens()}}
\item \href{#method-TextEmbeddingModel-embed}{\code{TextEmbeddingModel$embed()}}
\item \href{#method-TextEmbeddingModel-embed_large}{\code{TextEmbeddingModel$embed_large()}}
\item \href{#method-TextEmbeddingModel-fill_mask}{\code{TextEmbeddingModel$fill_mask()}}
\item \href{#method-TextEmbeddingModel-set_publication_info}{\code{TextEmbeddingModel$set_publication_info()}}
\item \href{#method-TextEmbeddingModel-get_publication_info}{\code{TextEmbeddingModel$get_publication_info()}}
\item \href{#method-TextEmbeddingModel-set_model_license}{\code{TextEmbeddingModel$set_model_license()}}
\item \href{#method-TextEmbeddingModel-get_model_license}{\code{TextEmbeddingModel$get_model_license()}}
\item \href{#method-TextEmbeddingModel-set_documentation_license}{\code{TextEmbeddingModel$set_documentation_license()}}
\item \href{#method-TextEmbeddingModel-get_documentation_license}{\code{TextEmbeddingModel$get_documentation_license()}}
\item \href{#method-TextEmbeddingModel-set_model_description}{\code{TextEmbeddingModel$set_model_description()}}
\item \href{#method-TextEmbeddingModel-get_model_description}{\code{TextEmbeddingModel$get_model_description()}}
\item \href{#method-TextEmbeddingModel-get_model_info}{\code{TextEmbeddingModel$get_model_info()}}
\item \href{#method-TextEmbeddingModel-get_package_versions}{\code{TextEmbeddingModel$get_package_versions()}}
\item \href{#method-TextEmbeddingModel-get_basic_components}{\code{TextEmbeddingModel$get_basic_components()}}
\item \href{#method-TextEmbeddingModel-get_transformer_components}{\code{TextEmbeddingModel$get_transformer_components()}}
\item \href{#method-TextEmbeddingModel-get_sustainability_data}{\code{TextEmbeddingModel$get_sustainability_data()}}
\item \href{#method-TextEmbeddingModel-get_ml_framework}{\code{TextEmbeddingModel$get_ml_framework()}}
\item \href{#method-TextEmbeddingModel-count_parameter}{\code{TextEmbeddingModel$count_parameter()}}
\item \href{#method-TextEmbeddingModel-is_configured}{\code{TextEmbeddingModel$is_configured()}}
\item \href{#method-TextEmbeddingModel-get_private}{\code{TextEmbeddingModel$get_private()}}
\item \href{#method-TextEmbeddingModel-get_all_fields}{\code{TextEmbeddingModel$get_all_fields()}}
\item \href{#method-TextEmbeddingModel-clone}{\code{TextEmbeddingModel$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-configure"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-configure}{}}}
\subsection{Method \code{configure()}}{
Method for creating a new text embedding model
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$configure(
  model_name = NULL,
  model_label = NULL,
  model_language = NULL,
  method = NULL,
  ml_framework = "pytorch",
  max_length = 0,
  chunks = 2,
  overlap = 0,
  emb_layer_min = "middle",
  emb_layer_max = "2_3_layer",
  emb_pool_type = "average",
  model_dir = NULL,
  trace = FALSE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{model_name}}{\code{string} containing the name of the new model.}

\item{\code{model_label}}{\code{string} containing the label/title of the new model.}

\item{\code{model_language}}{\code{string} containing the language which the model
represents (e.g., English).}

\item{\code{method}}{\code{string} determining the kind of embedding model. Currently
the following models are supported:
\code{method="bert"} for Bidirectional Encoder Representations from Transformers (BERT),
\code{method="roberta"} for A Robustly Optimized BERT Pretraining Approach (RoBERTa),
\code{method="longformer"} for Long-Document Transformer,
\code{method="funnel"} for Funnel-Transformer,
\code{method="deberta_v2"} for Decoding-enhanced BERT with Disentangled Attention (DeBERTa V2),
\verb{method="glove"`` for GlobalVector Clusters, and }method="lda"` for topic modeling. See
details for more information.}

\item{\code{ml_framework}}{\code{string} Framework to use for the model.
\code{ml_framework="tensorflow"} for 'tensorflow' and \code{ml_framework="pytorch"}
for 'pytorch'. Only relevant for transformer models. To request bag-of-words model
set \code{ml_framework=NULL}.}

\item{\code{max_length}}{\code{int} determining the maximum length of token
sequences used in transformer models. Not relevant for the other methods.}

\item{\code{chunks}}{\code{int} Maximum number of chunks. Must be at least 2.}

\item{\code{overlap}}{\code{int} determining the number of tokens which should be added
at the beginning of the next chunk. Only relevant for transformer models.}

\item{\code{emb_layer_min}}{\code{int} or \code{string} determining the first layer to be included
in the creation of embeddings. An integer correspondents to the layer number. The first
layer has the number 1. Instead of an integer the following strings are possible:
\code{"start"} for the first layer, \code{"middle"} for the middle layer,
\code{"2_3_layer"} for the layer two-third layer, and \code{"last"} for the last layer.}

\item{\code{emb_layer_max}}{\code{int} or \code{string} determining the last layer to be included
in the creation of embeddings. An integer correspondents to the layer number. The first
layer has the number 1. Instead of an integer the following strings are possible:
\code{"start"} for the first layer, \code{"middle"} for the middle layer,
\code{"2_3_layer"} for the layer two-third layer, and \code{"last"} for the last layer.}

\item{\code{emb_pool_type}}{\code{string} determining the method for pooling the token embeddings
within each layer. If \code{"cls"} only the embedding of the CLS token is used. If
\code{"average"} the token embedding of all tokens are averaged (excluding padding tokens).
\verb{"cls} is not supported for \code{method="funnel"}.}

\item{\code{model_dir}}{\code{string} path to the directory where the
BERT model is stored.}

\item{\code{trace}}{\code{bool} \code{TRUE} prints information about the progress.
\code{FALSE} does not.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
In the case of any transformer (e.g.\code{method="bert"},
\code{method="roberta"}, and \code{method="longformer"}),
a pretrained transformer model must be supplied via \code{model_dir}.
}

\subsection{Returns}{
Returns an object of class \link{TextEmbeddingModel}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-load_from_disk"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-load_from_disk}{}}}
\subsection{Method \code{load_from_disk()}}{
loads an object from disk
and updates the object to the current version of the package.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$load_from_disk(dir_path)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{dir_path}}{Path where the object set is stored.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Method does not return anything. It loads an object from disk.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-load"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-load}{}}}
\subsection{Method \code{load()}}{
Method for loading a transformers model into R.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$load(dir_path)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{dir_path}}{\code{string} containing the path to the relevant
model directory.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used for loading a saved
transformer model into the R interface.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-save"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-save}{}}}
\subsection{Method \code{save()}}{
Method for saving a transformer model on disk.Relevant
only for transformer models.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$save(dir_path, folder_name)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{dir_path}}{\code{string} containing the path to the relevant
model directory.}

\item{\code{folder_name}}{\code{string} Name for the folder created within the directory.
This folder contains all model files.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used for saving a transformer model
to disk.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-encode"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-encode}{}}}
\subsection{Method \code{encode()}}{
Method for encoding words of raw texts into integers.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$encode(
  raw_text,
  token_encodings_only = FALSE,
  to_int = TRUE,
  trace = FALSE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{raw_text}}{\code{vector}containing the raw texts.}

\item{\code{token_encodings_only}}{\code{bool} If \code{TRUE}, only the token
encodings are returned. If \code{FALSE}, the complete encoding is returned
which is important for some transformer models.}

\item{\code{to_int}}{\code{bool} If \code{TRUE} the integer ids of the tokens are
returned. If \code{FALSE} the tokens are returned. Argument only applies
for transformer models and if \code{token_encodings_only=TRUE}.}

\item{\code{trace}}{\code{bool} If \code{TRUE}, information of the progress
is printed. \code{FALSE} if not requested.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
\code{list} containing the integer or token sequences of the raw texts with
special tokens.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-decode"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-decode}{}}}
\subsection{Method \code{decode()}}{
Method for decoding a sequence of integers into tokens
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$decode(int_seqence, to_token = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{int_seqence}}{\code{list} containing the integer sequences which
should be transformed to tokens or plain text.}

\item{\code{to_token}}{\code{bool} If \code{FALSE} plain text is returned.
If \code{TRUE} a sequence of tokens is returned. Argument only relevant
if the model is based on a transformer.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
\code{list} of token sequences
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_special_tokens"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_special_tokens}{}}}
\subsection{Method \code{get_special_tokens()}}{
Method for receiving the special tokens of the model
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_special_tokens()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{matrix} containing the special tokens in the rows
and their type, token, and id in the columns.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-embed"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-embed}{}}}
\subsection{Method \code{embed()}}{
Method for creating text embeddings from raw texts.
This method should only be used if a small number of texts should be transformed
into text embeddings. For a large number of texts please use the method \code{embed_large}.
In the case of using a GPU and running out of memory while using 'tensorflow'  reduce the
batch size or restart R and switch to use cpu only via \code{set_config_cpu_only}. In general,
not relevant for 'pytorch'.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$embed(
  raw_text = NULL,
  doc_id = NULL,
  batch_size = 8,
  trace = FALSE,
  return_large_dataset = FALSE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{raw_text}}{\code{vector} containing the raw texts.}

\item{\code{doc_id}}{\code{vector} containing the corresponding IDs for every text.}

\item{\code{batch_size}}{\code{int} determining the maximal size of every batch.}

\item{\code{trace}}{\code{bool} \code{TRUE}, if information about the progression
should be printed on console.}

\item{\code{return_large_dataset}}{'bool' If \code{TRUE} the retuned object is of class
\link{LargeDataSetForTextEmbeddings}. If \code{FALSE} it is of class \link{EmbeddedText}}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Method returns an object of class \link{EmbeddedText} or \link{LargeDataSetForTextEmbeddings}. This object
contains the embeddings as a \link{data.frame} and information about the
model creating the embeddings.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-embed_large"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-embed_large}{}}}
\subsection{Method \code{embed_large()}}{
Method for creating text embeddings from raw texts.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$embed_large(
  large_datas_set,
  batch_size = 32,
  trace = FALSE,
  log_file = NULL,
  log_write_interval = 2
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{large_datas_set}}{Object of class \link{LargeDataSetForText} containing the
raw texts.}

\item{\code{batch_size}}{\code{int} determining the maximal size of every batch.}

\item{\code{trace}}{\code{bool} \code{TRUE}, if information about the progression
should be printed on console.}

\item{\code{log_file}}{\code{string} Path to the file where the log should be saved.
If no logging is desired set this argument to \code{NULL}.}

\item{\code{log_write_interval}}{\code{int} Time in seconds determining the interval in which
the logger should try to update the log files. Only relevant if \code{log_file} is not \code{NULL}.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Method returns an object of class \link{LargeDataSetForTextEmbeddings}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-fill_mask"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-fill_mask}{}}}
\subsection{Method \code{fill_mask()}}{
Method for calculating tokens behind mask tokens.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$fill_mask(text, n_solutions = 5)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{text}}{\code{string} Text containing mask tokens.}

\item{\code{n_solutions}}{\code{int} Number estimated tokens for every mask.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Returns a \code{list} containing a \code{data.frame} for every
mask. The \code{data.frame} contains the solutions in the rows and reports
the score, token id, and token string in the columns.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-set_publication_info"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-set_publication_info}{}}}
\subsection{Method \code{set_publication_info()}}{
Method for setting the bibliographic information of the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$set_publication_info(type, authors, citation, url = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{type}}{\code{string} Type of information which should be changed/added.
\code{developer}, and \code{modifier} are possible.}

\item{\code{authors}}{List of people.}

\item{\code{citation}}{\code{string} Citation in free text.}

\item{\code{url}}{\code{string} Corresponding URL if applicable.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used to set the private
members for publication information of the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_publication_info"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_publication_info}{}}}
\subsection{Method \code{get_publication_info()}}{
Method for getting the bibliographic information of the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_publication_info()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{list} of bibliographic information.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-set_model_license"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-set_model_license}{}}}
\subsection{Method \code{set_model_license()}}{
Method for setting the license of the model
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$set_model_license(license = "CC BY")}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{license}}{\code{string} containing the abbreviation of the license or
the license text.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used for setting the private
member for the software license of the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_model_license"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_model_license}{}}}
\subsection{Method \code{get_model_license()}}{
Method for requesting the license of the model
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_model_license()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{string} License of the model
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-set_documentation_license"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-set_documentation_license}{}}}
\subsection{Method \code{set_documentation_license()}}{
Method for setting the license of models' documentation.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$set_documentation_license(license = "CC BY")}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{license}}{\code{string} containing the abbreviation of the license or
the license text.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used to set the private member for the
documentation license of the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_documentation_license"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_documentation_license}{}}}
\subsection{Method \code{get_documentation_license()}}{
Method for getting the license of the models' documentation.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_documentation_license()}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{license}}{\code{string} containing the abbreviation of the license or
the license text.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-set_model_description"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-set_model_description}{}}}
\subsection{Method \code{set_model_description()}}{
Method for setting a description of the model
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$set_model_description(
  eng = NULL,
  native = NULL,
  abstract_eng = NULL,
  abstract_native = NULL,
  keywords_eng = NULL,
  keywords_native = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{eng}}{\code{string} A text describing the training of the classifier,
its theoretical and empirical background, and the different output labels
in English.}

\item{\code{native}}{\code{string} A text describing the training of the classifier,
its theoretical and empirical background, and the different output labels
in the native language of the model.}

\item{\code{abstract_eng}}{\code{string} A text providing a summary of the description
in English.}

\item{\code{abstract_native}}{\code{string} A text providing a summary of the description
in the native language of the classifier.}

\item{\code{keywords_eng}}{\code{vector}of keywords in English.}

\item{\code{keywords_native}}{\code{vector}of keywords in the native language of the classifier.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used to set the private members for the
description of the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_model_description"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_model_description}{}}}
\subsection{Method \code{get_model_description()}}{
Method for requesting the model description.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_model_description()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{list} with the description of the model in English
and the native language.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_model_info"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_model_info}{}}}
\subsection{Method \code{get_model_info()}}{
Method for requesting the model information
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_model_info()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{list} of all relevant model information
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_package_versions"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_package_versions}{}}}
\subsection{Method \code{get_package_versions()}}{
Method for requesting a summary of the R and python packages'
versions used for creating the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_package_versions()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{list} containing the versions of the relevant
R and python packages.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_basic_components"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_basic_components}{}}}
\subsection{Method \code{get_basic_components()}}{
Method for requesting the part of interface's configuration that is
necessary for all models.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_basic_components()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{list}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_transformer_components"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_transformer_components}{}}}
\subsection{Method \code{get_transformer_components()}}{
Method for requesting the part of interface's configuration that is
necessary for transformer models.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_transformer_components()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{list}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_sustainability_data"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_sustainability_data}{}}}
\subsection{Method \code{get_sustainability_data()}}{
Method for requesting a log of tracked energy consumption
during training and an estimate of the resulting CO2 equivalents in kg.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_sustainability_data()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{matrix} containing the tracked energy consumption,
CO2 equivalents in kg, information on the tracker used, and technical
information on the training infrastructure for every training run.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_ml_framework"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_ml_framework}{}}}
\subsection{Method \code{get_ml_framework()}}{
Method for requesting the machine learning framework used
for the classifier.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_ml_framework()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{string} describing the machine learning framework used
for the classifier.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-count_parameter"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-count_parameter}{}}}
\subsection{Method \code{count_parameter()}}{
Method for counting the trainable parameters of a model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$count_parameter(with_head = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{with_head}}{\code{bool} If \code{TRUE} the number of parameters is returned including
the language modeling head of the model. If \code{FALSE} only the number of parameters of
the core model is returned.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Returns the number of trainable parameters of the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-is_configured"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-is_configured}{}}}
\subsection{Method \code{is_configured()}}{
Method for checking if the model was successfully configured.
An object can only be used if this value is \code{TRUE}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$is_configured()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{bool} \code{TRUE} if the model is fully configured. \code{FALSE} if not.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_private"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_private}{}}}
\subsection{Method \code{get_private()}}{
Method for requesting all private fields and methods. Used
for loading and updating an object.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_private()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{list} with all private fields and methods.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-get_all_fields"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-get_all_fields}{}}}
\subsection{Method \code{get_all_fields()}}{
Return all fields.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$get_all_fields()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Method returns a \code{list} containing all public and private fields
of the object.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TextEmbeddingModel-clone"></a>}}
\if{latex}{\out{\hypertarget{method-TextEmbeddingModel-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TextEmbeddingModel$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
