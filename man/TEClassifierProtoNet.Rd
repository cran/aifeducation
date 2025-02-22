% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/TEClassifierProtoNet.R
\name{TEClassifierProtoNet}
\alias{TEClassifierProtoNet}
\title{Text embedding classifier with a ProtoNet}
\value{
Objects of this class are used for assigning texts to classes/categories. For the creation and training of a
classifier an object of class \link{EmbeddedText} or \link{LargeDataSetForTextEmbeddings} and a \code{factor} are necessary. The
object of class \link{EmbeddedText} or \link{LargeDataSetForTextEmbeddings} contains the numerical text representations (text
embeddings) of the raw texts generated by an object of class \link{TextEmbeddingModel}. The \code{factor} contains the
classes/categories for every text. Missing values (unlabeled cases) are supported. For predictions an object of
class \link{EmbeddedText} or \link{LargeDataSetForTextEmbeddings} has to be used which was created with the same
\link{TextEmbeddingModel} as for training.
}
\description{
Abstract class for neural nets with 'keras'/'tensorflow' and 'pytorch'.

This object represents in implementation of a prototypical network for few-shot learning as described by Snell,
Swersky, and Zemel (2017). The network uses a multi way contrastive loss described by Zhang et al. (2019). The
network learns to scale the metric as described by Oreshkin, Rodriguez, and Lacoste (2018)
}
\references{
Oreshkin, B. N., Rodriguez, P. & Lacoste, A. (2018). TADAM: Task dependent adaptive metric for improved
few-shot learning. https://doi.org/10.48550/arXiv.1805.10123

Snell, J., Swersky, K. & Zemel, R. S. (2017). Prototypical Networks for Few-shot Learning.
https://doi.org/10.48550/arXiv.1703.05175

Zhang, X., Nie, J., Zong, L., Yu, H. & Liang, W. (2019). One Shot Learning with Margin. In Q. Yang, Z.-H.
Zhou, Z. Gong, M.-L. Zhang & S.-J. Huang (Eds.), Lecture Notes in Computer Science. Advances in Knowledge Discovery
and Data Mining (Vol. 11440, pp. 305–317). Springer International Publishing.
https://doi.org/10.1007/978-3-030-16145-3_24
}
\seealso{
Other Classification: 
\code{\link{TEClassifierRegular}}
}
\concept{Classification}
\section{Super classes}{
\code{\link[aifeducation:AIFEBaseModel]{aifeducation::AIFEBaseModel}} -> \code{\link[aifeducation:TEClassifierRegular]{aifeducation::TEClassifierRegular}} -> \code{TEClassifierProtoNet}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-TEClassifierProtoNet-configure}{\code{TEClassifierProtoNet$configure()}}
\item \href{#method-TEClassifierProtoNet-train}{\code{TEClassifierProtoNet$train()}}
\item \href{#method-TEClassifierProtoNet-embed}{\code{TEClassifierProtoNet$embed()}}
\item \href{#method-TEClassifierProtoNet-plot_embeddings}{\code{TEClassifierProtoNet$plot_embeddings()}}
\item \href{#method-TEClassifierProtoNet-clone}{\code{TEClassifierProtoNet$clone()}}
}
}
\if{html}{\out{
<details><summary>Inherited methods</summary>
<ul>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="count_parameter"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-count_parameter'><code>aifeducation::AIFEBaseModel$count_parameter()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_all_fields"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_all_fields'><code>aifeducation::AIFEBaseModel$get_all_fields()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_documentation_license"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_documentation_license'><code>aifeducation::AIFEBaseModel$get_documentation_license()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_ml_framework"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_ml_framework'><code>aifeducation::AIFEBaseModel$get_ml_framework()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_model_description"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_model_description'><code>aifeducation::AIFEBaseModel$get_model_description()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_model_info"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_model_info'><code>aifeducation::AIFEBaseModel$get_model_info()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_model_license"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_model_license'><code>aifeducation::AIFEBaseModel$get_model_license()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_package_versions"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_package_versions'><code>aifeducation::AIFEBaseModel$get_package_versions()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_private"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_private'><code>aifeducation::AIFEBaseModel$get_private()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_publication_info"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_publication_info'><code>aifeducation::AIFEBaseModel$get_publication_info()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_sustainability_data"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_sustainability_data'><code>aifeducation::AIFEBaseModel$get_sustainability_data()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_text_embedding_model"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_text_embedding_model'><code>aifeducation::AIFEBaseModel$get_text_embedding_model()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="get_text_embedding_model_name"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-get_text_embedding_model_name'><code>aifeducation::AIFEBaseModel$get_text_embedding_model_name()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="is_configured"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-is_configured'><code>aifeducation::AIFEBaseModel$is_configured()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="load"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-load'><code>aifeducation::AIFEBaseModel$load()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="set_documentation_license"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-set_documentation_license'><code>aifeducation::AIFEBaseModel$set_documentation_license()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="set_model_description"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-set_model_description'><code>aifeducation::AIFEBaseModel$set_model_description()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="set_model_license"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-set_model_license'><code>aifeducation::AIFEBaseModel$set_model_license()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="AIFEBaseModel" data-id="set_publication_info"><a href='../../aifeducation/html/AIFEBaseModel.html#method-AIFEBaseModel-set_publication_info'><code>aifeducation::AIFEBaseModel$set_publication_info()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="TEClassifierRegular" data-id="check_embedding_model"><a href='../../aifeducation/html/TEClassifierRegular.html#method-TEClassifierRegular-check_embedding_model'><code>aifeducation::TEClassifierRegular$check_embedding_model()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="TEClassifierRegular" data-id="check_feature_extractor_object_type"><a href='../../aifeducation/html/TEClassifierRegular.html#method-TEClassifierRegular-check_feature_extractor_object_type'><code>aifeducation::TEClassifierRegular$check_feature_extractor_object_type()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="TEClassifierRegular" data-id="load_from_disk"><a href='../../aifeducation/html/TEClassifierRegular.html#method-TEClassifierRegular-load_from_disk'><code>aifeducation::TEClassifierRegular$load_from_disk()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="TEClassifierRegular" data-id="predict"><a href='../../aifeducation/html/TEClassifierRegular.html#method-TEClassifierRegular-predict'><code>aifeducation::TEClassifierRegular$predict()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="TEClassifierRegular" data-id="requires_compression"><a href='../../aifeducation/html/TEClassifierRegular.html#method-TEClassifierRegular-requires_compression'><code>aifeducation::TEClassifierRegular$requires_compression()</code></a></span></li>
<li><span class="pkg-link" data-pkg="aifeducation" data-topic="TEClassifierRegular" data-id="save"><a href='../../aifeducation/html/TEClassifierRegular.html#method-TEClassifierRegular-save'><code>aifeducation::TEClassifierRegular$save()</code></a></span></li>
</ul>
</details>
}}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TEClassifierProtoNet-configure"></a>}}
\if{latex}{\out{\hypertarget{method-TEClassifierProtoNet-configure}{}}}
\subsection{Method \code{configure()}}{
Creating a new instance of this class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TEClassifierProtoNet$configure(
  ml_framework = "pytorch",
  name = NULL,
  label = NULL,
  text_embeddings = NULL,
  feature_extractor = NULL,
  target_levels = NULL,
  dense_size = 4,
  dense_layers = 0,
  rec_size = 4,
  rec_layers = 2,
  rec_type = "gru",
  rec_bidirectional = FALSE,
  embedding_dim = 2,
  self_attention_heads = 0,
  intermediate_size = NULL,
  attention_type = "fourier",
  add_pos_embedding = TRUE,
  rec_dropout = 0.1,
  repeat_encoder = 1,
  dense_dropout = 0.4,
  recurrent_dropout = 0.4,
  encoder_dropout = 0.1,
  optimizer = "adam"
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{ml_framework}}{\code{string} Currently only pytorch is supported (\code{ml_framework="pytorch"}).}

\item{\code{name}}{\code{string} Name of the new classifier. Please refer to common name conventions. Free text can be used
with parameter \code{label}.}

\item{\code{label}}{\code{string} Label for the new classifier. Here you can use free text.}

\item{\code{text_embeddings}}{An object of class \link{TextEmbeddingModel} or \link{LargeDataSetForTextEmbeddings}.}

\item{\code{feature_extractor}}{Object of class \link{TEFeatureExtractor} which should be used in order to reduce the number
of dimensions of the text embeddings. If no feature extractor should be applied set \code{NULL}.}

\item{\code{target_levels}}{\code{vector} containing the levels (categories or classes) within the target data. Please not
that order matters. For ordinal data please ensure that the levels are sorted correctly with later levels
indicating a higher category/class. For nominal data the order does not matter.}

\item{\code{dense_size}}{\code{int} Number of neurons for each dense layer.}

\item{\code{dense_layers}}{\code{int} Number of dense layers.}

\item{\code{rec_size}}{\code{int} Number of neurons for each recurrent layer.}

\item{\code{rec_layers}}{\code{int} Number of recurrent layers.}

\item{\code{rec_type}}{\code{string} Type of the recurrent layers.\code{rec_type="gru"} for Gated Recurrent Unit and
\code{rec_type="lstm"} for Long Short-Term Memory.}

\item{\code{rec_bidirectional}}{\code{bool} If \code{TRUE} a bidirectional version of the recurrent layers is used.}

\item{\code{embedding_dim}}{\code{int} determining the number of dimensions for the text embedding.}

\item{\code{self_attention_heads}}{\code{int} determining the number of attention heads for a self-attention layer. Only
relevant if \code{attention_type="multihead"}.}

\item{\code{intermediate_size}}{\code{int} determining the size of the projection layer within a each transformer encoder.}

\item{\code{attention_type}}{\code{string} Choose the relevant attention type. Possible values are \code{"fourier"} and
\code{"multihead"}. Please note that you may see different values for a case for different input orders if you choose \code{fourier} on linux.}

\item{\code{add_pos_embedding}}{\code{bool} \code{TRUE} if positional embedding should be used.}

\item{\code{rec_dropout}}{\code{double} ranging between 0 and lower 1, determining the dropout between bidirectional
recurrent layers.}

\item{\code{repeat_encoder}}{\code{int} determining how many times the encoder should be added to the network.}

\item{\code{dense_dropout}}{\code{double} ranging between 0 and lower 1, determining the dropout between dense layers.}

\item{\code{recurrent_dropout}}{\code{double} ranging between 0 and lower 1, determining the recurrent dropout for each
recurrent layer. Only relevant for keras models.}

\item{\code{encoder_dropout}}{\code{double} ranging between 0 and lower 1, determining the dropout for the dense projection
within the encoder layers.}

\item{\code{optimizer}}{\code{string} \code{"adam"} or \code{"rmsprop"} .}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Returns an object of class \link{TEClassifierProtoNet} which is ready for training.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TEClassifierProtoNet-train"></a>}}
\if{latex}{\out{\hypertarget{method-TEClassifierProtoNet-train}{}}}
\subsection{Method \code{train()}}{
Method for training a neural net.

Training includes a routine for early stopping. In the case that loss<0.0001
and Accuracy=1.00 and Average Iota=1.00 training stops. The history uses the values
of the last trained epoch for the remaining epochs.

After training the model with the best values for Average Iota, Accuracy, and Loss
on the validation data set is used as the final model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TEClassifierProtoNet$train(
  data_embeddings,
  data_targets,
  data_folds = 5,
  data_val_size = 0.25,
  use_sc = TRUE,
  sc_method = "dbsmote",
  sc_min_k = 1,
  sc_max_k = 10,
  use_pl = TRUE,
  pl_max_steps = 3,
  pl_max = 1,
  pl_anchor = 1,
  pl_min = 0,
  sustain_track = TRUE,
  sustain_iso_code = NULL,
  sustain_region = NULL,
  sustain_interval = 15,
  epochs = 40,
  batch_size = 35,
  Ns = 5,
  Nq = 3,
  loss_alpha = 0.5,
  loss_margin = 0.5,
  sampling_separate = FALSE,
  sampling_shuffle = TRUE,
  dir_checkpoint,
  trace = TRUE,
  ml_trace = 1,
  log_dir = NULL,
  log_write_interval = 10,
  n_cores = auto_n_cores()
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{data_embeddings}}{Object of class \link{EmbeddedText} or \link{LargeDataSetForTextEmbeddings}.}

\item{\code{data_targets}}{\code{factor} containing the labels for cases stored in \code{data_embeddings}. Factor must be named
and has to use the same names used in \code{data_embeddings}.}

\item{\code{data_folds}}{\code{int} determining the number of cross-fold samples.}

\item{\code{data_val_size}}{\code{double} between 0 and 1, indicating the proportion of cases of each class which should be
used for the validation sample during the estimation of the model. The remaining cases are part of the training
data.}

\item{\code{use_sc}}{\code{bool} \code{TRUE} if the estimation should integrate synthetic cases. \code{FALSE} if not.}

\item{\code{sc_method}}{\code{vector} containing the method for generating synthetic cases. Possible are \code{sc_method="adas"},
\code{sc_method="smote"}, and \code{sc_method="dbsmote"}.}

\item{\code{sc_min_k}}{\code{int} determining the minimal number of k which is used for creating synthetic units.}

\item{\code{sc_max_k}}{\code{int} determining the maximal number of k which is used for creating synthetic units.}

\item{\code{use_pl}}{\code{bool} \code{TRUE} if the estimation should integrate pseudo-labeling. \code{FALSE} if not.}

\item{\code{pl_max_steps}}{\code{int} determining the maximum number of steps during pseudo-labeling.}

\item{\code{pl_max}}{\code{double} between 0 and 1, setting the maximal level of confidence for considering a case for
pseudo-labeling.}

\item{\code{pl_anchor}}{\code{double} between 0 and 1 indicating the reference point for sorting the new cases of every
label. See notes for more details.}

\item{\code{pl_min}}{\code{double} between 0 and 1, setting the minimal level of confidence for considering a case for
pseudo-labeling.}

\item{\code{sustain_track}}{\code{bool} If \code{TRUE} energy consumption is tracked during training via the python library
'codecarbon'.}

\item{\code{sustain_iso_code}}{\code{string} ISO code (Alpha-3-Code) for the country. This variable must be set if
sustainability should be tracked. A list can be found on Wikipedia:
\url{https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes}.}

\item{\code{sustain_region}}{Region within a country. Only available for USA and Canada See the documentation of
codecarbon for more information. \url{https://mlco2.github.io/codecarbon/parameters.html}}

\item{\code{sustain_interval}}{\code{int} Interval in seconds for measuring power usage.}

\item{\code{epochs}}{\code{int} Number of training epochs.}

\item{\code{batch_size}}{\code{int} Size of the batches for training.}

\item{\code{Ns}}{\code{int} Number of cases for every class in the sample.}

\item{\code{Nq}}{\code{int} Number of cases for every class in the query.}

\item{\code{loss_alpha}}{\code{double} Value between 0 and 1 indicating how strong the loss should focus on pulling cases to
its corresponding prototypes or pushing cases away from other prototypes. The higher the value the more the
loss concentrates on pulling cases to its corresponding prototypes.}

\item{\code{loss_margin}}{\code{double} Value greater 0 indicating the minimal distance of every case from prototypes of
other classes}

\item{\code{sampling_separate}}{\code{bool} If \code{TRUE} the cases for every class are divided into a data set for sample and for query.
These are never mixed. If \code{TRUE} sample and query cases are drawn from the same data pool. That is, a case can be
part of sample in one epoch and in another epoch it can be part of query. It is ensured that a case is never part of
sample and query at the same time. In addition, it is ensured that every cases exists only once during
a training step.}

\item{\code{sampling_shuffle}}{\code{bool} If \code{TRUE} cases a randomly drawn from the data during every step. If \code{FALSE}
the cases are not shuffled.}

\item{\code{dir_checkpoint}}{\code{string} Path to the directory where the checkpoint during training should be saved. If the
directory does not exist, it is created.}

\item{\code{trace}}{\code{bool} \code{TRUE}, if information about the estimation phase should be printed to the console.}

\item{\code{ml_trace}}{\code{int} \code{ml_trace=0} does not print any information about the training process from pytorch on the
console.}

\item{\code{log_dir}}{\code{string} Path to the directory where the log files should be saved. If no logging is desired set
this argument to \code{NULL}.}

\item{\code{log_write_interval}}{\code{int} Time in seconds determining the interval in which the logger should try to update
the log files. Only relevant if \code{log_dir} is not \code{NULL}.}

\item{\code{n_cores}}{\code{int} Number of cores which should be used during the calculation of synthetic cases. Only relevant if
\code{use_sc=TRUE}.}

\item{\code{balance_class_weights}}{\code{bool} If \code{TRUE} class weights are generated based on the frequencies of the
training data with the method Inverse Class Frequency'. If \code{FALSE} each class has the weight 1.}

\item{\code{balance_sequence_length}}{\code{bool} If \code{TRUE} sample weights are generated for the length of sequences based on
the frequencies of the training data with the method Inverse Class Frequency'. If \code{FALSE} each sequences length
has the weight 1.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
\itemize{
\item \code{sc_max_k}: All values from sc_min_k up to sc_max_k are successively used. If
the number of \code{sc_max_k} is too high, the value is reduced to a number that allows the calculating of synthetic
units.
\item \verb{pl_anchor:} With the help of this value, the new cases are sorted. For
this aim, the distance from the anchor is calculated and all cases are arranged into an ascending order.
}
}

\subsection{Returns}{
Function does not return a value. It changes the object into a trained classifier.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TEClassifierProtoNet-embed"></a>}}
\if{latex}{\out{\hypertarget{method-TEClassifierProtoNet-embed}{}}}
\subsection{Method \code{embed()}}{
Method for embedding documents. Please do not confuse this type of embeddings with the embeddings of
texts created by an object of class \link{TextEmbeddingModel}. These embeddings embed documents according to their
similarity to specific classes.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TEClassifierProtoNet$embed(embeddings_q = NULL, batch_size = 32)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{embeddings_q}}{Object of class \link{EmbeddedText} or \link{LargeDataSetForTextEmbeddings} containing the text
embeddings for all cases which should be embedded into the classification space.}

\item{\code{batch_size}}{\code{int} batch size.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Returns a \code{list} containing the following elements
\itemize{
\item \code{embeddings_q}: embeddings for the cases (query sample).
\item \code{embeddings_prototypes}: embeddings of the prototypes which were learned during training. They represents the
center for the different classes.
}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TEClassifierProtoNet-plot_embeddings"></a>}}
\if{latex}{\out{\hypertarget{method-TEClassifierProtoNet-plot_embeddings}{}}}
\subsection{Method \code{plot_embeddings()}}{
Method for creating a plot to visualize embeddings and their corresponding centers (prototypes).
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TEClassifierProtoNet$plot_embeddings(
  embeddings_q,
  classes_q = NULL,
  batch_size = 12,
  alpha = 0.5,
  size_points = 3,
  size_points_prototypes = 8,
  inc_unlabeled = TRUE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{embeddings_q}}{Object of class \link{EmbeddedText} or \link{LargeDataSetForTextEmbeddings} containing the text
embeddings for all cases which should be embedded into the classification space.}

\item{\code{classes_q}}{Named \code{factor} containg the true classes for every case. Please note that the names must match
the names/ids in \code{embeddings_q}.}

\item{\code{batch_size}}{\code{int} batch size.}

\item{\code{alpha}}{\code{float} Value indicating how transparent the points should be (important
if many points overlap). Does not apply to points representing prototypes.}

\item{\code{size_points}}{\code{int} Size of the points excluding the points for prototypes.}

\item{\code{size_points_prototypes}}{\code{int} Size of points representing prototypes.}

\item{\code{inc_unlabeled}}{\code{bool} If \code{TRUE} plot includes unlabeled cases as data points.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Returns a plot of class \code{ggplot}visualizing embeddings.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-TEClassifierProtoNet-clone"></a>}}
\if{latex}{\out{\hypertarget{method-TEClassifierProtoNet-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{TEClassifierProtoNet$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
