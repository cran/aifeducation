% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/AIFEBaseModel.R
\name{AIFEBaseModel}
\alias{AIFEBaseModel}
\title{Base class for models using neural nets}
\value{
Objects of this containing fields and methods used in several other classes in 'ai for education'. This class
is \strong{not} designed for a direct application and should only be used by developers.
}
\description{
Abstract class for all models that do not rely on the python library 'transformers'.
}
\concept{Classifiers for developers}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{model}}{('tensorflow_model' or 'pytorch_model')\cr
Field for storing the 'tensorflow' or 'pytorch' model after loading.}

\item{\code{model_config}}{('list()')\cr
List for storing information about the configuration of the model.}

\item{\code{last_training}}{('list()')\cr
List for storing the history, the configuration, and the results of the last
training. This information will be overwritten if a new training is started.
\itemize{
\item \code{last_training$start_time}: Time point when training started.
\item \code{last_training$learning_time}: Duration of the training process.
\item \code{last_training$finish_time}: Time when the last training finished.
\item \code{last_training$history}: History of the last training.
\item \code{last_training$data}: Object of class \code{table} storing the initial frequencies of the passed data.
\item \code{last_training$config}: List storing the configuration used for the last training.
}}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-AIFEBaseModel-get_model_info}{\code{AIFEBaseModel$get_model_info()}}
\item \href{#method-AIFEBaseModel-get_text_embedding_model}{\code{AIFEBaseModel$get_text_embedding_model()}}
\item \href{#method-AIFEBaseModel-set_publication_info}{\code{AIFEBaseModel$set_publication_info()}}
\item \href{#method-AIFEBaseModel-get_publication_info}{\code{AIFEBaseModel$get_publication_info()}}
\item \href{#method-AIFEBaseModel-set_model_license}{\code{AIFEBaseModel$set_model_license()}}
\item \href{#method-AIFEBaseModel-get_model_license}{\code{AIFEBaseModel$get_model_license()}}
\item \href{#method-AIFEBaseModel-set_documentation_license}{\code{AIFEBaseModel$set_documentation_license()}}
\item \href{#method-AIFEBaseModel-get_documentation_license}{\code{AIFEBaseModel$get_documentation_license()}}
\item \href{#method-AIFEBaseModel-set_model_description}{\code{AIFEBaseModel$set_model_description()}}
\item \href{#method-AIFEBaseModel-get_model_description}{\code{AIFEBaseModel$get_model_description()}}
\item \href{#method-AIFEBaseModel-save}{\code{AIFEBaseModel$save()}}
\item \href{#method-AIFEBaseModel-load}{\code{AIFEBaseModel$load()}}
\item \href{#method-AIFEBaseModel-get_package_versions}{\code{AIFEBaseModel$get_package_versions()}}
\item \href{#method-AIFEBaseModel-get_sustainability_data}{\code{AIFEBaseModel$get_sustainability_data()}}
\item \href{#method-AIFEBaseModel-get_ml_framework}{\code{AIFEBaseModel$get_ml_framework()}}
\item \href{#method-AIFEBaseModel-get_text_embedding_model_name}{\code{AIFEBaseModel$get_text_embedding_model_name()}}
\item \href{#method-AIFEBaseModel-check_embedding_model}{\code{AIFEBaseModel$check_embedding_model()}}
\item \href{#method-AIFEBaseModel-count_parameter}{\code{AIFEBaseModel$count_parameter()}}
\item \href{#method-AIFEBaseModel-is_configured}{\code{AIFEBaseModel$is_configured()}}
\item \href{#method-AIFEBaseModel-get_private}{\code{AIFEBaseModel$get_private()}}
\item \href{#method-AIFEBaseModel-get_all_fields}{\code{AIFEBaseModel$get_all_fields()}}
\item \href{#method-AIFEBaseModel-clone}{\code{AIFEBaseModel$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_model_info"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_model_info}{}}}
\subsection{Method \code{get_model_info()}}{
Method for requesting the model information.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_model_info()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{list} of all relevant model information.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_text_embedding_model"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_text_embedding_model}{}}}
\subsection{Method \code{get_text_embedding_model()}}{
Method for requesting the text embedding model information.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_text_embedding_model()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{list} of all relevant model information on the text embedding model underlying the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-set_publication_info"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-set_publication_info}{}}}
\subsection{Method \code{set_publication_info()}}{
Method for setting publication information of the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$set_publication_info(authors, citation, url = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{authors}}{List of authors.}

\item{\code{citation}}{Free text citation.}

\item{\code{url}}{URL of a corresponding homepage.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used for setting the private members for publication information.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_publication_info"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_publication_info}{}}}
\subsection{Method \code{get_publication_info()}}{
Method for requesting the bibliographic information of the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_publication_info()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{list} with all saved bibliographic information.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-set_model_license"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-set_model_license}{}}}
\subsection{Method \code{set_model_license()}}{
Method for setting the license of the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$set_model_license(license = "CC BY")}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{license}}{\code{string} containing the abbreviation of the license or the license text.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used for setting the private member for the software license of
the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_model_license"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_model_license}{}}}
\subsection{Method \code{get_model_license()}}{
Method for getting the license of the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_model_license()}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{license}}{\code{string} containing the abbreviation of the license or the license text.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
\code{string} representing the license for the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-set_documentation_license"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-set_documentation_license}{}}}
\subsection{Method \code{set_documentation_license()}}{
Method for setting the license of the model's documentation.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$set_documentation_license(license = "CC BY")}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{license}}{\code{string} containing the abbreviation of the license or the license text.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used for setting the private member for the documentation license
of the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_documentation_license"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_documentation_license}{}}}
\subsection{Method \code{get_documentation_license()}}{
Method for getting the license of the model's documentation.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_documentation_license()}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{license}}{\code{string} containing the abbreviation of the license or the license text.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Returns the license as a \code{string}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-set_model_description"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-set_model_description}{}}}
\subsection{Method \code{set_model_description()}}{
Method for setting a description of the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$set_model_description(
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
\item{\code{eng}}{\code{string} A text describing the training, its theoretical and empirical background, and output in
English.}

\item{\code{native}}{\code{string} A text describing the training , its theoretical and empirical background, and output in
the native language of the model.}

\item{\code{abstract_eng}}{\code{string} A text providing a summary of the description in English.}

\item{\code{abstract_native}}{\code{string} A text providing a summary of the description in the native language of the
model.}

\item{\code{keywords_eng}}{\code{vector} of keyword in English.}

\item{\code{keywords_native}}{\code{vector} of keyword in the native language of the model.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used for setting the private members for the description of the
model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_model_description"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_model_description}{}}}
\subsection{Method \code{get_model_description()}}{
Method for requesting the model description.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_model_description()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{list} with the description of the classifier in English and the native language.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-save"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-save}{}}}
\subsection{Method \code{save()}}{
Method for saving a model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$save(dir_path, folder_name)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{dir_path}}{\code{string} Path of the directory where the model should be saved.}

\item{\code{folder_name}}{\code{string} Name of the folder that should be created within the directory.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It saves the model to disk.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-load"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-load}{}}}
\subsection{Method \code{load()}}{
Method for importing a model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$load(dir_path)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{dir_path}}{\code{string} Path of the directory where the model is saved.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
Function does not return a value. It is used to load the weights of a model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_package_versions"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_package_versions}{}}}
\subsection{Method \code{get_package_versions()}}{
Method for requesting a summary of the R and python packages' versions used for creating the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_package_versions()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{list} containing the versions of the relevant R and python packages.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_sustainability_data"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_sustainability_data}{}}}
\subsection{Method \code{get_sustainability_data()}}{
Method for requesting a summary of tracked energy consumption during training and an estimate of the
resulting CO2 equivalents in kg.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_sustainability_data()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{list} containing the tracked energy consumption, CO2 equivalents in kg, information on the
tracker used, and technical information on the training infrastructure.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_ml_framework"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_ml_framework}{}}}
\subsection{Method \code{get_ml_framework()}}{
Method for requesting the machine learning framework used for the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_ml_framework()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{string} describing the machine learning framework used for the classifier.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_text_embedding_model_name"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_text_embedding_model_name}{}}}
\subsection{Method \code{get_text_embedding_model_name()}}{
Method for requesting the name (unique id) of the underlying text embedding model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_text_embedding_model_name()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{string} describing name of the text embedding model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-check_embedding_model"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-check_embedding_model}{}}}
\subsection{Method \code{check_embedding_model()}}{
Method for checking if the provided text embeddings are created with the same \link{TextEmbeddingModel}
as the model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$check_embedding_model(text_embeddings)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{text_embeddings}}{Object of class \link{EmbeddedText} or \link{LargeDataSetForTextEmbeddings}.}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
\code{TRUE} if the underlying \link{TextEmbeddingModel} are the same. \code{FALSE} if the models differ.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-count_parameter"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-count_parameter}{}}}
\subsection{Method \code{count_parameter()}}{
Method for counting the trainable parameters of a model.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$count_parameter()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns the number of trainable parameters of the model.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-is_configured"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-is_configured}{}}}
\subsection{Method \code{is_configured()}}{
Method for checking if the model was successfully configured. An object can only be used if this
value is \code{TRUE}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$is_configured()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{bool} \code{TRUE} if the model is fully configured. \code{FALSE} if not.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_private"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_private}{}}}
\subsection{Method \code{get_private()}}{
Method for requesting all private fields and methods. Used for loading and updating an object.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_private()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Returns a \code{list} with all private fields and methods.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-get_all_fields"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-get_all_fields}{}}}
\subsection{Method \code{get_all_fields()}}{
Return all fields.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$get_all_fields()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Method returns a \code{list} containing all public and private fields
of the object.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-AIFEBaseModel-clone"></a>}}
\if{latex}{\out{\hypertarget{method-AIFEBaseModel-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{AIFEBaseModel$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
