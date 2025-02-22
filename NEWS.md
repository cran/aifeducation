---
editor_options: 
  markdown: 
    wrap: 72
---

# aifeducation 1.0.2

- Fixed a bug with alpha 3 codes for sustainability tracking preventing
  Ai for Education Studio to start.
- The source urls of entries in LargeDataSetsForTexts are now displayed correctly
  within Ai for Education Studio.

# aifeducation 1.0.1

Fixed a bug with the initialization of 'codecarbon' on linux.

# aifeducation 1.0.0

First complete release of the package including major changes, bug
fixes, new features, and objects.

The most important change is that we decided to use 'PyTorch' for
several reasons. First, 'PyTorch' is a very flexible and stable machine
learning framework. At the moment, most new architectures are based on
'PyTorch' as can be seen on Hugging Face. Currently (11th November 2024)
there are 190,237 models for this framework compared to 13,346 models
for 'tensorflow'. Second, 'PyTorch' provides an easy installation and
supports native GPU acceleration on Linux and Windows while tensorflow
supports native GPU support only on Linux and for Windows only in
version 2.10 or lower. Fourth, keras, which was an important element of
'tensorflow', changed to a multi-back-end framework. However, keras 3.0
does not have a native Windows support. Since we assume that many
educational researchers use either Windows or Mac and are not familiar
with more complex system configurations (such as using Windows subsystem
for Linux (WSL)), this is problematic.

In addition, we changed the algorithm for saving and loading models,
data, and objects to ensure that models trained with the package are
working within future versions of *aifeducation* and can be updated to
new developments. This is also necessary to allow reproducibility of
models and research based on these models. To achieve this goal we had
to make some changes for models created with version 0.3.3 or lower. If
you still need these models, please install an older version of
*aifeducation*.

The following changes have been made:

**Major Changes**

-   The core machine learning framework is now 'PyTorch'. 'Tensorflow'
    is still supported but only for some models and limited to version
    2.15. Further implementation and support for 'tensorflow' models is
    currently not planned. We decided to base the package on 'PyTorch'
    because this framework is widely used in research, is very flexible,
    provides a broad GPU support, and offers more stable code across
    versions.
-   Implemented a new mechanic and new methods for all objects allowing
    objects that were created with an older version of the package to
    update to the current version during loading.
-   Removed the bag-of-words models from the package in order to focus
    the package on approaches which use AI.

**Installation and Configuration**

-   Added a new function for a convenient installation of 'python' and
    'pytorch'.

**Transformer Models**

-   Complete rewrite of all transformer functions into a modern
    object-oriented approach with R6 classes (AIFETransformerMaker).
-   Functions of type create_xxx_model and train_xxx_model are now
    deprecated.
-   Added support for MPNet with 'pytorch' and 'tensorflow'.

**TEFeatureExtractor**

-   Adding TEFeatureExtractor as a new class for 'pytorch' only.
-   TEFeatureExtractor are auto-encoders that can be used to reduce the
    number of features of text embeddings before passing them onto
    classifiers. Their aim is to reduce computational time and/or
    increase performance of classifiers.

**TextEmbeddingClassifiers**

-   TEClassifierRegular replaces TextEmbeddingClassifierNeuralNet. This
    new class provides additional methods and fixes a bug for pytorch
    models used to predict two classes.
-   TextEmbeddingClassifierNeuralNet is now deprecated.\
-   Added TEClassifierProtoNet which is a classifier that applys methods
    of meta-learning based on ProtoNets.
-   In comparison to TextEmbeddingClassifierNeuralNet, the training loop
    for the new classes was altered and reduced in its complexity for
    users. For example, only the type of pseudo-labeling described by
    Cascante-Bonilla et al. (2020) is now implemented and at the same
    type the technique described by Lee (2013) was removed. In addition,
    it is now possible to add synthetic cases within every step of
    pseudo-labeling. See the vignettes for more details.

**Graphical User Interface Aifeducation Studio**

-   Complete rewrite of the user interface based on bslib while removing
    the dependencies to shinydashboard.
-   User interface only supports pytorch and no longer tensorflow.
-   Implemented long running tasks such as training a transformer as a
    shiny ExtendedTask. This allows the computation of the task in the
    background and the shiny app to stay responsive. This, in turn,
    avoids "greying out" of the app.
-   Implemented a new reporting system for providing a feedback to the
    user during computations.

**Data Management**

-   Introduced two new classes LargeDataSetForTextEmbeddings and
    LargeDataSetForText based on the python libraries 'arrow' and
    'datasets' allowing to store and use data that would not fit into
    memory. LargeDataSetForText stores raw texts while
    LargeDataSetForTextEmbeddings contain text embeddings.
-   Added support to all AI models for these new kinds of objects to
    allow training with large data sets.
-   Added new methods to objects of class EmbeddedTexts (e.g. for
    converting EmbeddedTexts into a LargeDataSetForTextEmbeddings). See
    the corresponding documentation for more details.
-   The function combine_embeddings is now deprecated. Please use the
    corresponding method of EmbeddedTexts.

**Saving and Loading**

-   Introduced save_to_disk and load_from_disk as the new core functions
    for saving and loading objects and models of this package.
-   Functions load_ai_model and save_ai_model are now deprecated. Please
    use these functions only for models created with version 0.3.3 or
    lower.

**Further Changes**

-   Removed the dependencies to package abind and irr.
-   Updated vignettes.

# aifeducation 0.3.3

**Graphical User Interface Aifeducation Studio**

-   Fixed a bug concerning the IDs of .pdf and .csv files. Now the IDs
    are correctly saved within a text collection file.
-   Fixed a bug while checking for the selection of at least one file
    type during creation of a text collection.

**TextEmbeddingClassifiers**

-   Fixed the process for checking if TextEmbeddingModels are
    compatible.

**Python Installation**

-   Fixed a bug which caused the installation of incompatible versions
    of keras and Tensorflow.

**Further Changes**

-   Removed quanteda.textmodels as necessary library for testing the
    package.
-   Added a dataset for testing the package based on Maas et al. (2011).

# aifeducation 0.3.2

**TextEmbeddingClassifiers**

-   Fixed a bug in GlobalAveragePooling1D_PT. Now the layer makes a
    correct pooling. **This change has an effect on PyTorch models
    trained with version 0.3.1.**

**TextEmbeddingModel**

-   Replaced the parameter 'aggregation' with three new parameters
    allowing to explicitly choose the start and end layer to be included
    in the creation of embeddings. Furthermore, two options for the
    pooling method within each layer is added ("cls" and "average").
-   Added support for reporting the training and validation loss during
    training the corresponding base model.

**Transformer Models**

-   Fixed a bug in the creation of all transformer models except funnel.
    Now choosing the number of layers is working.
-   A file 'history.log' is now saved within the model's folder
    reporting the loss and validation loss during training for each
    epoch.

**EmbeddedText**

-   Changed the process for validating if EmbeddedTexts are compatible.
    Now only the model's unique name is used for the validation.
-   Added new fields and updated methods to account for the new options
    in creating embeddings (layer selection and pooling type).

**Graphical User Interface Aifeducation Studio**

-   Adapted the interface according to the changes made in this version.
-   Improved the read of raw texts. Reading now reduces multiple spaces
    characters to one single space character. Hyphenation is removed.

**Python Installation**

-   Updated installation to account for the new version of keras.

# aifeducation 0.3.1

**Graphical User Interface Aifeducation Studio**

-   Added a shiny app to the package that serves as a graphical user
    interface.

**Transformer Models**

-   Fixed a bug in all transformers except BERT concerning the
    unk_token.
-   Switched from SentencePiece tokenizer to WordPiece tokenizer for
    DeBERTa_V2.
-   Add the possibility to train DeBERTa_V2 and FunnelTransformer models
    with Whole Word Masking.

**TextEmbeddingModel**

-   Added a method for 'fill-mask'.
-   Added a new argument to the method 'encode', allowing to chose
    between encoding into token ids or into token strings.
-   Added a new argument to the method 'decode', allowing to chose
    between decoding into single tokens or into plain text.
-   Fixed a bug for embedding texts when using pytorch. The fix should
    decrease computational time and enables gpu support (if available on
    machine).
-   Fixed two missing columns for saving the results of sustainability
    tracking on machines without gpu.
-   Implemented the advantages of datasets from the python library
    'datasets' increasing computational speed and allowing the use of
    large datasets.

**TextEmbeddingClassifiers**

-   Adding support for pytorch without the need for kerasV3 or
    keras-core. Classifiers for pytorch are now implemented in native
    pytorch.
-   Changed the architecture for new classifiers and extended the
    abilities of neural nets by adding the possibility to add positional
    embedding.
-   Changed the architecture for new classifiers and extended the
    abilities of neural nets by adding an alternative method for the
    self-attention mechanism via fourier transformation (similar to
    FNet).
-   Added balanced_accuracy as the new metric for determining which
    state of a model predicts classes best.
-   Fixed error that training history is not saved correctly.
-   Added a record metric for the test dataset to training history with
    pytorch.
-   Added the option to balance class weights for calculating training
    loss according to the Inverse Frequency method. Balance class
    weights is activated by default.
-   Added a method for checking the compatibility of the underlying
    TextEmbeddingModels of a classifier and an object of class
    EmbeddedText.
-   Added precision, recall, and f1-score as new metrics.

**Python Installation**

-   Added an argument to 'install_py_modules', allowing to choose which
    machine learning framework should be installed.
-   Updated 'check_aif_py_modules'.

**Further Changes**

-   Setting the machine learning framework at the start of a session is
    no longer necessary. The function for setting the global
    ml_framework remains active for convenience. The ml_framework can
    now be switched at any time during a session.
-   Updated documentation.

# aifeducation 0.3.0

-   Added DeBERTa and Funnel-Transformer support.
-   Fixed issues for installing the required python packages.
-   Fixed issues in training transformer models.
-   Fixed an issue for calculating the final iota values in classifiers
    if pseudo labeling is active.
-   Added support for PyTorch and Tensorflow for all transformer models.
-   Added support for PyTorch for classifier objects via keras 3 in the
    future.
-   Removed augmentation of vocabulary from training BERT models.
-   Updated documentation.
-   Changed the reported values for kappa.

# aifeducation 0.2.0

-   First release on CRAN
