## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, include = FALSE---------------------------------------------------
library(aifeducation)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_model$set_model_description(
#    eng=NULL,
#    native=NULL,
#    abstract_eng=NULL,
#    abstract_native=NULL,
#    keywords_eng=NULL,
#    keywords_native=NULL)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_model$get_model_description()

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_model$set_publication_info(
#    type,
#    authors,
#    citation,
#    url=NULL)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_model$get_publication_info()

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_model$set_software_license("GPL-3")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_model$set_documentation_license("CC BY-SA")

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
#    model_dir="text_embedding_models/model_topic_modeling",
#    ml_framework=aifeducation_config$get_framework())
#  
#  global_vector_clusters_modeling<-load_ai_model(
#    model_dir="text_embedding_models/model_global_vectors",
#    ml_framework=aifeducation_config$get_framework())
#  
#  bert_modeling<-load_ai_model(
#    model_dir="text_embedding_models/model_transformer_bert",
#    ml_framework=aifeducation_config$get_framework())

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  topic_modeling<-load_ai_model(
#    model_dir="text_embedding_models/topic_model_embedding_ID_DfO25E1Guuaqw7tM")
#  
#  global_vector_clusters_modeling<-load_ai_model(
#    model_dir="text_embedding_models/global_vector_clusters_embedding_ID_5Tu8HFHegIuoW14l")
#  
#  bert_modeling<-load_ai_model(
#    model_dir="text_embedding_models/bert_embedding_ID_CmyAQKtts5RdlLaS")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_classifier$set_model_description(
#  eng="This classifier targets the realization of the need for competence from
#    the self-determination theory of motivation by Deci and Ryan in lesson plans
#    and materials. It describes a learner’s need to perceive themselves as capable.
#    In this classifier, the need for competence can take on the values 0 to 2.
#    A value of 0 indicates that the learners have no space in the lesson plan to
#    perceive their own learning progress and that there is no possibility for
#    self-comparison. At level 1, competence growth is made visible implicitly,
#    e.g. by demonstrating the ability to carry out complex exercises or peer
#    control. At level 2, the increase in competence is made explicit by giving
#    each learner insights into their progress towards the competence goal. For
#    example, a comparison between the target vs. actual development towards the
#    learning objectives of the lesson can be made, or the learners receive
#    explicit feedback on their competence growth from the teacher. Self-assessment
#    is also possible. The classifier was trained using 790 lesson plans, 298
#    materials and up to 1,400 textbook tasks. Two people who received coding
#    training were involved in the coding and the inter-coder reliability for the
#    need for competence increased from a dynamic iota value of 0.615 to 0.646 over
#    two rounds of training. The Krippendorffs alpha value, on the other hand,
#    decreased from 0.516 to 0.484. The classifier is suitable for use in all
#    settings in which lesson plans and materials are to be reviewed with regard
#    to their implementation of the need for competence.",
#  native="Dieser Classifier bewertet Unterrichtsentwürfe und Lernmaterial danach,
#    ob sie das Bedürfnis nach Kompetenzerleben aus der Selbstbestimmungstheorie
#    der Motivation nach Deci und Ryan unterstützen. Das Kompetenzerleben stellt
#    das Bedürfnis dar, sich als wirksam zu erleben. Der Classifer unterteilt es
#    in drei Stufen, wobei 0 bedeutet, dass die Lernenden im Unterrichtsentwurf
#    bzw. Material keinen Raum haben, ihren eigenen Lernfortschritt wahrzunehmen
#    und auch keine Möglichkeit zum Selbstvergleich besteht. Bei einer Ausprägung
#    von 1 wird der Kompetenzzuwachs implizit, also z.B. durch die Durchführung
#    komplexer Übungen oder einer Peer-Kontrolle ermöglicht. Auf Stufe 2 wird der
#    Kompetenzzuwachs explizit aufgezeigt, indem jede:r Lernende einen objektiven
#    Einblick erhält. So kann hier bspw. ein Soll-Ist-Vergleich mit den Lernzielen
#    der Stunde erfolgen oder die Lernenden erhalten dezidiertes Feedback zu ihrem
#    Kompetenzzuwachs durch die Lehrkraft. Auch eine Selbstbewertung ist möglich.
#    Der Classifier wurde anhand von 790 Unterrichtsentwürfen, 298 Materialien und
#    bis zu 1400 Schulbuchaufgaben traniert. Es waren an der Kodierung zwei Personen
#    beteiligt, die eine Kodierschulung erhalten haben und die
#    Inter-Coder-Reliabilität für das Kompetenzerleben würde über zwei
#    Trainingsrunden von einem dynamischen Iota-Wert von 0,615 auf 0,646 gesteigert.
#    Der Krippendorffs Alpha-Wert sank hingegen von 0,516 auf 0,484. Er eignet sich
#    zum Einsatz in allen Settings, in denen Unterrichtsentwürfe und Lernmaterial
#    hinsichtlich ihrer Umsetzung des Kompetenzerlebens überprüft werden sollen.",
#  abstract_eng="This classifier targets the realization of the need for
#    competence from Deci and Ryan’s self-determination theory of motivation in l
#    esson plans and materials. It describes a learner’s need to perceive themselves
#    as capable. The variable need for competence is assessed by a scale of 0-2.
#    The classifier was developed using 790 lesson plans, 298 materials and up to
#    1,400 textbook tasks. A coding training was conducted and the inter-coder
#    reliabilities of different measures
#    (i.e. Krippendorff’s Alpha and Dynamic Iota Index) of the individual categories
#    were calculated at different points in time.",
#  abstract_native="Dieser Classifier bewertet Unterrichtsentwürfe und
#    Lernmaterial danach, ob sie das Bedürfnis nach Kompetenzerleben aus der
#    Selbstbestimmungstheorie der Motivation nach Deci & Ryan unterstützen. Das
#    Kompetenzerleben stellt das Bedürfnis dar, sich als wirksam zu erleben. Der
#    Classifer unterteilt es in drei Stufen und wurde anhand von 790
#    Unterrichtsentwürfen, 298 Materialien und bis zu 1400 Schulbuchaufgaben
#    entwickelt. Es wurden stets Kodierschulungen durchgeführt und die
#    Inter-Coder-Reliabilitäten der einzelnen Kategorien zu verschiedenen
#    Zeitpunkten berechnet.",
#  keywords_eng=c("Self-determination theory", "motivation", "lesson planning", "business didactics"),
#  keywords_native=c("Selbstbestimmungstheorie", "Motivation", "Unterrichtsplanung", "Wirtschaftsdidaktik")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_classifier$set_publication_info(
#    authors,
#    citation,
#    url=NULL)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_classifier$set_software_license("GPL-3")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  example_classifier$set_documentation_license("CC BY-SA")

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  save_ai_model(
#    model=classifier,
#    model_dir="classifiers",
#    dir_name="movie_review_classifier",
#    save_format = "default",
#    append_ID = FALSE)

## ----include = TRUE, eval=FALSE-----------------------------------------------
#  classifier<-load_ai_model(
#    model_dir="classifiers/movie_review_classifier")

