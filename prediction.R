

# Introduction

In der R-Programmierung sind Prognosemodelle äußerst nützlich für die Vorhersage zukünftiger Ergebnisse .

Mit Hilfe von Modellen können Sie zukünftiges Verhalten auf der Grundlage von vergangenem Verhalten vorhersagen. Nachdem Sie ein Modell aufgebaut haben, verwenden Sie es, um neue Daten zu bewerten, d. h. um Vorhersagen zu treffen.

Mit R können Sie viele Arten von Modellen erstellen.


these R models:
  
  
  glm :    Generalized linear models

lda :    Linear Discriminant Analysis

knn :    K-nearest Neighbors Algorithm

rpart:   Recursive Partitioning and Regression Trees

kmeans:  k-Means clustering

rf :     Random Forest 



# Load Libraries

```{r,  warning=FALSE,include= TRUE,echo=TRUE,results='hide', message=FALSE}

library(magrittr)
library(plyr)
library(dplyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(stringr)
library(caret)
library(mvtnorm)
library(ggplot2)
library(randomForest)
library(caret)
library(pROC)
library(klaR)
library(psych)
library(MASS)
library(devtools)
library(ROCR)

```   


# Load Data 

```{r}
workPath ="D:\\DataMining\\"
cardio <- readRDS(paste (workPath , "cardio.rds", sep =""))
```


### Train test split 

Wenn wir Modelle für maschinelles Lernen erstellen, müssen wir unser Modell auf einer Teilmenge der verfügbaren Daten trainieren und die Genauigkeit des Modells auf einer Teilmenge der Daten testen. 


verwenden wir die Funktion complete.cases() in R, um (Missing values) in einem Vektor, einer Matrix oder einem Datenrahmen zu entfernen.

```{r}
cardio_complete <- cardio[(complete.cases(cardio)),]
nrow(cardio_complete)
```

Nach der Entfernung der Missing Valuses  haben wir `r toString(nrow(cardio_complete))`  Datensätze, von denen wir 1500 Datensätze für den Test verwenden werden.


```{r}
set.seed(1001)

n_test <- 1500

idx_test <- sample(1:nrow(cardio_complete), n_test)

cardio_test <- cardio_complete[ idx_test,]
cardio_train <- cardio_complete[ -idx_test,]

```


### Train first models 

Accuarcy ist ein Maßstab für die Bewertung von Klassifizierungsmodellen und 

Formal ist die accuracy wie folgt definiert:
  
  \[
    Accuracy = \frac{Number of Correct Prediction}{Total Number of Predicted}
    \]



```{r}
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"

```


The train function can be used to:
  
  * evaluate, using resampling, the effect of model tuning parameters on performance

* choose the "optimal" model across these parameters

* estimate model performance from a training set


```{r}
mod.glm <- train(target~., data=cardio_train, method = "glm",family = "binomial",metric=metric,trControl=control)
```

The Kappa statistic (or value) is a metric that compares an Observed Accuracy with an Expected Accuracy (random chance). The kappa statistic is used not only to evaluate a single classifier, but also to evaluate classifiers amongst themselves. In addition, it takes into account random chance (agreement with a random classifier), which generally means it is less misleading than simply using accuracy as a metric (an Observed Accuracy of 80% is a lot less impressive with an Expected Accuracy of 75% versus an Expected Accuracy of 50%). Computation of Observed Accuracy and Expected Accuracy is integral to comprehension of the kappa statistic, and is most easily illustrated through use of a confusion matrix

```{r}
mod.lda <- train(target~., data=cardio_train, method="lda",metric=metric,trControl=control)
mod.lda$results
```


```{r}
mod.cart <- train(target~., data=cardio_train, method="rpart",metric=metric,trControl=control)
mod.cart$results
```


```{r}
mod.knn <- train(target~., data=cardio_train, method="knn", metric=metric, trControl=control)
mod.knn$results
```


```{r}
mod.svm.radial <- train(target~., data=cardio_train, method="svmRadial", metric=metric, trControl=control)

mod.svm.radial$results
```


```{r}
mod.svm.linear <- train(target~., data=cardio_train, method="svmLinear", metric=metric, trControl=control)
mod.svm.linear$results
```


```{r}
mod.rf <- train(target~., data=cardio_train, method="rf", metric=metric, trControl=control)
mod.rf$results
```

# summarize all models 
```{r}

results <- resamples(list(glm = mod.glm , lda= mod.lda, cart=mod.cart, knn=mod.knn, svm.radial=mod.svm.radial, svm.linear=mod.svm.linear, rf=mod.rf))
summary(results)
saveRDS(results,file  =paste (workPath , "results.rds", sep =""))

```

# plot accuracies
```{r}
dotplot(results)

```




### Analyse der ersten Modelle 

AUC (Area Under The Curve) Die ROC-Kurve (Receiver Operating Characteristics) ist eines der wichtigsten Bewertungsmaße für die Beurteilung der Leistung von binären Klassifikationsproblemen.
ROAC ist eine Wahrscheinlichkeitskurve, die die TPR (True Positive Rate) gegen die FPR (False Positive Rate) aufträgt. AUC ist das Maß für die Trennbarkeit und zeigt an, wie gut unser Modell in der Lage ist, zwischen den Klassen zu unterscheiden.
Die AUC gibt an, wie gut das Modell zwischen positiven und negativen Klassen unterscheidet. Je größer der AUC, desto besser.



```{r}
computeData <- F
models <- c("rf", "lda", "rpart", "knn", "glm")
workPath = "D:\\DataMining\\Seafile\\01_cardio\\Projekt\\"

if (computeData) {
  performanceDf <- data.frame(model = models)
  performanceDf$accuracy <- NA
  performanceDf$sensitivity <- NA
  performanceDf$specificity <- NA
  performanceDf$auc <- NA
  for (ir in 1:nrow (performanceDf)) {
    mod <-
      train(
        target ~ .,
        data = cardio_train,
        method = performanceDf$model[ir]
        ,
        metric = metric,
        trControl = control
      )
    
    y_pred_prob <- predict(mod ,  cardio_test, type = "prob")
    
    y_pred <- predict(mod ,  cardio_test)
    
    table(y_pred, cardio_test$target)
    
    accuracy = table(y_pred == cardio_test$target)["TRUE"] / length (y_pred)
    
    performanceDf$accuracy[ir] = accuracy
    
    sensitivity   = sensitivity(factor(y_pred), factor(cardio_test$target))
    
    performanceDf$sensitivity[ir] = sensitivity
    
    specificity = specificity(factor(y_pred), factor(cardio_test$target))
    performanceDf$specificity[ir] = specificity
    
    auc <- roc(cardio_test$target , y_pred_prob$CHD)$auc
    performanceDf$auc[ir] = auc
    
  }
  saveRDS(performanceDf, file  = paste (workPath , "performanceDf.rds", sep =
                                          ""))
}
performanceDf <-
  readRDS(paste (workPath , "performanceDf.rds", sep = ""))
performanceDf

```


```{r}
ggplot(performanceDf, aes(x =performanceDf$model , y = auc )) +
  geom_point(colour = "red",size = 3) 
```


Die Modelle Linear Discriminant Analysis (lda) und generalization of ordinary linear regression (glm )zeigten das beste Ergebnis  mit einem auc von 0.7011527 und 0.7068537.

as Modell k nearest neighbor hat die schlechteste Performance mit einem auc von 0,3858732.
