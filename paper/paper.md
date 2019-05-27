---
title: '``survPen``: an **R** package for hazard and excess hazard modelling with multidimensional penalized splines'
tags:
  - R
  - survival
  - time-to-event
  - excess hazard
  - penalized splines
authors:
  - name: Mathieu Fauvernier
    orcid: 0000-0002-3809-8248
    affiliation: 1, 2
  - name: Laurent Remontet
    affiliation: 1, 2
  - name: Zoé Uhry
    affiliation: 3, 1, 2
  - name: Nadine Bossard
    affiliation: 1, 2
  - name: Laurent Roche
    affiliation: 1, 2
affiliations:
 - name: Hospices Civils de Lyon, Pôle Santé Publique, Service de Biostatistique - Bioinformatique, Lyon, France
   index: 1
 - name: Université de Lyon; Université Lyon 1; CNRS; UMR 5558, Laboratoire de Biométrie et Biologie Évolutive, Équipe Biostatistique-Santé, Villeurbanne, France
   index: 2
 - name: Département des Maladies Non-Transmissibles et des Traumatismes, Santé Publique France, Saint-Maurice, France
   index: 3
date: 12 April 2019
bibliography: paper.bib
---

# Background


In survival and net survival analysis, in addition to modelling the effect of time (via the baseline hazard), 
one has often to deal with several continuous covariates and model their functional forms, their time-dependent 
effects, and their interactions. Model specification becomes therefore a complex problem and penalized regression 
splines [@Ruppert2003 ; @Wood2017] represent an appealing solution to that problem as splines offer the required 
flexibility while penalization limits overfitting issues. 

Current implementations of penalized survival models can be slow or unstable and sometimes lack some key features 
like taking into account expected mortality to provide net survival and excess hazard estimates. In contrast, based 
on the framework and algorithms developed by Wood, survPen provides an automated, fast, and stable implementation 
(thanks to explicit calculation of the derivatives of the likelihood) and offers a unified framework for 
multidimensional penalized hazard and excess hazard models.

In epidemiology, as patients may die from their disease or from other causes, it is relevant to study the mortality 
due to their disease; also called “excess mortality”. This excess mortality is useful to make comparisons between 
different countries and time periods [@Uhry2017 ; @Allemani2018] and is directly linked to the concept 
of net survival which is another important indicator in epidemiology [@Perme2012]. 


# Summary

``survPen`` is an implementation of multidimensional penalized hazard and excess hazard models 
for time-to-event data in **R** [@R]. It implements the method detailed in 
[@Fauvernier2019] which is itself included in the framework for general smooth models 
proposed by [@Wood2016].
Other **R** packages propose to fit flexible survival models via penalized regression splines 
(``rstpm2``, ``bamlss``, ``R2BayesX``, etc). However, the way they estimate the smoothing parameters is not optimal
as they rely on either derivative-free optimization (``rstpm2``) or MCMC (``bamlss``, ``R2BayesX``), leading to possibly
unstable or time-consuming analyses.
The main objective of the ``survPen`` package is to offer a fully automatic, fast, stable and convergent 
procedure in order to model simultaneously non-proportional, non-linear effects of covariates and 
interactions between them. A second objective is to extend the approach to excess hazard modelling 
[@Esteve1990 ; @Remontet2007].
``survPen`` is a free and open-source **R** package, available via GitHub at https://github.com/fauvernierma/survPen or via
the CRAN repository at https://CRAN.R-project.org/package=survPen. 
The major features of survPen are documented in a walkthrough vignette that is included with the package 
https://htmlpreview.github.io/?https://github.com/fauvernierma/survPen/blob/master/inst/doc/survival_analysis_with_survPen.html

Those features include:

 - Univariate penalized splines for the baseline hazard as well as any other continuous covariate.
 - Penalized tensor product splines for time-dependent effects and interactions between several 
 continuous covariates.
 - Interactions between penalized splines and unpenalized continuous or categorical variables.
 - Automatic smoothing parameter estimation by either optimizing the Laplace approximate marginal 
 likelihood (LAML, [@Wood2016]) or likelihood cross-validation criterion (LCV, [@OSullivan1988]).
 - Excess hazard modelling by specifying expected mortality rates.

``survPen`` may be of interest to those who 1) analyse any kind of time-to-event data: mortality, disease relapse, 
machinery breakdown, unemployment, etc 2) wish to describe the associated hazard and to understand which predictors 
impact its dynamics.

Using the ``survPen`` package for time-to-event data analyses will help choose the appropriate degree of 
complexity in survival and net survival contexts while simplifying the model building process.


Multidimensional splines with ``survPen`` are currently being used in three major ongoing projects:

 - Modelling the effects of time since diagnosis, age at diagnosis and year of diagnosis on the mortality 
 due to cancer using French cancer registries data (FRANCIM network, around 1,200,000 tumours diagnosed 
 between 1989 and 2015). This study will provide the new national estimates of cancer survival in France and 
 its results will be used in the evaluation of the French “Plan Cancer” at the end of 2019.
 - Modelling the effect of the European Deprivation Index (EDI) on the mortality due to cancer in France, using 
 data from the FRANCIM network; this is the first time that EDI is available in all FRANCIM registries 
 (around 210,000 tumours, diagnosed between 2006 and 2009 in 18 registries).
 - For the first time modelling the effects of time since onset, age at onset, current age, year of onset and 
 sex on the mortality due to multiple sclerosis in the biggest cohort of multiple sclerosis patients in France 
 (37,524 patients diagnosed over the period 1960-2014 in 18 OFSEP centres).



# Acknowledgements
This research was conducted as part of the first author’s PhD thesis supported by the French Ministère 
de lʼEnseignement supérieur, de la Recherche et de lʼInnovation. The authors thank the ANR (Agence 
Nationale de la Recherche) for supporting this study of the CENSUR group (ANR grant number ANR-12-BSV1-0028). 
This research was also carried out within the context of a four-institute cancer surveillance program 
partnership involving the Institut National du Cancer (INCa), Santé Publique France (SPF), the French 
network of cancer registries (FRANCIM), and Hospices Civils de Lyon (HCL) through a grant from INCa 
(attributive decision N° 2016-131). The authors are grateful Jacques Estève for his valuable advice.


# References















