
# survPen
## hazard and excess hazard modelling with multidimensional penalized splines
[![badge](https://img.shields.io/badge/Launch-getTBinR-blue.svg)](https://mybinder.org/v2/gh/fauvernierma/survPen/master?urlpath=rstudio)
[![Travis-CI Build Status](https://travis-ci.org/fauvernierma/survPen.svg?branch=master)](https://travis-ci.org/fauvernierma/survPen)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/fauvernierma/survPen?branch=master&svg=true)](https://ci.appveyor.com/project/fauvernierma/survpen)
[![Coverage Status](https://img.shields.io/codecov/c/github/fauvernierma/survPen/master.svg)](https://codecov.io/github/fauvernierma/survPen?branch=master)
[![DOI](https://zenodo.org/badge/181266005.svg)](https://zenodo.org/badge/latestdoi/181266005)


In survival and net survival analysis, in addition to modelling the effect of time (via the baseline hazard), 
one has often to deal with several continuous covariates and model their functional forms, their time-dependent 
effects, and their interactions. Model specification becomes therefore a complex problem and penalized regression 
splines represent an appealing solution to that problem as splines offer the required 
flexibility while penalization limits overfitting issues. 

Current implementations of penalized survival models can be slow or unstable and sometimes lack some key features 
like taking into account expected mortality to provide net survival and excess hazard estimates. In contrast, 
survPen provides an automated, fast, and stable implementation 
(thanks to explicit calculation of the derivatives of the likelihood) and offers a unified framework for 
multidimensional penalized hazard and excess hazard models.

`survPen` may be of interest to those who 1) analyse any kind of time-to-event data: mortality, disease relapse, 
machinery breakdown, unemployment, etc 2) wish to describe the associated hazard and to understand which predictors 
impact its dynamics.


You can install this R package from GitHub:


```r
install.packages("devtools")
library(devtools)
install_github("fauvernierma/survPen")
```

or directly from the CRAN repository:


```r
install.packages("survPen")
```

and then use it by typing


```r
library(survPen)
```

## The functionalities of the `survPen` package are extensively detailed in the following vignette
Available from
[GitHub](https://htmlpreview.github.io/?https://github.com/fauvernierma/survPen/blob/master/inst/doc/survival_analysis_with_survPen.html)
or from [CRAN](https://cran.r-project.org/web/packages/survPen/vignettes/survival_analysis_with_survPen.html)

## Contributing

File an issue [here](https://github.com/fauvernierma/survPen/issues) if there is a feature, or a dataset, that you think is missing from the package, or better yet submit a pull request!

Please note that the `survPen` project is released with a [Contributor Code of Conduct](.github/CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## Citing 

If using `survPen` please consider citing the package in the relevant work. Citation information can be generated in R using the following (after installing the package),


```r
citation("survPen")
#> 
#> To cite survPen in publications use:
#> 
#>   Mathieu Fauvernier, Laurent Remontet, Zoe Uhry, Nadine Bossard,
#>   Laurent Roche and the CENSUR working survival group (2019).
#>   survPen: an R package for hazard and excess hazard modelling
#>   with multidimensional penalized splines in revision in the
#>   Journal of Open Source Software
#> 
#> A BibTeX entry for LaTeX users is
#> 
#>   @Article{,
#>     title = {survPen: an R package for hazard and excess hazard modelling with multidimensional penalized splines},
#>     author = {Mathieu Fauvernier and Laurent Remontet and Zoe Uhry and Nadine Bossard and Laurent Roche and the CENSUR working survival group},
#>     journal = {in revision in the Journal of Open Source Software},
#>     year = {2019},
#>   }
```

You may also consider citing the method article:
Fauvernier, M., Roche, L., Uhry, Z., Tron, L., Bossard, N., Remontet, L. and the CENSUR Working Survival Group (in press). Multi-dimensional penalized hazard model with continuous covariates: applications for studying trends and social inequalities in cancer survival, 
Journal of the Royal Statistical Society, series C. doi: 10.1111/rssc.12368









