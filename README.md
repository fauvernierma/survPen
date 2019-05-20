
# survPen
## hazard and excess hazard modelling with multidimensional penalized splines
[![DOI](https://zenodo.org/badge/181266005.svg)](https://zenodo.org/badge/latestdoi/181266005)

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

## The functionalities of the *survPen* package are extensively detailed in the following vignette
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
#> To cite package 'survPen' in publications use:
#> 
#>   Mathieu Fauvernier, Laurent Roche, Laurent Remontet, Zoe Uhry
#>   and Nadine Bossard (2019). survPen: Multidimensional Penalized
#>   Splines for Survival and Net Survival Models. R package version
#>   1.1.0. https://CRAN.R-project.org/package=survPen
#> 
#> A BibTeX entry for LaTeX users is
#> 
#>   @Manual{,
#>     title = {survPen: Multidimensional Penalized Splines for Survival and Net Survival
#> Models},
#>     author = {Mathieu Fauvernier and Laurent Roche and Laurent Remontet and Zoe Uhry and Nadine Bossard},
#>     year = {2019},
#>     note = {R package version 1.1.0},
#>     url = {https://CRAN.R-project.org/package=survPen},
#>   }
#> 
#> ATTENTION: This citation information has been auto-generated from
#> the package DESCRIPTION file and may need manual editing, see
#> 'help("citation")'.
```

You may also consider citing the method article:

Fauvernier, M., Roche, L., Uhry, Z., Tron, L., Bossard, N., Remontet, L. and the CENSUR Working Survival Group (2019). Multidimensional penalized hazard model with continuous covariates: applications for studying trends and social inequalities in cancer survival, in revision in the Journal of the Royal Statistical Society, series C.





