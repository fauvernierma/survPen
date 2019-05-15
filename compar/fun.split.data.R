#---------------------------------------------------------------------
# fun.split.data

# We specify a dataframe and a list of bands of follow-up times.
# From these bands, the function creates a split version of the original dataframe
fun.split.data <- function(dat, bands)
{

# preparing data with lexis function
splitdata <- lexis(entry = 0, exit = fu, fail = dead, breaks = bands,
                         include = list(Ident,age),
                          data = dat)

# length of each time interval						  
splitdata$tik      <-  splitdata$Exit - splitdata$Entry

# 'point milieu' approximation of the cumulative hazard by taking the middle of each interval as
# the observed follow-up time
splitdata$fu <- (splitdata$Entry + splitdata$Exit) / 2

# redefining the name of the event indicator
splitdata$dead <- splitdata$Fail

# if death does occur, the final observed time must correspond to the true time of death and not
# the middle of the final interval
splitdata[splitdata$dead == 1, c("fu")] <- splitdata[splitdata$dead == 1, c("Exit")]
 
 
return(splitdata)

}
