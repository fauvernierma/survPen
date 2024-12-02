 
 context("RelativeMortalityRatio")

data(datCancer)
data(expected.table)

#-------------------- creating split dataset for multiplicative model

splitdat <- splitmult(datCancer, cut = (1:5), end = "fu", 
event = "dead")
			
#-------------------- merging with expected mortality table

# deriving current age and year (closest whole number)
splitdat$age_current <- floor(splitdat$age + splitdat$fu + 0.5)

splitdat$year_current <- floor(splitdat$yod + splitdat$fu + 0.5)

splitdat <- merge(splitdat, expected.table, 
                 by.x=c("age_current","year_current"), by.y=c("Age","Year"),all.x=T)

# fitting model

f1 <- ~age

# In terms of Gauss-Legendre quadrature, as the follow-up is split into up to 5 parts, the cumulative hazard approximation
# requires less nodes than an approximation on the whole range of definition. If not supplied, the default number of nodes
# for a relative mortality ratio model will be 10 (as opposed to 20 for an excess hazard model) 
mod.ratio <- survPen(f1,data=splitdat,t1=fu,event=dead,expected=mx,method="LAML",type="mult")

# predictions of the model
new.age <- 50
pred.ratio <- predict(mod.ratio,data.frame(age=50,fu=5))

test_that("Relative mortality ratio prediction ok", {
  expect_true(abs(pred.ratio$ratio - 7.6626182413580004038) < 1e-10)
})

test_that("Relative mortality ratio standard error ok", {
  expect_true(abs(summary(mod.ratio)$coefficients[2,2]
 -  0.0021372585539319003019) < 1e-10)
})
   
  


