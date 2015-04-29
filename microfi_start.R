## microfinance network
## data from BANERJEE, CHANDRASEKHAR, DUFLO, JACKSON 2012

## data on 8622 households
hh <- read.csv("microfi_households.csv", row.names="hh")
hh$village <- factor(hh$village)

## We'll kick off with a bunch of network stuff.
## This will be covered in more detail in lecture 6.
## get igraph off of CRAN if you don't have it
## install.packages("igraph")
## this is a tool for network analysis
## (see http://igraph.sourceforge.net/)
library(igraph)
edges <- read.table("microfi_edges.txt", colClasses="character")
## edges holds connections between the household ids
hhnet <- graph.edgelist(as.matrix(edges))
hhnet <- as.undirected(hhnet) # two-way connections.

## igraph is all about plotting.
V(hhnet) ## our 8000+ household vertices
## Each vertex (node) has some attributes, and we can add more.
V(hhnet)$village <- as.character(hh[V(hhnet),'village'])
## we'll color them by village membership
vilcol <- rainbow(nlevels(hh$village))
names(vilcol) <- levels(hh$village)
V(hhnet)$color = vilcol[V(hhnet)$village]
## drop HH labels from plot
V(hhnet)$label=NA

# graph plots try to force distances proportional to connectivity
# imagine nodes connected by elastic bands that you are pulling apart
# The graphs can take a very long time, but I've found
# edge.curved=FALSE speeds things up a lot.  Not sure why.

## we'll use induced.subgraph and plot a couple villages
village1 <- induced.subgraph(hhnet, v=which(V(hhnet)$village=="1"))
village33 <- induced.subgraph(hhnet, v=which(V(hhnet)$village=="33"))

# vertex.size=3 is small.  default is 15
plot(village1, vertex.size=3, edge.curved=FALSE)
plot(village33, vertex.size=3, edge.curved=FALSE)

######  now, on to your homework stuff

library(gamlr)

## match id's; I call these 'zebras' because they are like crosswalks
zebra <- match(rownames(hh), V(hhnet)$name)

## calculate the `degree' of each hh:
##  number of commerce/friend/family connections
degree <- degree(hhnet)[zebra]
names(degree) <- rownames(hh)
degree[is.na(degree)] <- 0 # unconnected houses, not in our graph

## if you run a full glm, it takes forever and is an overfit mess
# > summary(full <- glm(loan ~ degree + .^2, data=hh, family="binomial"))
# Warning messages:
# 1: glm.fit: algorithm did not converge
# 2: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Q1 - Log Transformation of degree
# We add the '+1' term so that the log is defined when degree = 0
d <- log(degree+1)

## Q2 - Use a Lasso Regression to fit dhat based on x
# Create a matrix to hold the controls
# First need to convert beds, rooms, and leader to factors
hh$rooms <- factor(hh$rooms)
hh$beds <- factor(hh$beds)
hh$electricity <- factor(hh$electricity)
hh$leader <- factor(hh$leader)

# Now put all of the controls in a single data frame
controls <- data.frame(hh[,c(2:9)])

# And re level everything to add NA
controls$village <- factor(controls$village,levels=c(NA,levels(controls$village)), exclude=NULL)
controls$religion <- factor(controls$religion,levels=c(NA,levels(controls$religion)), exclude=NULL)
controls$roof <- factor(controls$roof,levels=c(NA,levels(controls$roof)), exclude=NULL)
controls$rooms <- factor(controls$rooms,levels=c(NA,levels(controls$rooms)), exclude=NULL)
controls$beds <- factor(controls$beds,levels=c(NA,levels(controls$beds)), exclude=NULL)
controls$electricity <- factor(controls$electricity,levels=c(NA,levels(controls$electricity)), exclude=NULL)
controls$ownership <- factor(controls$ownership,levels=c(NA,levels(controls$ownership)), exclude=NULL)
controls$leader <- factor(controls$leader,levels=c(NA,levels(controls$leader)), exclude=NULL)

# Convert the data frame into a sparse model matrix
controls.smm <- sparse.model.matrix(~., data=controls)[,-1]

# And the y variable into a separate vector
y <- hh[,1]

# Now we can fit the controls to d
treat <- gamlr(controls.smm,d)

# Predict dhat from the fitted betas
dhat <- predict(treat, controls.smm, type="response")

# And plot to see the relationship between d and dhat
plot(dhat,d,bty="n",pch=21,bg=8)

# There does not graphically appear to be a strong relationship. But check R^2
cor(drop(dhat),d)^2
# ~> [1] 0.08478302
# R^2 is low which is good for our sake as the controls do not help predict the treatment effect

## Q3 - Re run with dhat estimator effect to find casual effect
causal <- gamlr(cBind(d,dhat,controls.smm),y,free=2,family="binomial")
# coef(causal)["d",]
# ~> [1] 0.1409793
# exp(coef(causal)["d",])
# ~> [1] 1.151401
# Interpreting gamma.hat we would say that a 1% increase in connectedness increases the odds
# of having a microloan by 15%. This is a non-trivial impact and appears to confirm the thesis
# that connectivity is structurally connected with the propensity to get a microfinance loan.

## Q4 - Compute the gamma.hat for a naive lasso
naive <- gamlr(cBind(d,controls.smm),y,family="binomial")
# coef(naive)["d",]
# ~> [1] 0.1413462
# exp(coef(naive)["d",])
# ~> [1] 1.151823
# So the naive regression estimates nearly the exact same impact of connectedness. This is an
# interesting result but perhaps shouldn't be unexpected because we found that the controls
# are not a good predictor of d (R^2 ~= .08).

## Q5 - Bootstrap the estimator from Q3 and describe the uncertainty
n <- nrow(controls.smm)

## Bootstrapping our lasso causal estimator is easy
gamb <- c() # empty gamma
for(b in 1:20){
  ## create a matrix of resampled indices
  ib <- sample(1:n, n, replace=TRUE)
  ## create the resampled data
  controls.b <- controls.smm[ib,]
  db <- d[ib]
  yb <- y[ib]
  ## run the treatment regression
  treatb <- gamlr(controls.b,db,lambda.min.ratio=1e-3)
  dhatb <- predict(treatb, controls.b, type="response")

  fitb <- gamlr(cBind(db,dhatb,controls.b),yb,free=2,family="binomial")
  gamb <- c(gamb,coef(fitb)["db",])
  print(b)
}
## not very exciting though: all zeros
summary(gamb)
