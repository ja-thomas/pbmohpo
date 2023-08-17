setwd("mlruns/266560418568116824/")
setwd("mlruns/168232482512585886/")

library(data.table)
library(mlr3misc)

library(ggplot2)

runs = setdiff(dir(), "meta.yaml")

results = map_dtr(runs, function(run) {
  optimizer = readLines(paste0(run, "/params/OPTIMIZER_TYPE"), n = -1)
  if (optimizer %in% c("BO", "RS")) {
    seed = readLines(paste0(run, "/params/SEED"), n = -1)
    auc = strsplit(readLines(paste0(run, "/metrics/auc")), " ")[[1]][2]
    nf = strsplit(readLines(paste0(run, "/metrics/nf")), " ")[[1]][2]
    utility = strsplit(readLines(paste0(run, "/metrics/utility")), " ")[[1]][2]
    data.table(optimizer = optimizer, seed = seed, auc = as.numeric(auc), nf = as.numeric(nf), utility = as.numeric(utility))
  } else {
    data.table()
  }
})

ggplot(aes(x = seed, y = utility, colour = optimizer), data = results) +
  geom_boxplot()

setorderv(results, col = "seed")
bo_results = results[optimizer == "BO"]
rs_results = results[optimizer == "RS"]
diff_results = bo_results
diff_results$diff_utility = bo_results$utility - rs_results$utility

ggplot(aes(x = seed, y = diff_utility), data = diff_results) +
  geom_boxplot()

agg = results[, .(mean_utility = mean(utility)), by = .(optimizer, seed)]

ggplot(aes(x = optimizer, y = mean_utility), data = agg) +
  geom_boxplot()
