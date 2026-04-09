setwd("C:/Users/075be/Downloads/thesis")
pkgs <- c("caret","pROC","ggplot2","dplyr","corrplot","randomForest",
          "e1071","xgboost","glmnet","gridExtra","RColorBrewer",
          "scales","reshape2","themis","Boruta","doParallel","recipes")

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, dependencies = TRUE)
}

suppressPackageStartupMessages({
  library(caret); library(pROC); library(ggplot2); library(dplyr)
  library(corrplot); library(randomForest); library(e1071)
  library(xgboost); library(glmnet); library(gridExtra)
  library(RColorBrewer); library(scales); library(reshape2)
  library(themis); library(Boruta); library(doParallel); library(recipes)
})

cl <- makeCluster(max(1, detectCores() - 1))
registerDoParallel(cl)
set.seed(42)

# 1. Load & Prepare Data
setwd("C:/Users/075be/Downloads/thesis")

df <- read.csv("gait_features_rich.csv", stringsAsFactors = FALSE)
View(df)
df <- df[, !(colnames(df) %in% "Subject")]
df$Class <- factor(ifelse(df$Class == "ASD", "ASD", "NonASD"),
                   levels = c("NonASD", "ASD"))
feat_cols <- setdiff(colnames(df), "Class")

cat("Dimensions:", dim(df), "\n")
cat("Class distribution:\n"); print(table(df$Class)); cat("\n")

# 2. Remove NZV & Highly Correlated Features 
nzv_idx <- nearZeroVar(df[, feat_cols])
if (length(nzv_idx) > 0) {
  df        <- df[, !(colnames(df) %in% feat_cols[nzv_idx])]
  feat_cols <- setdiff(colnames(df), "Class")
  cat("Removed", length(nzv_idx), "near-zero-variance features.\n")
}

cor_mat  <- cor(df[, feat_cols], use = "pairwise.complete.obs")
high_cor <- findCorrelation(cor_mat, cutoff = 0.95, verbose = FALSE)
if (length(high_cor) > 0) {
  df        <- df[, -which(colnames(df) %in% feat_cols[high_cor])]
  feat_cols <- setdiff(colnames(df), "Class")
  cat("Removed", length(high_cor), "highly correlated features.\n")
}
cat("Features remaining:", length(feat_cols), "\n\n")

################################

# test_res <- do.call(rbind, lapply(feat_cols, function(col) {
#   sw_p <- tryCatch(shapiro.test(df[[col]])$p.value, error = function(e) NA)
#   
#   t_res <- t.test(df[[col]] ~ df$Class, var.equal = FALSE)   # Welch
#   w_res <- wilcox.test(df[[col]] ~ df$Class, exact = FALSE)
#   
#   chosen_test <- if (!is.na(sw_p) && sw_p > 0.05) "Welch_t" else "Wilcoxon"
#   chosen_p <- if (!is.na(sw_p) && sw_p > 0.05) t_res$p.value else w_res$p.value
#   
#   data.frame(
#     Feature = col,
#     Shapiro_p = round(sw_p, 4),
#     Chosen_Test = chosen_test,
#     Chosen_p = round(chosen_p, 4),
#     Welch_p = round(t_res$p.value, 4),
#     Wilcoxon_p = round(w_res$p.value, 4),
#     Sig = ifelse(chosen_p < 0.05, "***", "ns"),
#     stringsAsFactors = FALSE
#   )
# }))
# 
# print(test_res)
#################################

# 3. Group Difference Tests
test_res <- do.call(rbind, lapply(feat_cols, function(col) {
  sw_p <- tryCatch(shapiro.test(df[[col]])$p.value, error = function(e) NA)
  if (!is.na(sw_p) && sw_p > 0.05) {
    r <- t.test(df[[col]] ~ df$Class)
    data.frame(Feature = col, Test = "t-test",
               p_value = round(r$p.value, 4),
               Sig = ifelse(r$p.value < 0.05, "***", "ns"),
               stringsAsFactors = FALSE)
  } else {
    r <- wilcox.test(df[[col]] ~ df$Class, exact = FALSE)
    data.frame(Feature = col, Test = "Wilcoxon",
               p_value = round(r$p.value, 4),
               Sig = ifelse(r$p.value < 0.05, "***", "ns"),
               stringsAsFactors = FALSE)
  }
}))

sig_feats <- test_res$Feature[test_res$Sig == "***"]
cat(sprintf("Significant features (p<0.05): %d / %d\n\n",
            length(sig_feats), length(feat_cols)))
print(test_res[test_res$Sig == "***", ])

# 4. Boxplots of Significant Features 
if (length(sig_feats) >= 2) {
  plot_feats <- sig_feats[1:min(9, length(sig_feats))]
  bp_list <- lapply(plot_feats, function(f) {
    ggplot(df, aes(x = Class, y = .data[[f]], fill = Class)) +
      geom_boxplot(alpha = 0.85, outlier.shape = 21) +
      scale_fill_manual(values = c("NonASD" = "#4393C3", "ASD" = "#D6604D")) +
      labs(title = f, x = NULL, y = NULL) +
      theme_minimal(base_size = 10) +
      theme(legend.position = "none")
  })
  png("plot_sig_features_boxplot.png", width = 1100, height = 900, res = 100)
  grid.arrange(grobs = bp_list, ncol = 3,
               top = "Significant Features: ASD vs Non-ASD")
  dev.off()
  cat("Boxplot saved.\n\n")
}

# 5. Boruta Feature Selection
set.seed(42)
boruta_res <- tryCatch(
  Boruta(Class ~ ., data = df, doTrace = 0, maxRuns = 200),
  error = function(e) NULL
)
selected <- if (!is.null(boruta_res)) {
  getSelectedAttributes(TentativeRoughFix(boruta_res))
} else character(0)

if (length(selected) == 0) {
  cat("Boruta found no features.\n")
  selected <- if (length(sig_feats) > 0) sig_feats else feat_cols
}
cat("Selected features (", length(selected), "):",
    paste(selected, collapse = ", "), "\n\n")
plot(boruta_res,las=2,cex.axis=0.7)
df_sel    <- df[, c(selected, "Class")]
sel_feats <- selected

# ── 6. Train / Test Split
split_idx  <- createDataPartition(df_sel$Class, p = 0.70, list = FALSE)
train_raw  <- df_sel[ split_idx, ]
test_raw   <- df_sel[-split_idx, ]
cat(sprintf("Train: %d  |  Test: %d\n\n", nrow(train_raw), nrow(test_raw)))

# ── 7. Preprocessing + SMOTE (train only) 
pre_proc     <- preProcess(train_raw[, sel_feats], method = c("center", "scale"))
train_scaled <- cbind(predict(pre_proc, train_raw[, sel_feats]),
                      Class = train_raw$Class)
test_scaled  <- cbind(predict(pre_proc, test_raw[, sel_feats]),
                      Class = test_raw$Class)

train_balanced <- recipe(Class ~ ., data = train_scaled) |>
  themis::step_smote(Class, over_ratio = 1, seed = 42) |>
  prep() |> bake(new_data = NULL)

cat("After SMOTE:"); print(table(train_balanced$Class)); cat("\n")
n_train <- nrow(train_balanced)

#  8. CV Control 
ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel   = TRUE
)

# 9. Evaluation Helper 
eval_model <- function(model, test_data, name) {
  probs <- predict(model, newdata = test_data, type = "prob")[, "ASD"]
  preds <- predict(model, newdata = test_data)
  cm    <- confusionMatrix(preds, test_data$Class, positive = "ASD")
  roc_o <- roc(as.numeric(test_data$Class == "ASD"), probs, quiet = TRUE)
  auc_v <- as.numeric(auc(roc_o))
  
  cat(sprintf(
    "\n══════════════════════════════════════════\n  %s\n══════════════════════════════════════════\n",
    name))
  print(cm$table)
  cat(sprintf(
    "  Accuracy   : %.4f\n  Sensitivity: %.4f\n  Specificity: %.4f\n  Precision  : %.4f\n  F1 Score   : %.4f\n  AUC        : %.4f\n",
    cm$overall["Accuracy"], cm$byClass["Sensitivity"],
    cm$byClass["Specificity"], cm$byClass["Precision"],
    cm$byClass["F1"], auc_v))
  
  list(name        = name,      model  = model,  cm     = cm,
       roc         = roc_o,     auc    = auc_v,
       accuracy    = as.numeric(cm$overall["Accuracy"]),
       sensitivity = as.numeric(cm$byClass["Sensitivity"]),
       specificity = as.numeric(cm$byClass["Specificity"]),
       precision   = as.numeric(cm$byClass["Precision"]),
       f1          = as.numeric(cm$byClass["F1"]))
}

results_list <- list()   # collect results safely

# 10. Logistic Regression (Elastic Net) 
cat("\n>>> Logistic Regression (Elastic Net) ...\n")
lr_grid <- expand.grid(
  alpha  = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10^seq(-4, 1, length = 40)
)
lr_model <- train(Class ~ ., data = train_balanced,
                  method = "glmnet", trControl = ctrl,
                  tuneGrid = lr_grid, metric = "ROC")
cat("Best LR — alpha:", lr_model$bestTune$alpha,
    "  lambda:", round(lr_model$bestTune$lambda, 5), "\n")
results_list[["LR"]] <- eval_model(lr_model, test_scaled,
                                   "Logistic Regression (Elastic Net)")

# 11. Random Forest 
cat("\n>>> Random Forest ...\n")
rf_grid <- data.frame(mtry = unique(c(2, 3, 4,
                                      round(sqrt(length(sel_feats))),
                                      round(length(sel_feats) / 2))))
rf_model <- train(Class ~ ., data = train_balanced,
                  method = "rf", trControl = ctrl,
                  tuneGrid = rf_grid, ntree = 1000,
                  importance = TRUE, metric = "ROC")
cat("Best RF mtry:", rf_model$bestTune$mtry, "\n")
results_list[["RF"]] <- eval_model(rf_model, test_scaled, "Random Forest")

png("plot_rf_importance.png", width = 800, height = 600, res = 100)
varImpPlot(rf_model$finalModel,
           main = "Random Forest – Variable Importance", pch = 19)
dev.off()

# 12. SVM (RBF) 
cat("\n>>> SVM (RBF) ...\n")
svm_grid <- expand.grid(
  C     = c(0.01, 0.1, 0.5, 1, 5, 10, 50, 100),
  sigma = c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1)
)
svm_model <- train(Class ~ ., data = train_balanced,
                   method = "svmRadial", trControl = ctrl,
                   tuneGrid = svm_grid, metric = "ROC")
cat("Best SVM — C:", svm_model$bestTune$C,
    "  sigma:", svm_model$bestTune$sigma, "\n")
results_list[["SVM"]] <- eval_model(svm_model, test_scaled, "SVM (RBF Kernel)")

# 13. XGBoost
cat("\n>>> XGBoost ...\n")
xgb_grid <- expand.grid(
  nrounds          = c(25, 50, 100),          # fewer rounds for small n
  max_depth        = c(2, 3),                 # shallow trees
  eta              = c(0.05, 0.1, 0.3),
  gamma            = c(0, 0.1),
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(1, 3),
  subsample        = c(0.8, 1.0)
)
cat("XGBoost grid size:", nrow(xgb_grid), "combinations\n")

xgb_model <- tryCatch(
  train(Class ~ ., data = train_balanced,
        method    = "xgbTree",
        trControl = ctrl,
        tuneGrid  = xgb_grid,
        metric    = "ROC",
        verbosity = 0),
  error = function(e) {
    cat("[WARN] XGBoost failed:", conditionMessage(e), "\n")
    NULL
  }
)

if (!is.null(xgb_model)) {
  cat("Best XGB:\n"); print(xgb_model$bestTune)
  results_list[["XGB"]] <- eval_model(xgb_model, test_scaled, "XGBoost")
} else {
  cat("[INFO] XGBoost skipped — it will not appear in comparison.\n")
}

#  14. Stop Cluster 
stopCluster(cl); registerDoSEQ()

# 15. Comparison Table
comparison <- do.call(rbind, lapply(results_list, function(r) {
  data.frame(
    Model       = r$name,
    Accuracy    = round(r$accuracy,    4),
    Sensitivity = round(r$sensitivity, 4),
    Specificity = round(r$specificity, 4),
    Precision   = round(r$precision,   4),
    F1          = round(r$f1,          4),
    AUC         = round(r$auc,         4),
    stringsAsFactors = FALSE
  )
}))
rownames(comparison) <- NULL


cat("              MODEL COMPARISON SUMMARY\n")

print(comparison)



#############################################################################


best <- comparison[which.max(comparison$AUC), ]
cat(sprintf("\n★  Best Model: %s  |  AUC=%.4f  Acc=%.4f  F1=%.4f\n\n",
            best$Model, best$AUC, best$Accuracy, best$F1))

write.csv(comparison, "model_comparison_results.csv", row.names = FALSE)

# ── 16. ROC Overlay ───────────────────────────────────────────────────────────
pal  <- c("#1B7837", "#2166AC", "#D6604D", "#762A83",
          "#E08214", "#4D9221")[seq_along(results_list)]
labs <- sapply(results_list, `[[`, "name")
aucs <- sapply(results_list, `[[`, "auc")

png("plot_roc_all_models.png", width = 700, height = 650, res = 110)
plot(results_list[[1]]$roc, col = pal[1], lwd = 2.5,
     main = "ROC Curves – All Models", legacy.axes = TRUE)
if (length(results_list) > 1) {
  for (i in seq(2, length(results_list))) {
    plot(results_list[[i]]$roc, col = pal[i], lwd = 2.5, add = TRUE)
  }
}
abline(a = 0, b = 1, lty = 2, col = "grey55")
legend("bottomright", bty = "n",
       legend = sprintf("%s  (AUC=%.3f)", labs, aucs),
       col = pal, lwd = 2.5, cex = 0.85)
dev.off()

# ── 17. Performance Bar Chart ─────────────────────────────────────────────────
comp_long <- reshape2::melt(
  comparison[, c("Model", "Accuracy", "Sensitivity",
                 "Specificity", "F1", "AUC")],
  id.vars = "Model", variable.name = "Metric", value.name = "Value"
)
p_bar <- ggplot(comp_long,
                aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = position_dodge(0.8), width = 0.72, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.2f", Value)),
            position = position_dodge(0.8),
            vjust = -0.45, size = 2.9) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1.12),
                     labels = percent_format(accuracy = 1)) +
  labs(title = "Model Performance Comparison", x = NULL, y = "Score") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave("plot_model_comparison_bar.png", p_bar,
       width = 10, height = 5.5, dpi = 150)

# ── 18. CV Resampling Dotplot ─────────────────────────────────────────────────
model_objs <- list(
  LR_ElasticNet = lr_model,
  RandomForest  = rf_model,
  SVM_RBF       = svm_model
)
if (exists("xgb_model") && !is.null(xgb_model))
  model_objs[["XGBoost"]] <- xgb_model

resamps <- resamples(model_objs)
cat("\n=== Cross-Validation ROC Summary ===\n")
print(summary(resamps)$statistics$ROC)

png("plot_cv_dotplot.png", width = 750, height = 500, res = 110)
dotplot(resamps, metric = "ROC", main = "CV AUC Distribution by Model")
dev.off()

cat("\nAll outputs saved. Done!\n")
