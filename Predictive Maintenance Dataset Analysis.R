# ============================================
# Predictive Maintenance Dataset Analysis
# AI4I 2020 Dataset
# ============================================

# 1. Load the data
library(readr)
df <- read_csv("ai4i2020.csv")

# 2. View basic info
cat("=== DATASET OVERVIEW ===\n")
cat("Rows:", nrow(df), "\n")
cat("Columns:", ncol(df), "\n\n")

# 3. Check column names
cat("=== COLUMN NAMES ===\n")
print(names(df))
cat("\n")

# 4. Check for missing values
cat("=== MISSING VALUES ===\n")
print(colSums(is.na(df)))
cat("\n")

# 5. Target variable: Machine failure
# Note: The dataset has "Machine failure" column (with space)
# Let's find the correct column name
failure_col <- grep("failure", names(df), ignore.case = TRUE, value = TRUE)
cat("Failure column:", failure_col, "\n\n")

# 6. Calculate anomaly rate
failure_col_name <- "Machine failure"  # Based on the dataset file
anomaly_rate <- mean(df[[failure_col_name]])

cat("=== What is the anomaly rate in the dataset ? ===\n")
cat("Anomaly (Machine Failure) Rate:", anomaly_rate * 100, "%\n")
cat("Number of failures:", sum(df[[failure_col_name]]), "out of", nrow(df), "\n")
cat("Number of normal operations:", sum(df[[failure_col_name]] == 0), "\n\n")

# 7. Check if suitable for anomaly detection (should be 1-10%)
if(anomaly_rate > 0.01 & anomaly_rate < 0.10) {
  cat("✅ Dataset is suitable for anomaly detection!\n")
  cat("   (Anomaly rate is between 1% and 10%)\n")
} else {
  cat("⚠️ Anomaly rate is", anomaly_rate * 100, "%\n")
  cat("   Ideal range for anomaly detection is 1-10%\n")
}
cat("\n")

# 8. Check failure types distribution
cat("=== FAILURE TYPE DISTRIBUTION ===\n")
failure_types <- c("TWF", "HDF", "PWF", "OSF", "RNF")
for(ft in failure_types) {
  if(ft %in% names(df)) {
    cat(ft, ":", sum(df[[ft]] == 1), "occurrences\n")
  }
}