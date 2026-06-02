# ============================================
# Complete EDA for Predictive Maintenance
# Visual Analytics Project
# ============================================

# Load libraries
library(ggplot2)
library(dplyr)
library(corrplot)
library(tidyr)

# Read data
df <- read_csv("ai4i2020.csv")

# Fix column names (remove spaces for easier handling)
names(df) <- gsub(" ", "_", names(df))
names(df) <- gsub("\\[", "", names(df))
names(df) <- gsub("\\]", "", names(df))

print("Column names after cleaning:")
print(names(df))

# ============================================
# 1. ANOMALY RATE
# ============================================
anomaly_rate <- mean(df$Machine_failure)
cat("\n========== What is the anomaly rate in the dataset ? ==========\n")
cat("Anomaly Rate:", round(anomaly_rate * 100, 2), "%\n")
cat("Total failures:", sum(df$Machine_failure), "out of", nrow(df), "samples\n")
cat("Total normal:", sum(df$Machine_failure == 0), "samples\n\n")

# ============================================
# 2. VISUALIZATIONS
# ============================================

# Set up plotting
par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))

# 2.1 Distribution of target variable
barplot(table(df$Machine_failure),
        names.arg = c("Normal (0)", "Failure (1)"),
        main = "Machine Failure Distribution",
        col = c("steelblue", "firebrick"),
        ylab = "Count",
        las = 1)
text(x = c(0.7, 1.9),
     y = c(table(df$Machine_failure)[1], table(df$Machine_failure)[2]) + 200,
     labels = table(df$Machine_failure))

# 2.2 Boxplot: Rotational Speed by Failure
boxplot(Rotational_speed_rpm ~ Machine_failure, data = df,
        main = "Rotational Speed vs Failure",
        xlab = "Machine Failure (0=No, 1=Yes)",
        ylab = "Rotational Speed (rpm)",
        col = c("steelblue", "firebrick"))

# 2.3 Boxplot: Torque by Failure
boxplot(Torque_Nm ~ Machine_failure, data = df,
        main = "Torque vs Failure",
        xlab = "Machine Failure (0=No, 1=Yes)",
        ylab = "Torque (Nm)",
        col = c("steelblue", "firebrick"))

# 2.4 Boxplot: Tool Wear by Failure
boxplot(Tool_wear_min ~ Machine_failure, data = df,
        main = "Tool Wear vs Failure",
        xlab = "Machine Failure (0=No, 1=Yes)",
        ylab = "Tool Wear (min)",
        col = c("steelblue", "firebrick"))

# 2.5 Histogram of Air Temperature
hist(df$Air_temperature_K,
     main = "Air Temperature Distribution",
     xlab = "Air Temperature (K)",
     col = "lightblue",
     breaks = 30)

# 2.6 Histogram of Process Temperature
hist(df$Process_temperature_K,
     main = "Process Temperature Distribution",
     xlab = "Process Temperature (K)",
     col = "lightgreen",
     breaks = 30)

# ============================================
# 3. CORRELATION MATRIX
# ============================================

# Select numeric features
numeric_cols <- c("Air_temperature_K", "Process_temperature_K",
                  "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min")

# Calculate correlation
cor_matrix <- cor(df[, numeric_cols])

# Plot correlation heatmap
par(mfrow = c(1, 1))
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.8,
         title = "Correlation Matrix of Sensor Features",
         mar = c(0, 0, 2, 0))

# ============================================
# 4. SCATTER PLOTS FOR ANOMALY DETECTION
# ============================================

# Scatter: Rotational Speed vs Torque (colored by failure)
ggplot(df, aes(x = Rotational_speed_rpm, y = Torque_Nm,
               color = factor(Machine_failure))) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = c("steelblue", "firebrick"),
                     labels = c("Normal", "Failure"),
                     name = "Status") +
  labs(title = "Rotational Speed vs Torque",
       subtitle = paste("Failures:", sum(df$Machine_failure),
                        "out of", nrow(df), "samples"),
       x = "Rotational Speed (rpm)",
       y = "Torque (Nm)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Scatter: Tool Wear vs Rotational Speed
ggplot(df, aes(x = Tool_wear_min, y = Rotational_speed_rpm,
               color = factor(Machine_failure))) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = c("steelblue", "firebrick"),
                     labels = c("Normal", "Failure"),
                     name = "Status") +
  labs(title = "Tool Wear vs Rotational Speed",
       x = "Tool Wear (min)",
       y = "Rotational Speed (rpm)") +
  theme_minimal()

# ============================================
# 5. SUMMARY STATISTICS
# ============================================

cat("\n========== SUMMARY STATISTICS ==========\n")
cat("\n--- By Failure Status ---\n")

# Summary for normal vs failure
normal_data <- df[df$Machine_failure == 0, ]
failure_data <- df[df$Machine_failure == 1, ]

cat("\nNormal operations (n =", nrow(normal_data), "):\n")
cat("  Rotational Speed (mean ± sd):",
    round(mean(normal_data$Rotational_speed_rpm), 1), "±",
    round(sd(normal_data$Rotational_speed_rpm), 1), "rpm\n")
cat("  Torque (mean ± sd):",
    round(mean(normal_data$Torque_Nm), 1), "±",
    round(sd(normal_data$Torque_Nm), 1), "Nm\n")
cat("  Tool Wear (mean ± sd):",
    round(mean(normal_data$Tool_wear_min), 1), "±",
    round(sd(normal_data$Tool_wear_min), 1), "min\n")

cat("\nFailure operations (n =", nrow(failure_data), "):\n")
cat("  Rotational Speed (mean ± sd):",
    round(mean(failure_data$Rotational_speed_rpm), 1), "±",
    round(sd(failure_data$Rotational_speed_rpm), 1), "rpm\n")
cat("  Torque (mean ± sd):",
    round(mean(failure_data$Torque_Nm), 1), "±",
    round(sd(failure_data$Torque_Nm), 1), "Nm\n")
cat("  Tool Wear (mean ± sd):",
    round(mean(failure_data$Tool_wear_min), 1), "±",
    round(sd(failure_data$Tool_wear_min), 1), "min\n")

# ============================================
# 6. MACHINE TYPE ANALYSIS
# ============================================

cat("\n========== MACHINE TYPE ANALYSIS ==========\n")
type_summary <- df %>%
  group_by(Type) %>%
  summarise(
    Count = n(),
    Failures = sum(Machine_failure),
    Failure_Rate = round(Failures / Count * 100, 2)
  )
print(type_summary)

# Bar plot by machine type
ggplot(df, aes(x = Type, fill = factor(Machine_failure))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("steelblue", "firebrick"),
                    labels = c("Normal", "Failure"),
                    name = "Status") +
  labs(title = "Failure Rate by Machine Type",
       y = "Proportion",
       x = "Machine Type (L=Low, M=Medium, H=High)") +
  theme_minimal() +
  coord_flip()

# ============================================
# 7. FINAL ASSESSMENT
# ============================================

cat("\n========== PROJECT SUITABILITY ASSESSMENT ==========\n")
cat("✓ Anomaly rate:", round(anomaly_rate * 100, 2), "% (Good for anomaly detection)\n")
cat("✓ Features:", paste(numeric_cols, collapse = ", "), "\n")
cat("✓ Target: Binary classification (Normal vs Failure)\n")
cat("✓ Sample size:", nrow(df), "rows\n\n")

cat("Recommended ML Models for this dataset:\n")
cat("  - Isolation Forest (pure anomaly detection)\n")
cat("  - XGBoost Classifier (good with SHAP for XAI)\n")
cat("  - Random Forest (interpretable with feature importance)\n\n")

cat("Recommended XAI Methods:\n")
cat("  - SHAP (TreeSHAP) - Best for tree-based models\n")
cat("  - LIME - Good for local explanations\n")
cat("  - Partial Dependence Plots - Global feature effects\n")