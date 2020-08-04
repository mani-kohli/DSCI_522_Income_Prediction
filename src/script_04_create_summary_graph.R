#! /usr/bin/env Rscript 

# script_04_create_summary_graph.R 
# Olivia Lin, Mani Kohli Nov 23 2018

# This script takes in summary tables and returns graphs for the summary tables.
# This script should be run under the root folder.

# Usage: Rscript src/script_04_create_summary_graph.R arg1 arg2 arg3 arg4
# Description: arg1 = input data file 1, arg1 = input data file 2, arg3 = output graph 1, arg4 = output graph 4
# Example: Rscript src/script_04_create_summary_graph.R results/depth_summary.csv results/feature_summary.csv results/depth_graph.png results/feature_graph.png 

# import libraries/packages
library(tidyverse)

# parse/define command line arguments here
args <- commandArgs(trailingOnly = TRUE)
depth_table <- args[1]
feature_table <- args[2]
depth_graph <- args[3]
feature_graph <- args[4]

# loading data
depth <- read_csv(depth_table)
feature <- read_csv(feature_table)

# creating max_depth vs accuracy graph
depth %>% 
  ggplot(aes(max_depth, accuracy)) +
  geom_line() +
  geom_point() +
  labs(x = "Decision Tree Max Depth", y = "Cross Validation Accuracy", title = "Find Optimal Tree Depth")
# save depth plot
ggsave(filename = depth_graph, plot = last_plot(), width = 7.29, height = 4.51, units = "in")

# creating feature importance graph
feature %>% 
  ggplot(aes(fct_reorder(feature$variable, feature$feature_importance), feature_importance)) +
  geom_col() +
  coord_flip() +
  labs(x = "Variables", y = "Decision Tree Feature Importance", title = "Feature Importances for Decision Tree")
# save depth plot
ggsave(filename = feature_graph, plot = last_plot(), width = 7.29, height = 4.51, units = "in")
