/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include "gbdt.h"

namespace LightGBM {

void GBDT::PredictRaw(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  int early_stop_round_counter = 0;
  // set zero
  std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
  for (int i = 0; i < num_iteration_for_pred_; ++i) {
    // predict all the trees for one iteration
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      auto value = models_[i * num_tree_per_iteration_ + k]->Predict(features);
      printf("[cpp debug] GBDT::PredictRaw value: %f\n", value);
      output[k] += value;
      printf("[cpp debug] GBDT::PredictRaw output[%d]: %f\n", k, output[k]);
    }
    // check early stopping
    ++early_stop_round_counter;
    if (early_stop->round_period == early_stop_round_counter) {
      if (early_stop->callback_function(output, num_tree_per_iteration_)) {
        return;
      }
      early_stop_round_counter = 0;
    }
  }
}

void GBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  int early_stop_round_counter = 0;
  // set zero
  std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
  for (int i = 0; i < num_iteration_for_pred_; ++i) {
    // predict all the trees for one iteration
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] += models_[i * num_tree_per_iteration_ + k]->PredictByMap(features);
    }
    // check early stopping
    ++early_stop_round_counter;
    if (early_stop->round_period == early_stop_round_counter) {
      if (early_stop->callback_function(output, num_tree_per_iteration_)) {
        return;
      }
      early_stop_round_counter = 0;
    }
  }
}

void GBDT::Predict(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  PredictRaw(features, output, early_stop);
  printf("[cpp debug] GBDT::Predict %.8f\n", *output);
  double value = 1.0f / (1.0f + std::exp(-1 * (*output)));
  printf("[cpp debug] GBDT::Predict after sigmoid %.8f\n", value);
  if (average_output_) {
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] /= num_iteration_for_pred_;
    }
  }
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output);
  }
}

void GBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  PredictRawByMap(features, output, early_stop);
  if (average_output_) {
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] /= num_iteration_for_pred_;
    }
  }
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output);
  }
}

void GBDT::PredictLeafIndex(const double* features, double* output) const {
  int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
  for (int i = 0; i < total_tree; ++i) {
    output[i] = models_[i]->PredictLeafIndex(features);
  }
}

void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {
  int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
  for (int i = 0; i < total_tree; ++i) {
    output[i] = models_[i]->PredictLeafIndexByMap(features);
  }
}

}  // namespace LightGBM
