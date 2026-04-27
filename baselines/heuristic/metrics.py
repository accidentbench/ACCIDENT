from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np

LABELS_PATH = "metadata-real.csv"

#################
# TEMPORAL TASK #
#################
def temporal_accuracy_metric(predictions: Iterable[int], truth: Iterable[int], sigma: float) -> float:
    predictions, truth = np.array(predictions), np.array(truth)
    sigmas = np.ones_like(predictions) * sigma
    scores = np.exp( -(predictions - truth)**2 / (2 * sigmas**2))
    return float(np.mean(scores))

def print_temporal_accuracy(predictions: pd.DataFrame, dataset_path: Path, sigmas: list[float] = [1/2, 1, 2]):
	true_df = pd.read_csv(dataset_path / LABELS_PATH)
	merged = true_df.merge(
			predictions,
			on="path",
			how="inner",
			suffixes=("_true", "_pred")
		)
      
	print("Temporal task:")
	print("____________________")
	print("| Sigma | Accuracy |")
	for sigma in sigmas:
		acc = temporal_accuracy_metric(
			predictions=merged["accident_time_pred"],
			truth=merged["accident_time_true"],
			sigma=sigma
		)
		
		print(f"| {sigma:.2f}  | {acc:.3f}    |")
	print("\n")

################
# SPATIAL TASK #
################

def spatial_accuracy_metric(predictions: Iterable[tuple[float, float]], truth: Iterable[tuple[float, float]], sigma: tuple[float, float]) -> float:
    predictions, truth = np.array(predictions), np.array(truth)
    sigmas = np.ones_like(predictions) * np.array(sigma)
    scores = np.exp(-(
        ((predictions[:, 0] - truth[:, 0])**2 / (2 * sigmas[:, 0]**2)) +
        ((predictions[:, 1] - truth[:, 1])**2 / (2 * sigmas[:, 1]**2))
    ))
    return float(np.mean(scores))


def print_spatial_accuracy(predictions: pd.DataFrame, dataset_path: Path, sigmas: list[float] = [1/2, 1, 2]):
	true_df = pd.read_csv(dataset_path / LABELS_PATH)
	merged = true_df.merge(
			predictions,
			on="path",
			how="inner",
			suffixes=("_true", "_pred")
		)
	normalized_sigma_x = np.array(true_df["x2"] - true_df["x1"]).mean()
	normalized_sigma_y = np.array(true_df["y2"] - true_df["y1"]).mean()
	
	print("Spatial task:")
	print("Normalized sigmas: ", normalized_sigma_x, normalized_sigma_y)
	print("____________________")
	print("| Sigma | Accuracy |")
	for sigma in sigmas:
		acc = spatial_accuracy_metric(
			predictions=list(zip(merged["center_x_pred"], merged["center_y_pred"])),
			truth=list(zip(merged["center_x_true"], merged["center_y_true"])),
			sigma=(normalized_sigma_x * sigma, normalized_sigma_y * sigma)
		)
		
		print(f"| {sigma:.2f}  | {acc:.3f}    |")
	print("\n")
