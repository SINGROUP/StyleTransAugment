#!/usr/bin/env python

import io
import os
import tarfile
import argparse
import subprocess
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import gaussian_kde

try:
	import cv2
except Exception:
	cv2 = None

# Original data
original_data_path = "/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-bilayer-FB/Water-bilayer_FB_Ref"
# Style translated data
# This one type of dataset with L1L2=(20, 1)
styletrans_data_path = "/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-bilayer-FB/Water-bilayer_FB_PPAFM2Exp_CoAll_L20_L1_Elatest"

output_dir = "../../results/dataWideStyletransAnalysis/L20_L1"


def build_validation_tar_list():
	"""Return expected validation tar names Water-bilayer-K-{1..10}_val_{0..3}.tar."""
	tar_names = []
	for k in range(1, 11):
		for v in range(4):
			tar_names.append(f"Water-bilayer-K-{k}_val_{v}.tar")
	return tar_names


def parse_slice_member_name(member_name):
	"""Parse 'sample.slice.png' and return (sample_id, slice_idx) or None."""
	base = os.path.basename(member_name)
	parts = base.split(".")
	if len(parts) != 3:
		return None
	if parts[2].lower() != "png":
		return None
	if not parts[0].isdigit() or not parts[1].isdigit():
		return None
	return int(parts[0]), int(parts[1])


def list_sample_slices_in_tar(tar_path):
	"""Return mapping sample_id -> {slice_idx: member_name} for a tar file."""
	sample_map = defaultdict(dict)
	with tarfile.open(tar_path, "r") as tar:
		for m in tar.getmembers():
			if not m.isfile():
				continue
			parsed = parse_slice_member_name(m.name)
			if parsed is None:
				continue
			sample_id, slice_idx = parsed
			sample_map[sample_id][slice_idx] = m.name
	return sample_map


def read_tar_grayscale_image(tar_obj, member_name):
	"""Read PNG member from tar to grayscale float32 array in [0, 1]."""
	fobj = tar_obj.extractfile(member_name)
	if fobj is None:
		raise ValueError(f"Cannot extract member {member_name}")
	raw = fobj.read()
	arr = np.array(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.float32) / 255.0
	return arr


def compute_ssim_2d(img1, img2, sigma=1.5, k1=0.01, k2=0.03):
	"""Compute SSIM between two 2D arrays using Gaussian local statistics."""
	if img1.shape != img2.shape:
		raise ValueError("SSIM images must have the same shape")

	x = img1.astype(np.float64)
	y = img2.astype(np.float64)
	data_range = max(float(max(x.max(), y.max()) - min(x.min(), y.min())), 1e-8)

	c1 = (k1 * data_range) ** 2
	c2 = (k2 * data_range) ** 2

	mu_x = gaussian_filter(x, sigma=sigma)
	mu_y = gaussian_filter(y, sigma=sigma)

	mu_x2 = mu_x * mu_x
	mu_y2 = mu_y * mu_y
	mu_xy = mu_x * mu_y

	sigma_x2 = gaussian_filter(x * x, sigma=sigma) - mu_x2
	sigma_y2 = gaussian_filter(y * y, sigma=sigma) - mu_y2
	sigma_xy = gaussian_filter(x * y, sigma=sigma) - mu_xy

	numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
	denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
	ssim_map = numerator / (denominator + 1e-12)
	return float(np.mean(ssim_map))


def compute_flat_correlation(img1, img2):
	"""Pearson correlation of flattened images; returns nan for degenerate arrays."""
	x = img1.ravel()
	y = img2.ravel()
	sx = np.std(x)
	sy = np.std(y)
	if sx < 1e-12 or sy < 1e-12:
		return np.nan
	return float(np.corrcoef(x, y)[0, 1])


def to_u8_image(img):
	x = img.astype(np.float64)
	x = np.clip(x, 0.0, 1.0)
	return np.round(255.0 * x).astype(np.uint8)


def enhance_for_dark(gray_u8, sigma):
	background = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
	return cv2.subtract(background, gray_u8)


def enhance_for_bright(gray_u8, sigma):
	background = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
	return cv2.subtract(gray_u8, background)


def build_blob_detector(min_area, max_area, min_circularity, min_inertia):
	params = cv2.SimpleBlobDetector_Params()
	params.minThreshold = 5
	params.maxThreshold = 220
	params.thresholdStep = 5
	params.minDistBetweenBlobs = 8
	params.filterByColor = True
	params.blobColor = 255
	params.filterByArea = True
	params.minArea = float(min_area)
	params.maxArea = float(max_area)
	params.filterByCircularity = True
	params.minCircularity = float(min_circularity)
	params.filterByInertia = True
	params.minInertiaRatio = float(min_inertia)
	params.filterByConvexity = False
	return cv2.SimpleBlobDetector_create(params)


def keypoints_to_blob_dicts(keypoints, score_image, shape, border_margin):
	h, w = shape
	blobs = []
	for kp in keypoints:
		x = float(kp.pt[0])
		y = float(kp.pt[1])
		radius = max(2.0, float(kp.size) * 0.5)
		margin = max(int(border_margin), int(round(radius + 2)))
		if x < margin or x > (w - margin) or y < margin or y > (h - margin):
			continue

		yi = int(np.clip(round(y), 0, h - 1))
		xi = int(np.clip(round(x), 0, w - 1))
		score = float(score_image[yi, xi])
		blobs.append({"y": y, "x": x, "radius": radius, "score": score})

	blobs.sort(key=lambda b: b["score"], reverse=True)
	return blobs


def select_dark_holes(blobs, score_ratio=0.42, max_blobs=8):
	if not blobs:
		return []
	top = float(blobs[0]["score"])
	selected = [b for b in blobs if float(b["score"]) >= float(score_ratio) * top]
	return selected[: int(max_blobs)]


def select_bright_blobs(blobs, score_ratio=0.55, max_blobs=3):
	if not blobs:
		return []
	top = float(blobs[0]["score"])
	selected = [b for b in blobs if float(b["score"]) >= float(score_ratio) * top]
	return selected[: int(max_blobs)]


def blobs_to_mask(shape, blobs, radius_scale=1.8):
	h, w = shape
	yy, xx = np.ogrid[:h, :w]
	mask = np.zeros((h, w), dtype=bool)
	for b in blobs:
		r = float(radius_scale) * float(b["radius"])
		disk = (yy - float(b["y"])) ** 2 + (xx - float(b["x"])) ** 2 <= (r * r)
		mask |= disk
	return mask


def detect_dark_bright_feature_masks(img, cfg, return_blobs=False):
	if cv2 is None:
		raise ImportError("OpenCV (cv2) is required for feature-method 'blob_dark_bright'.")

	gray_u8 = to_u8_image(img)
	dark_enhanced = enhance_for_dark(gray_u8, sigma=cfg["dark_sigma"])
	bright_enhanced = enhance_for_bright(gray_u8, sigma=cfg["bright_sigma"])

	dark_detector = build_blob_detector(
		min_area=cfg["dark_min_area"],
		max_area=cfg["dark_max_area"],
		min_circularity=cfg["dark_min_circularity"],
		min_inertia=cfg["dark_min_inertia"],
	)
	bright_detector = build_blob_detector(
		min_area=cfg["bright_min_area"],
		max_area=cfg["bright_max_area"],
		min_circularity=cfg["bright_min_circularity"],
		min_inertia=cfg["bright_min_inertia"],
	)

	dark_candidates = keypoints_to_blob_dicts(
		dark_detector.detect(dark_enhanced),
		score_image=dark_enhanced,
		shape=gray_u8.shape,
		border_margin=cfg["dark_border_margin"],
	)
	bright_candidates = keypoints_to_blob_dicts(
		bright_detector.detect(bright_enhanced),
		score_image=bright_enhanced,
		shape=gray_u8.shape,
		border_margin=cfg["bright_border_margin"],
	)

	dark_blobs = select_dark_holes(
		dark_candidates,
		score_ratio=cfg["dark_score_ratio"],
		max_blobs=cfg["dark_max_blobs"],
	)
	bright_blobs = select_bright_blobs(
		bright_candidates,
		score_ratio=cfg["bright_score_ratio"],
		max_blobs=cfg["bright_max_blobs"],
	)

	dark_mask = blobs_to_mask(gray_u8.shape, dark_blobs, radius_scale=cfg["dark_radius_scale"])
	bright_mask = blobs_to_mask(gray_u8.shape, bright_blobs, radius_scale=cfg["bright_radius_scale"])
	combined = np.logical_or(dark_mask, bright_mask)
	if return_blobs:
		return dark_mask, bright_mask, combined, dark_blobs, bright_blobs
	return dark_mask, bright_mask, combined


def compute_dice_distance_from_masks(mx, my):
	den = float(mx.sum() + my.sum())
	if den <= 0.0:
		return np.nan, np.nan
	inter = float(np.logical_and(mx, my).sum())
	dice = (2.0 * inter) / (den + 1e-12)
	distance = 1.0 - dice
	return float(dice), float(distance)


def compute_feature_mask_metrics(img1, img2, top_percent=5.0, use_gradient=True, method="legacy_percentile", blob_cfg=None):
	"""Compute feature-location metrics using top-percentile masks.

	Returns (dice_overlap, feature_distance), where feature_distance = 1 - dice_overlap.
	"""
	if img1.shape != img2.shape:
		raise ValueError("Feature-mask images must have the same shape")

	if method == "blob_dark_bright":
		if blob_cfg is None:
			raise ValueError("blob_cfg is required for method='blob_dark_bright'")
		_, _, mx = detect_dark_bright_feature_masks(img1, blob_cfg)
		_, _, my = detect_dark_bright_feature_masks(img2, blob_cfg)
		return compute_dice_distance_from_masks(mx, my)

	if method != "legacy_percentile":
		raise ValueError(f"Unsupported feature method: {method}")

	x = img1.astype(np.float64)
	y = img2.astype(np.float64)

	if use_gradient:
		gx1 = sobel(x, axis=1, mode="reflect")
		gy1 = sobel(x, axis=0, mode="reflect")
		gx2 = sobel(y, axis=1, mode="reflect")
		gy2 = sobel(y, axis=0, mode="reflect")
		x = np.sqrt(gx1 * gx1 + gy1 * gy1)
		y = np.sqrt(gx2 * gx2 + gy2 * gy2)

	top_percent = float(np.clip(top_percent, 0.01, 100.0))
	cut_percentile = 100.0 - top_percent

	tx = np.percentile(x, cut_percentile)
	ty = np.percentile(y, cut_percentile)

	mx = x >= tx
	my = y >= ty

	den = float(mx.sum() + my.sum())
	if den <= 0.0:
		return np.nan, np.nan

	inter = float(np.logical_and(mx, my).sum())
	dice = (2.0 * inter) / (den + 1e-12)
	distance = 1.0 - dice
	return float(dice), float(distance)


def sem(values):
	valid = np.asarray(values, dtype=float)
	valid = valid[np.isfinite(valid)]
	if valid.size <= 1:
		return np.nan
	return float(np.std(valid, ddof=1) / np.sqrt(valid.size))


def plot_kde_density(ax, values, color, label, x_grid):
	valid = np.asarray(values, dtype=float)
	valid = valid[np.isfinite(valid)]
	if valid.size == 0:
		return

	# Fall back to a narrow Gaussian when KDE is ill-posed (e.g., single/constant values).
	if valid.size <= 1 or np.allclose(valid, valid[0]):
		mu = float(valid[0])
		sigma = 0.015
		y = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
	else:
		try:
			y = gaussian_kde(valid, bw_method="scott")(x_grid)
		except Exception:
			mu = float(np.mean(valid))
			sigma = max(1e-3, float(np.std(valid, ddof=0)))
			y = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

	ax.plot(x_grid, y, color=color, linewidth=2.0, label=label)
	ax.fill_between(x_grid, 0.0, y, color=color, alpha=0.16)


def create_sanity_plot(ssim_df, corr_df, output_path, max_samples=2):
	sample_keys = (
		ssim_df[["tar_file", "sample_id"]]
		.drop_duplicates()
		.sort_values(["tar_file", "sample_id"])
		.head(max_samples)
	)
	n_samples = len(sample_keys)
	if n_samples == 0:
		return

	fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples), squeeze=False)
	for row_idx, (_, row) in enumerate(sample_keys.iterrows()):
		tar_name = row["tar_file"]
		sample_id = int(row["sample_id"])

		sub_ssim = ssim_df[(ssim_df["tar_file"] == tar_name) & (ssim_df["sample_id"] == sample_id)].sort_values("slice_idx")
		sub_corr = corr_df[(corr_df["tar_file"] == tar_name) & (corr_df["sample_id"] == sample_id)].sort_values("pair_idx")

		ax_left = axes[row_idx, 0]
		ax_left.plot(sub_ssim["slice_idx"], sub_ssim["ssim"], marker="o", linewidth=1.5)
		ax_left.set_ylim(0.0, 1.05)
		ax_left.set_xlabel("Slice index")
		ax_left.set_ylabel("SSIM")
		ax_left.set_title(f"{tar_name} | sample {sample_id}: slice SSIM")
		ax_left.grid(alpha=0.2)

		ax_right = axes[row_idx, 1]
		ax_right.plot(sub_corr["pair_idx"], sub_corr["corr_original"], marker="o", linewidth=1.5, label="Original")
		ax_right.plot(sub_corr["pair_idx"], sub_corr["corr_translated"], marker="o", linewidth=1.5, label="Translated")
		ax_right.set_xlabel("Adjacent pair index (i: i vs i+1)")
		ax_right.set_ylabel("Correlation")
		ax_right.set_ylim(-1.05, 1.05)
		ax_right.set_title(f"{tar_name} | sample {sample_id}: adjacent-slice corr")
		ax_right.legend(loc="best", frameon=False)
		ax_right.grid(alpha=0.2)

	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description="Dataset-wide content-preservation analysis for AFM CycleGAN outputs")
	parser.add_argument(
		"--styletrans-data-path",
		default=styletrans_data_path,
		help="Path to translated data root for one dataset",
	)
	parser.add_argument(
		"--output-dir",
		default=output_dir,
		help="Output directory for one dataset run",
	)
	parser.add_argument(
		"--compare-l1-values",
		default="",
		help="Comma-separated L1 values (for example: 10,20,30,40,50) to run batch comparisons",
	)
	parser.add_argument(
		"--l2-value",
		type=int,
		default=1,
		help="L2 value used with --compare-l1-values (default: 1)",
	)
	parser.add_argument(
		"--styletrans-path-template",
		default="/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-bilayer-FB/Water-bilayer_FB_PPAFM2Exp_CoAll_L{l1}_L{l2}_Elatest",
		help="Template for translated dataset paths. Uses placeholders {l1} and {l2}",
	)
	parser.add_argument(
		"--output-root",
		default=output_dir,
		help="Root output directory for batch compare mode",
	)
	parser.add_argument(
		"--recompute",
		action="store_true",
		help="Force recomputation from tar files even if cached CSVs already exist",
	)
	parser.add_argument(
		"--feature-top-percent",
		type=float,
		default=5.0,
		help="Top percentile of strongest feature pixels used for mask-based feature metrics",
	)
	parser.add_argument(
		"--feature-no-gradient",
		action="store_true",
		help="Disable gradient preprocessing before legacy-percentile feature-mask computation",
	)
	parser.add_argument(
		"--feature-method",
		default="blob_dark_bright",
		choices=["legacy_percentile", "blob_dark_bright"],
		help="Feature map method used to compute feature Dice",
	)
	parser.add_argument("--feature-dark-sigma", type=float, default=13.0, help="Gaussian sigma for dark enhancement")
	parser.add_argument("--feature-bright-sigma", type=float, default=8.0, help="Gaussian sigma for bright enhancement")
	parser.add_argument("--feature-dark-min-area", type=float, default=35.0, help="Minimum area for dark-hole detector")
	parser.add_argument("--feature-dark-max-area", type=float, default=1500.0, help="Maximum area for dark-hole detector")
	parser.add_argument("--feature-dark-min-circularity", type=float, default=0.24, help="Minimum circularity for dark-hole detector")
	parser.add_argument("--feature-dark-min-inertia", type=float, default=0.06, help="Minimum inertia ratio for dark-hole detector")
	parser.add_argument("--feature-bright-min-area", type=float, default=15.0, help="Minimum area for bright-spot detector")
	parser.add_argument("--feature-bright-max-area", type=float, default=2000.0, help="Maximum area for bright-spot detector")
	parser.add_argument("--feature-bright-min-circularity", type=float, default=0.05, help="Minimum circularity for bright-spot detector")
	parser.add_argument("--feature-bright-min-inertia", type=float, default=0.01, help="Minimum inertia ratio for bright-spot detector")
	parser.add_argument("--feature-dark-border-margin", type=int, default=14, help="Border margin for dark-hole blob candidates")
	parser.add_argument("--feature-bright-border-margin", type=int, default=18, help="Border margin for bright-spot blob candidates")
	parser.add_argument("--feature-dark-score-ratio", type=float, default=0.42, help="Keep dark blobs with score >= ratio * top dark score")
	parser.add_argument("--feature-bright-score-ratio", type=float, default=0.30, help="Keep bright blobs with score >= ratio * top bright score")
	parser.add_argument("--feature-dark-max-blobs", type=int, default=8, help="Maximum number of selected dark blobs")
	parser.add_argument("--feature-bright-max-blobs", type=int, default=8, help="Maximum number of selected bright blobs")
	parser.add_argument("--feature-dark-radius-scale", type=float, default=1.8, help="Radius scale used when rasterizing dark blobs")
	parser.add_argument("--feature-bright-radius-scale", type=float, default=2.0, help="Radius scale used when rasterizing bright blobs")
	args = parser.parse_args()

	blob_cfg = {
		"dark_sigma": args.feature_dark_sigma,
		"bright_sigma": args.feature_bright_sigma,
		"dark_min_area": args.feature_dark_min_area,
		"dark_max_area": args.feature_dark_max_area,
		"dark_min_circularity": args.feature_dark_min_circularity,
		"dark_min_inertia": args.feature_dark_min_inertia,
		"bright_min_area": args.feature_bright_min_area,
		"bright_max_area": args.feature_bright_max_area,
		"bright_min_circularity": args.feature_bright_min_circularity,
		"bright_min_inertia": args.feature_bright_min_inertia,
		"dark_border_margin": args.feature_dark_border_margin,
		"bright_border_margin": args.feature_bright_border_margin,
		"dark_score_ratio": args.feature_dark_score_ratio,
		"bright_score_ratio": args.feature_bright_score_ratio,
		"dark_max_blobs": args.feature_dark_max_blobs,
		"bright_max_blobs": args.feature_bright_max_blobs,
		"dark_radius_scale": args.feature_dark_radius_scale,
		"bright_radius_scale": args.feature_bright_radius_scale,
	}

	if args.compare_l1_values.strip():
		l1_values = [int(v.strip()) for v in args.compare_l1_values.split(",") if v.strip()]
		if not l1_values:
			raise ValueError("--compare-l1-values was provided but no valid integers were parsed")

		os.makedirs(args.output_root, exist_ok=True)
		print(f"Batch compare mode over L1 values: {l1_values} with L2={args.l2_value}", flush=True)
		for l1 in l1_values:
			style_path = args.styletrans_path_template.format(l1=l1, l2=args.l2_value)
			sub_out_dir = os.path.join(args.output_root, f"L{l1}_L{args.l2_value}")
			cmd = [
				sys.executable,
				os.path.abspath(__file__),
				"--styletrans-data-path",
				style_path,
				"--output-dir",
				sub_out_dir,
				"--feature-top-percent",
				str(args.feature_top_percent),
				"--feature-method",
				args.feature_method,
				"--feature-dark-sigma",
				str(args.feature_dark_sigma),
				"--feature-bright-sigma",
				str(args.feature_bright_sigma),
				"--feature-dark-min-area",
				str(args.feature_dark_min_area),
				"--feature-dark-max-area",
				str(args.feature_dark_max_area),
				"--feature-dark-min-circularity",
				str(args.feature_dark_min_circularity),
				"--feature-dark-min-inertia",
				str(args.feature_dark_min_inertia),
				"--feature-bright-min-area",
				str(args.feature_bright_min_area),
				"--feature-bright-max-area",
				str(args.feature_bright_max_area),
				"--feature-bright-min-circularity",
				str(args.feature_bright_min_circularity),
				"--feature-bright-min-inertia",
				str(args.feature_bright_min_inertia),
				"--feature-dark-border-margin",
				str(args.feature_dark_border_margin),
				"--feature-bright-border-margin",
				str(args.feature_bright_border_margin),
				"--feature-dark-score-ratio",
				str(args.feature_dark_score_ratio),
				"--feature-bright-score-ratio",
				str(args.feature_bright_score_ratio),
				"--feature-dark-max-blobs",
				str(args.feature_dark_max_blobs),
				"--feature-bright-max-blobs",
				str(args.feature_bright_max_blobs),
				"--feature-dark-radius-scale",
				str(args.feature_dark_radius_scale),
				"--feature-bright-radius-scale",
				str(args.feature_bright_radius_scale),
			]
			if args.recompute:
				cmd.append("--recompute")
			if args.feature_no_gradient:
				cmd.append("--feature-no-gradient")

			print(f"\n=== Running dataset L1={l1}, L2={args.l2_value} ===", flush=True)
			print(f"Translated root: {style_path}", flush=True)
			print(f"Output dir: {sub_out_dir}", flush=True)
			subprocess.run(cmd, check=True)

		print("\nBatch compare completed.", flush=True)
		return

	run_styletrans_data_path = args.styletrans_data_path
	run_output_dir = args.output_dir

	os.makedirs(run_output_dir, exist_ok=True)
	ssim_csv_path = os.path.join(run_output_dir, "ssim_values.csv")
	feature_csv_path = os.path.join(run_output_dir, "feature_mask_metrics.csv")
	corr_csv_path = os.path.join(run_output_dir, "adjacent_slice_correlations.csv")
	blob_csv_path = os.path.join(run_output_dir, "blob_features.csv")

	expected_tars = build_validation_tar_list()
	matched_tars = []
	missing_original = []
	missing_translated = []

	for tar_name in expected_tars:
		o_path = os.path.join(original_data_path, tar_name)
		t_path = os.path.join(run_styletrans_data_path, tar_name)
		has_o = os.path.isfile(o_path)
		has_t = os.path.isfile(t_path)
		if has_o and has_t:
			matched_tars.append(tar_name)
		elif not has_o:
			missing_original.append(tar_name)
		elif not has_t:
			missing_translated.append(tar_name)

	print(f"Expected validation tars: {len(expected_tars)}", flush=True)
	print(f"Matched validation tars: {len(matched_tars)}", flush=True)
	if missing_original:
		print(f"Missing in original domain: {len(missing_original)}", flush=True)
	if missing_translated:
		print(f"Missing in translated domain: {len(missing_translated)}", flush=True)

	if (not args.recompute) and os.path.isfile(ssim_csv_path) and os.path.isfile(feature_csv_path) and os.path.isfile(corr_csv_path):
		print("Using cached metric CSVs (skip tar processing).", flush=True)
		ssim_df = pd.read_csv(ssim_csv_path)
		feature_df = pd.read_csv(feature_csv_path)
		corr_df = pd.read_csv(corr_csv_path)
		blob_df = pd.read_csv(blob_csv_path) if os.path.isfile(blob_csv_path) else pd.DataFrame()
		n_samples_used = int(ssim_df[["tar_file", "sample_id"]].drop_duplicates().shape[0]) if len(ssim_df) > 0 else 0
		n_samples_skipped = np.nan
	else:
		ssim_rows = []
		feature_rows = []
		corr_rows = []
		blob_rows = []
		n_samples_used = 0
		n_samples_skipped = 0

		for tar_idx, tar_name in enumerate(matched_tars, start=1):
			print(f"[{tar_idx}/{len(matched_tars)}] Processing {tar_name}", flush=True)
			o_tar_path = os.path.join(original_data_path, tar_name)
			t_tar_path = os.path.join(run_styletrans_data_path, tar_name)

			o_map = list_sample_slices_in_tar(o_tar_path)
			t_map = list_sample_slices_in_tar(t_tar_path)

			common_samples = sorted(set(o_map.keys()).intersection(set(t_map.keys())))
			with tarfile.open(o_tar_path, "r") as o_tar, tarfile.open(t_tar_path, "r") as t_tar:
				for sample_id in common_samples:
					o_slices = o_map[sample_id]
					t_slices = t_map[sample_id]
					common_slice_ids = sorted(set(o_slices.keys()).intersection(set(t_slices.keys())))

					if len(common_slice_ids) < 2:
						n_samples_skipped += 1
						continue

					o_imgs = []
					t_imgs = []
					valid_slice_ids = []

					for slice_idx in common_slice_ids:
						try:
							o_img = read_tar_grayscale_image(o_tar, o_slices[slice_idx])
							t_img = read_tar_grayscale_image(t_tar, t_slices[slice_idx])
						except Exception:
							continue

						if o_img.shape != t_img.shape:
							continue

						ssim_val = compute_ssim_2d(o_img, t_img)
						if args.feature_method == "blob_dark_bright":
							_, _, o_mask, o_dark_blobs, o_bright_blobs = detect_dark_bright_feature_masks(
								o_img,
								blob_cfg,
								return_blobs=True,
							)
							_, _, t_mask, t_dark_blobs, t_bright_blobs = detect_dark_bright_feature_masks(
								t_img,
								blob_cfg,
								return_blobs=True,
							)
							feature_dice, feature_distance = compute_dice_distance_from_masks(o_mask, t_mask)

							for domain, dark_blobs, bright_blobs in [
								("original", o_dark_blobs, o_bright_blobs),
								("translated", t_dark_blobs, t_bright_blobs),
							]:
								for rank, b in enumerate(dark_blobs):
									blob_rows.append(
										{
											"tar_file": tar_name,
											"sample_id": sample_id,
											"slice_idx": slice_idx,
											"domain": domain,
											"blob_type": "dark",
											"rank": rank,
											"y": float(b["y"]),
											"x": float(b["x"]),
											"radius": float(b["radius"]),
											"radius_draw": float(blob_cfg["dark_radius_scale"]) * float(b["radius"]),
											"score": float(b["score"]),
										}
									)
								for rank, b in enumerate(bright_blobs):
									blob_rows.append(
										{
											"tar_file": tar_name,
											"sample_id": sample_id,
											"slice_idx": slice_idx,
											"domain": domain,
											"blob_type": "bright",
											"rank": rank,
											"y": float(b["y"]),
											"x": float(b["x"]),
											"radius": float(b["radius"]),
											"radius_draw": float(blob_cfg["bright_radius_scale"]) * float(b["radius"]),
											"score": float(b["score"]),
										}
									)
						else:
							feature_dice, feature_distance = compute_feature_mask_metrics(
								o_img,
								t_img,
								top_percent=args.feature_top_percent,
								use_gradient=(not args.feature_no_gradient),
								method=args.feature_method,
								blob_cfg=blob_cfg,
							)
						ssim_rows.append(
							{
								"tar_file": tar_name,
								"sample_id": sample_id,
								"slice_idx": slice_idx,
								"ssim": ssim_val,
							}
						)
						feature_rows.append(
							{
								"tar_file": tar_name,
								"sample_id": sample_id,
								"slice_idx": slice_idx,
								"feature_dice": feature_dice,
								"feature_distance": feature_distance,
							}
						)
						o_imgs.append(o_img)
						t_imgs.append(t_img)
						valid_slice_ids.append(slice_idx)

					if len(valid_slice_ids) < 2:
						n_samples_skipped += 1
						continue

					for i in range(len(valid_slice_ids) - 1):
						corr_o = compute_flat_correlation(o_imgs[i], o_imgs[i + 1])
						corr_t = compute_flat_correlation(t_imgs[i], t_imgs[i + 1])
						corr_rows.append(
							{
								"tar_file": tar_name,
								"sample_id": sample_id,
								"pair_idx": i,
								"slice_idx_i": valid_slice_ids[i],
								"slice_idx_ip1": valid_slice_ids[i + 1],
								"corr_original": corr_o,
								"corr_translated": corr_t,
								"diff_signed": corr_t - corr_o,
								"diff_abs": abs(corr_t - corr_o),
							}
						)

					n_samples_used += 1

			print(
				f"[{tar_idx}/{len(matched_tars)}] Done {tar_name}: cumulative samples used={n_samples_used}, skipped={n_samples_skipped}",
				flush=True,
			)

		ssim_df = pd.DataFrame(ssim_rows)
		feature_df = pd.DataFrame(feature_rows)
		corr_df = pd.DataFrame(corr_rows)
		blob_df = pd.DataFrame(blob_rows)
		ssim_df.to_csv(ssim_csv_path, index=False)
		feature_df.to_csv(feature_csv_path, index=False)
		corr_df.to_csv(corr_csv_path, index=False)
		if args.feature_method == "blob_dark_bright":
			blob_df.to_csv(blob_csv_path, index=False)

	if len(ssim_df) > 0:
		ssim_values = ssim_df["ssim"].to_numpy(dtype=float)
		ssim_mean = float(np.nanmean(ssim_values))
		ssim_median = float(np.nanmedian(ssim_values))
		ssim_std = float(np.nanstd(ssim_values, ddof=0))
	else:
		ssim_mean = np.nan
		ssim_median = np.nan
		ssim_std = np.nan

	if len(feature_df) > 0:
		feature_dice_values = feature_df["feature_dice"].to_numpy(dtype=float)
		feature_distance_values = feature_df["feature_distance"].to_numpy(dtype=float)
		feature_dice_mean = float(np.nanmean(feature_dice_values))
		feature_dice_median = float(np.nanmedian(feature_dice_values))
		feature_dice_std = float(np.nanstd(feature_dice_values, ddof=0))
		feature_distance_mean = float(np.nanmean(feature_distance_values))
		feature_distance_median = float(np.nanmedian(feature_distance_values))
		feature_distance_std = float(np.nanstd(feature_distance_values, ddof=0))
	else:
		feature_dice_mean = np.nan
		feature_dice_median = np.nan
		feature_dice_std = np.nan
		feature_distance_mean = np.nan
		feature_distance_median = np.nan
		feature_distance_std = np.nan

	if len(corr_df) > 0:
		avg_curve_diff_abs = float(np.nanmean(corr_df["diff_abs"].to_numpy(dtype=float)))
	else:
		avg_curve_diff_abs = np.nan

	summary_df = pd.DataFrame(
		[
			{
				"metric": "ssim_mean",
				"value": ssim_mean,
			},
			{
				"metric": "ssim_median",
				"value": ssim_median,
			},
			{
				"metric": "ssim_std",
				"value": ssim_std,
			},
			{
				"metric": "feature_dice_mean",
				"value": feature_dice_mean,
			},
			{
				"metric": "feature_dice_median",
				"value": feature_dice_median,
			},
			{
				"metric": "feature_dice_std",
				"value": feature_dice_std,
			},
			{
				"metric": "feature_distance_mean",
				"value": feature_distance_mean,
			},
			{
				"metric": "feature_distance_median",
				"value": feature_distance_median,
			},
			{
				"metric": "feature_distance_std",
				"value": feature_distance_std,
			},
			{
				"metric": "avg_curve_diff_abs",
				"value": avg_curve_diff_abs,
			},
			{
				"metric": "num_validation_tars_matched",
				"value": len(matched_tars),
			},
			{
				"metric": "num_samples_used",
				"value": n_samples_used,
			},
			{
				"metric": "num_samples_skipped",
				"value": n_samples_skipped,
			},
			{
				"metric": "num_ssim_rows",
				"value": len(ssim_df),
			},
			{
				"metric": "num_corr_rows",
				"value": len(corr_df),
			},
			{
				"metric": "num_feature_rows",
				"value": len(feature_df),
			},
			{
				"metric": "num_missing_original_tars",
				"value": len(missing_original),
			},
			{
				"metric": "num_missing_translated_tars",
				"value": len(missing_translated),
			},
		]
	)
	summary_path = os.path.join(run_output_dir, "summary_stats.csv")
	summary_df.to_csv(summary_path, index=False)

	# Match styling with sanity panels c/d.
	simcolor = "#ed9d2c"
	dftcolor = "#2ca3cf"
	expcolor = "#de461c"
	ssim_color = "#4a4a4a"

	fig = plt.figure(figsize=(8.5, 4.0))
	gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
	ax_dist = fig.add_subplot(gs[0, 0])
	ax1 = fig.add_subplot(gs[0, 1])

	if len(ssim_df) > 0 or len(feature_df) > 0:
		x_density = np.linspace(0.0, 1.0, 401)
		if len(ssim_df) > 0:
			plot_kde_density(
				ax_dist,
				ssim_df["ssim"].to_numpy(dtype=float),
				color=ssim_color,
				label="SSIM",
				x_grid=x_density,
			)
		if len(feature_df) > 0:
			plot_kde_density(
				ax_dist,
				feature_df["feature_dice"].to_numpy(dtype=float),
				color=dftcolor,
				label="Feature Dice",
				x_grid=x_density,
			)

	ax_dist.set_xlabel(r"Similarity $s$")
	ax_dist.set_ylabel(r"Probability density $\rho(s)$")
	ax_dist.set_xlim(0.0, 1.0)
	ax_dist.set_ylim(bottom=0.0)
	ax_dist.legend(loc="best", frameon=False, ncol=2)
	ax_dist.tick_params(axis="both", which="both", direction="in", top=True, right=True)

	if len(corr_df) > 0:
		grouped = corr_df.groupby("pair_idx", as_index=False).agg(
			mean_original=("corr_original", "mean"),
			mean_translated=("corr_translated", "mean"),
			sem_original=("corr_original", sem),
			sem_translated=("corr_translated", sem),
		)
		x = grouped["pair_idx"].to_numpy(dtype=float)
		y_o = grouped["mean_original"].to_numpy(dtype=float)
		y_t = grouped["mean_translated"].to_numpy(dtype=float)
		e_o = grouped["sem_original"].to_numpy(dtype=float)
		e_t = grouped["sem_translated"].to_numpy(dtype=float)

		ax1.plot(
			x,
			y_o,
			marker="o",
			linewidth=1.6,
			label="Original",
			color=simcolor,
			markerfacecolor="none",
			markeredgecolor=simcolor,
		)
		ax1.errorbar(
			x,
			y_o,
			yerr=e_o,
			fmt="none",
			ecolor=simcolor,
			elinewidth=1.4,
			capsize=3,
			alpha=1.0,
			zorder=4,
		)
		ax1.fill_between(x, y_o - e_o, y_o + e_o, alpha=0.15, color=simcolor)

		ax1.plot(
			x,
			y_t,
			marker="p",
			linewidth=1.6,
			label="Translated",
			color=expcolor,
			markerfacecolor="none",
			markeredgecolor=expcolor,
		)
		ax1.errorbar(
			x,
			y_t,
			yerr=e_t,
			fmt="none",
			ecolor=expcolor,
			elinewidth=1.4,
			capsize=3,
			alpha=1.0,
			zorder=4,
		)
		ax1.fill_between(x, y_t - e_t, y_t + e_t, alpha=0.15, color=expcolor)
	ax1.set_xlabel(r"Slice index $i$")
	ax1.set_ylabel(r"Correlation $C(i, i+1)$")
	ax1.set_ylim(-1.05, 1.05)
	ax1.legend(loc="best", frameon=False, ncol=2)
	ax1.grid(alpha=0.2)
	ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True)

	ax_dist.text(-0.15, 1.03, "a", transform=ax_dist.transAxes, fontsize=16, ha="left", va="top")
	ax1.text(-0.15, 1.03, "b", transform=ax1.transAxes, fontsize=16, ha="left", va="top")

	fig.subplots_adjust(left=0.08, right=0.995, top=0.97, bottom=0.16, wspace=0.28)
	plt.savefig(os.path.join(run_output_dir, "content_preservation.png"), dpi=300)
	plt.savefig(os.path.join(run_output_dir, "content_preservation.svg"))
	plt.savefig(os.path.join(run_output_dir, "content_preservation.pdf"))
	plt.close(fig)

	print(f"Matched validation tars: {len(matched_tars)}")
	print(f"Missing original tars: {len(missing_original)}")
	print(f"Missing translated tars: {len(missing_translated)}")
	print(f"Samples used: {n_samples_used}")
	print(f"Samples skipped: {n_samples_skipped}")
	print(f"SSIM mean/median/std: {ssim_mean:.6f}, {ssim_median:.6f}, {ssim_std:.6f}")
	print(
		f"Feature Dice mean/median/std: {feature_dice_mean:.6f}, {feature_dice_median:.6f}, {feature_dice_std:.6f}"
	)
	print(
		f"Feature distance mean/median/std: {feature_distance_mean:.6f}, {feature_distance_median:.6f}, {feature_distance_std:.6f}"
	)
	print(f"Average abs curve difference: {avg_curve_diff_abs:.6f}")
	print(f"Wrote: {ssim_csv_path}")
	print(f"Wrote: {feature_csv_path}")
	print(f"Wrote: {corr_csv_path}")
	if args.feature_method == "blob_dark_bright" and os.path.isfile(blob_csv_path):
		print(f"Wrote: {blob_csv_path}")
	print(f"Wrote: {summary_path}")
	print(f"Wrote: {os.path.join(run_output_dir, 'content_preservation.png')}")


if __name__ == "__main__":
	main()

