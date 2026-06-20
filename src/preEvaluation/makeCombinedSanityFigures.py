#!/usr/bin/env python

# Run it
# python makeCombinedSanityFigures.py --analysis-dir /scratch/phys/sin/Jie/Github/StyleTransAugment/results/dataWideStyletransAnalysis/L20_L1 --feature-figure --feature-method blob_dark_bright --sample-ids 0,1,2,3,4,5,6,7 --name-prefix sample_sanity_check_set

import argparse
import io
import os
import re
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import sobel

try:
    import cv2
except Exception:
    cv2 = None

DEFAULT_ORIGINAL_ROOT = "/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-bilayer-FB/Water-bilayer_FB_Ref"
DEFAULT_TRANSLATED_ROOT = "/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-bilayer-FB/Water-bilayer_FB_PPAFM2Exp_CoAll_L20_L1_Elatest"
DEFAULT_ANALYSIS_DIR = "results/dataWideStyletransAnalysis"
DEFAULT_TAR_NAME = "Water-bilayer-K-1_val_0.tar"
SLICE_MEMBER_PATTERN = re.compile(r"^(\d+)\.(\d+)\.png$")
AFM_CMAP = "inferno"

TEXT_BBOX = {
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.5,
    "pad": 0.3,
}

simcolor = '#ed9d2c'
dftcolor = '#2ca3cf'
expcolor = '#de461c'
bg07color = '#479FB1'
bv17color = '#6E7CBC'


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (r, g, b)


DFT_BLOB_RGB = hex_to_rgb(dftcolor)

def parse_int_list(raw_text):
    values = []
    for part in raw_text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def discover_sample_ids(tar_obj):
    sample_ids = set()
    for member in tar_obj.getmembers():
        if not member.isfile():
            continue
        base = os.path.basename(member.name)
        m = SLICE_MEMBER_PATTERN.match(base)
        if m is None:
            continue
        sample_ids.add(int(m.group(1)))
    return sorted(sample_ids)


def read_slice_from_tar(tar_obj, sample_id, slice_idx):
    member_name = f"{sample_id}.{slice_idx:02d}.png"
    member_file = tar_obj.extractfile(member_name)
    if member_file is None:
        raise FileNotFoundError(member_name)
    return np.array(Image.open(io.BytesIO(member_file.read())).convert("L"), dtype=np.float32)


def get_sample_curves(ssim_df, corr_df, tar_name, sample_id):
    ssim_sub = ssim_df[(ssim_df["tar_file"] == tar_name) & (ssim_df["sample_id"] == sample_id)].copy()
    corr_sub = corr_df[(corr_df["tar_file"] == tar_name) & (corr_df["sample_id"] == sample_id)].copy()
    return ssim_sub.sort_values("slice_idx"), corr_sub.sort_values("pair_idx")


def get_sample_metric_curve(metric_df, tar_name, sample_id, metric_col):
    metric_sub = metric_df[(metric_df["tar_file"] == tar_name) & (metric_df["sample_id"] == sample_id)].copy()
    cols = ["slice_idx", metric_col]
    return metric_sub[cols].sort_values("slice_idx")


def gradient_preprocess_image(img):
    gx = sobel(img.astype(np.float64), axis=1, mode="reflect")
    gy = sobel(img.astype(np.float64), axis=0, mode="reflect")
    grad = np.sqrt(gx * gx + gy * gy)
    return grad / (float(np.max(grad)) + 1e-12)


def feature_map_image(img, top_percent=5.0, use_gradient=True):
    x = img.astype(np.float64)
    if use_gradient:
        gx = sobel(x, axis=1, mode="reflect")
        gy = sobel(x, axis=0, mode="reflect")
        x = np.sqrt(gx * gx + gy * gy)

    top_percent = float(np.clip(top_percent, 0.01, 100.0))
    threshold = np.percentile(x, 100.0 - top_percent)
    mask = (x >= threshold).astype(np.float32)
    return mask


def to_u8_image(img):
    x = img.astype(np.float64)
    x = np.clip(x, 0.0, 255.0 if x.max() > 1.0 else 1.0)
    if x.max() <= 1.0:
        x = x * 255.0
    return np.round(x).astype(np.uint8)


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


def draw_blob_circle(rgb, center, radius, color, thickness=2, dotted=False):
    if not dotted:
        cv2.circle(rgb, center, max(1, int(radius)), color, thickness, lineType=cv2.LINE_8)
        return

    # Draw dotted style as short arc segments around the circle.
    r = max(1, int(radius))
    for start in range(0, 360, 24):
        end = start + 12
        cv2.ellipse(
            rgb,
            center,
            (r, r),
            0,
            start,
            end,
            color,
            thickness,
            lineType=cv2.LINE_8,
        )


def make_blob_annotated_rgb(gray_u8, dark_blobs, bright_blobs, dark_radius_scale=1.8, bright_radius_scale=1.8):
    rgb = np.repeat(gray_u8[..., None], 3, axis=2).astype(np.uint8)
    for b in dark_blobs:
        r = int(round(float(dark_radius_scale) * float(b["radius"])))
        c = (int(round(float(b["x"]))), int(round(float(b["y"]))))
        draw_blob_circle(rgb, c, r, DFT_BLOB_RGB, thickness=3, dotted=False)
    for b in bright_blobs:
        r = int(round(float(bright_radius_scale) * float(b["radius"])))
        c = (int(round(float(b["x"]))), int(round(float(b["y"]))))
        draw_blob_circle(rgb, c, r, DFT_BLOB_RGB, thickness=3, dotted=True)
    return rgb


def extract_blob_feature_map_and_mark(img, cfg):
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for feature-method 'blob_dark_bright'.")

    gray_u8 = to_u8_image(img)
    dark_background = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=float(cfg["dark_sigma"]), sigmaY=float(cfg["dark_sigma"]))
    bright_background = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=float(cfg["bright_sigma"]), sigmaY=float(cfg["bright_sigma"]))
    dark_enhanced = cv2.subtract(dark_background, gray_u8)
    bright_enhanced = cv2.subtract(gray_u8, bright_background)

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

    dark_blobs = select_dark_holes(dark_candidates, score_ratio=cfg["dark_score_ratio"], max_blobs=cfg["dark_max_blobs"])
    bright_blobs = select_bright_blobs(
        bright_candidates,
        score_ratio=cfg["bright_score_ratio"],
        max_blobs=cfg["bright_max_blobs"],
    )
    dark_mask = blobs_to_mask(gray_u8.shape, dark_blobs, radius_scale=cfg["dark_radius_scale"])
    bright_mask = blobs_to_mask(gray_u8.shape, bright_blobs, radius_scale=cfg["bright_radius_scale"])
    combined_mask = np.logical_or(dark_mask, bright_mask)
    annotated = make_blob_annotated_rgb(
        gray_u8,
        dark_blobs,
        bright_blobs,
        dark_radius_scale=cfg["dark_radius_scale"],
        bright_radius_scale=cfg["bright_radius_scale"],
    )
    return combined_mask.astype(np.float32), annotated


def make_cached_blob_annotated_rgb(gray_u8, dark_blobs, bright_blobs):
    rgb = np.repeat(gray_u8[..., None], 3, axis=2).astype(np.uint8)
    for b in dark_blobs:
        r = int(round(float(b["radius_draw"])))
        c = (int(round(float(b["x"]))), int(round(float(b["y"]))))
        draw_blob_circle(rgb, c, r, DFT_BLOB_RGB, thickness=2, dotted=False)
    for b in bright_blobs:
        r = int(round(float(b["radius_draw"])))
        c = (int(round(float(b["x"]))), int(round(float(b["y"]))))
        draw_blob_circle(rgb, c, r, DFT_BLOB_RGB, thickness=2, dotted=True)
    return rgb


def cached_blob_mark_image(blob_df, tar_name, sample_id, slice_idx, domain, fallback_img, blob_cfg):
    if blob_df is None or blob_df.empty:
        _, mark = extract_blob_feature_map_and_mark(fallback_img, blob_cfg)
        return mark

    sub = blob_df[
        (blob_df["tar_file"] == tar_name)
        & (blob_df["sample_id"] == sample_id)
        & (blob_df["slice_idx"] == slice_idx)
        & (blob_df["domain"] == domain)
    ].copy()
    if sub.empty:
        _, mark = extract_blob_feature_map_and_mark(fallback_img, blob_cfg)
        return mark

    if "rank" in sub.columns:
        sub = sub.sort_values("rank")
    dark = sub[sub["blob_type"] == "dark"]
    bright = sub[sub["blob_type"] == "bright"]
    gray_u8 = to_u8_image(fallback_img)
    return make_cached_blob_annotated_rgb(
        gray_u8,
        dark.to_dict("records"),
        bright.to_dict("records"),
    )


def add_image_text(ax, x, y, text, **kwargs):
    default_kwargs = {
        "transform": ax.transAxes,
        "va": "top",
        "ha": "left",
        "fontsize": 7.2,
        "color": "black",
        "bbox": TEXT_BBOX,
    }
    default_kwargs.update(kwargs)
    ax.text(x, y, text, **default_kwargs)


def build_combined_figure(orig_tar, trans_tar, tar_name, sample_id, slice_ids, ssim_sub, corr_sub, out_png, out_svg, out_pdf):
    if len(slice_ids) != 14:
        raise ValueError("This layout expects exactly 14 slice indices.")

    top_images_orig = []
    top_images_trans = []
    for slice_idx in slice_ids:
        top_images_orig.append(read_slice_from_tar(orig_tar, sample_id, slice_idx))
        top_images_trans.append(read_slice_from_tar(trans_tar, sample_id, slice_idx))

    stacked = np.stack(top_images_orig + top_images_trans, axis=0)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))

    first_block = slice_ids[:7]
    second_block = slice_ids[7:]

    fig = plt.figure(figsize=(8.5, 8))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[3.9, 1.25], hspace=0.18)
    gs_top = gs_main[0].subgridspec(4, 7, wspace=0.01, hspace=0.03)
    gs_bottom = gs_main[1].subgridspec(1, 2, wspace=0.18)

    row_specs = [
        (0, "Original", first_block, top_images_orig, False),
        (1, "Translated", first_block, top_images_trans, True),
        (2, "Original", second_block, top_images_orig, False),
        (3, "Translated", second_block, top_images_trans, True),
    ]

    for row_idx, row_label, row_slices, row_images, annotate_ssim in row_specs:
        for col, slice_idx in enumerate(row_slices):
            img_idx = slice_ids.index(slice_idx)
            ax = fig.add_subplot(gs_top[row_idx, col])
            ax.imshow(row_images[img_idx], cmap=AFM_CMAP, vmin=vmin, vmax=vmax)
            ax.axis("off")
            if row_idx in [0, 2]:
                ax.text(0.02, 0.95, f"i={slice_idx:02d}", transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color="white")

            if col == 0:
                label_color = simcolor if "Orig" in row_label else expcolor
                ax.text(
                    -0.01,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                    color=label_color,
                )
			

            if annotate_ssim:
                ssim_at_slice = ssim_sub[ssim_sub["slice_idx"] == slice_idx]
                if not ssim_at_slice.empty:
                    val = float(ssim_at_slice.iloc[0]["ssim"])
                    ax.text(0.02, 0.95, f"SSIM={val:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color="white")

    ax_ssim = fig.add_subplot(gs_bottom[0, 0])
    ax_corr = fig.add_subplot(gs_bottom[0, 1])

    ax_ssim.plot(ssim_sub["slice_idx"], ssim_sub["ssim"], marker="o", linewidth=1.8, color="tab:blue")
    ax_ssim.set_ylim(0.0, 1.05)
    ax_ssim.set_xlabel("Slice index")
    ax_ssim.set_ylabel("SSIM")
    #ax_ssim.set_title("SSIM vs slice index")
    ax_ssim.grid(alpha=0.25)

    ax_corr.plot(corr_sub["pair_idx"], corr_sub["corr_original"], marker="o", linewidth=1.6, label="Original", color="tab:green")
    ax_corr.plot(corr_sub["pair_idx"], corr_sub["corr_translated"], marker="o", linewidth=1.6, label="Translated", color="tab:orange")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.set_xlabel("Adjacent pair index (i vs i+1)")
    ax_corr.set_ylabel("Correlation")
    #ax_corr.set_title("Adjacent-slice correlation")
    ax_corr.grid(alpha=0.25)
    ax_corr.legend(loc="best")

    #fig.suptitle(f"{tar_name} | sample {sample_id}", fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.07)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_gradient_combined_figure(orig_tar, trans_tar, tar_name, sample_id, slice_ids, gssim_sub, corr_sub, out_png, out_svg, out_pdf):
    if len(slice_ids) != 14:
        raise ValueError("This layout expects exactly 14 slice indices.")

    top_images_orig = []
    top_images_trans = []
    top_images_orig_grad = []
    top_images_trans_grad = []
    for slice_idx in slice_ids:
        img_o = read_slice_from_tar(orig_tar, sample_id, slice_idx)
        img_t = read_slice_from_tar(trans_tar, sample_id, slice_idx)
        top_images_orig.append(img_o)
        top_images_trans.append(img_t)
        top_images_orig_grad.append(gradient_preprocess_image(img_o))
        top_images_trans_grad.append(gradient_preprocess_image(img_t))

    stacked_raw = np.stack(top_images_orig + top_images_trans, axis=0)
    raw_vmin = float(np.min(stacked_raw))
    raw_vmax = float(np.max(stacked_raw))

    stacked_grad = np.stack(top_images_orig_grad + top_images_trans_grad, axis=0)
    grad_vmin = float(np.min(stacked_grad))
    grad_vmax = float(np.max(stacked_grad))

    first_block = slice_ids[:7]
    second_block = slice_ids[7:]

    fig = plt.figure(figsize=(8.5, 13.0))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[7.3, 1.25], hspace=0.18)
    gs_top = gs_main[0].subgridspec(8, 7, wspace=0.01, hspace=0.03)
    gs_bottom = gs_main[1].subgridspec(1, 2, wspace=0.18)

    row_specs = [
        (0, "Original", first_block, top_images_orig, raw_vmin, raw_vmax, False),
        (1, "Translated", first_block, top_images_trans, raw_vmin, raw_vmax, True),
        (2, "Original", second_block, top_images_orig, raw_vmin, raw_vmax, False),
        (3, "Translated", second_block, top_images_trans, raw_vmin, raw_vmax, True),
        (4, "Orig grad", first_block, top_images_orig_grad, grad_vmin, grad_vmax, False),
        (5, "Trans grad", first_block, top_images_trans_grad, grad_vmin, grad_vmax, True),
        (6, "Orig grad", second_block, top_images_orig_grad, grad_vmin, grad_vmax, False),
        (7, "Trans grad", second_block, top_images_trans_grad, grad_vmin, grad_vmax, True),
    ]

    for row_idx, row_label, row_slices, row_images, row_vmin, row_vmax, annotate_gssim in row_specs:
        for col, slice_idx in enumerate(row_slices):
            img_idx = slice_ids.index(slice_idx)
            ax = fig.add_subplot(gs_top[row_idx, col])
            ax.imshow(row_images[img_idx], cmap=AFM_CMAP, vmin=row_vmin, vmax=row_vmax)
            ax.axis("off")
            if row_idx in [0, 2, 4, 6]:
                ax.text(0.02, 0.95, f"i={slice_idx:02d}", transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color="white")

            if col == 0:
                ax.text(-0.01, 0.5, row_label, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8.5)

            if annotate_gssim:
                gssim_at_slice = gssim_sub[gssim_sub["slice_idx"] == slice_idx]
                if not gssim_at_slice.empty:
                    val = float(gssim_at_slice.iloc[0]["gssim"])
                    ax.text(0.02, 0.95, f"GSSIM={val:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=7.2, color="white")

    ax_gssim = fig.add_subplot(gs_bottom[0, 0])
    ax_corr = fig.add_subplot(gs_bottom[0, 1])

    ax_gssim.plot(gssim_sub["slice_idx"], gssim_sub["gssim"], marker="o", linewidth=1.8, color="tab:green")
    ax_gssim.set_ylim(0.0, 1.05)
    ax_gssim.set_xlabel("Slice index")
    ax_gssim.set_ylabel("Gradient-SSIM")
    ax_gssim.grid(alpha=0.25)

    ax_corr.plot(corr_sub["pair_idx"], corr_sub["corr_original"], marker="o", linewidth=1.6, label="Original", color="tab:green")
    ax_corr.plot(corr_sub["pair_idx"], corr_sub["corr_translated"], marker="o", linewidth=1.6, label="Translated", color="tab:orange")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.set_xlabel("Adjacent pair index (i vs i+1)")
    ax_corr.set_ylabel("Correlation")
    ax_corr.grid(alpha=0.25)
    ax_corr.legend(loc="bottom right", frameon=False)

    fig.subplots_adjust(left=0.02, right=0.995, top=0.975, bottom=0.045)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_feature_combined_figure(
    orig_tar,
    trans_tar,
    tar_name,
    sample_id,
    slice_ids,
    ssim_sub,
    feature_sub,
    corr_sub,
    out_png,
    out_svg,
    out_pdf,
    feature_top_percent,
    feature_use_gradient,
    feature_method,
    blob_cfg,
    blob_df=None,
):
    if len(slice_ids) != 14:
        raise ValueError("This layout expects exactly 14 slice indices.")

    top_images_orig = []
    top_images_trans = []
    top_feat_orig = []
    top_feat_trans = []
    for slice_idx in slice_ids:
        img_o = read_slice_from_tar(orig_tar, sample_id, slice_idx)
        img_t = read_slice_from_tar(trans_tar, sample_id, slice_idx)
        top_images_orig.append(img_o)
        top_images_trans.append(img_t)
        if feature_method == "blob_dark_bright":
            mark_o = cached_blob_mark_image(
                blob_df,
                tar_name,
                sample_id,
                slice_idx,
                "original",
                img_o,
                blob_cfg,
            )
            mark_t = cached_blob_mark_image(
                blob_df,
                tar_name,
                sample_id,
                slice_idx,
                "translated",
                img_t,
                blob_cfg,
            )
            top_feat_orig.append(mark_o)
            top_feat_trans.append(mark_t)
        else:
            mask_o = feature_map_image(img_o, top_percent=feature_top_percent, use_gradient=feature_use_gradient)
            mask_t = feature_map_image(img_t, top_percent=feature_top_percent, use_gradient=feature_use_gradient)
            top_feat_orig.append(mask_o)
            top_feat_trans.append(mask_t)

    stacked = np.stack(top_images_orig + top_images_trans, axis=0)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))

    first_block = slice_ids[:7]
    second_block = slice_ids[7:]

    fig = plt.figure(figsize=(8.5, 10.4))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[6.5, 1.25], hspace=0.08)
    gs_top = gs_main[0].subgridspec(8, 7, wspace=0.01, hspace=0.03)
    gs_bottom = gs_main[1].subgridspec(1, 2, wspace=0.18)

    row_specs = [
        (0, "Original", first_block, top_images_orig, False, False),
        (1, "Translated", first_block, top_images_trans, True, False),
        (2, "Original", second_block, top_images_orig, False, False),
        (3, "Translated", second_block, top_images_trans, True, False),
        (4, "Orig feat", first_block, top_feat_orig, False, True),
        (5, "Trans feat", first_block, top_feat_trans, True, True),
        (6, "Orig feat", second_block, top_feat_orig, False, True),
        (7, "Trans feat", second_block, top_feat_trans, True, True),
    ]

    for row_idx, row_label, row_slices, row_images, annotate_feature, is_feature_row in row_specs:
        for col, slice_idx in enumerate(row_slices):
            img_idx = slice_ids.index(slice_idx)
            ax = fig.add_subplot(gs_top[row_idx, col])
            if is_feature_row:
                if feature_method == "blob_dark_bright":
                    ax.imshow(row_images[img_idx], interpolation="nearest")
                else:
                    ax.imshow(row_images[img_idx], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            else:
                ax.imshow(row_images[img_idx], cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if row_idx in [0, 2, 4, 6]:
                add_image_text(ax, 0.02, 0.95, f"i={slice_idx:02d}")

            if col == 0:
                label_color = simcolor if row_label in ["Original", "Orig feat"] else expcolor
                ax.text(
                    -0.01,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                    color=label_color,
                )

            if annotate_feature and row_idx in [1, 3]:
                ssim_at_slice = ssim_sub[ssim_sub["slice_idx"] == slice_idx]
                if not ssim_at_slice.empty:
                    sval = float(ssim_at_slice.iloc[0]["ssim"])
                    add_image_text(ax, 0.02, 0.95, rf"$\mathrm{{SSIM}}(u_i,\tilde{{v}}_i)={sval:.2f}$")

            if annotate_feature and row_idx in [5, 7]:
                feat_at_slice = feature_sub[feature_sub["slice_idx"] == slice_idx]
                if not feat_at_slice.empty:
                    fdice = float(feat_at_slice.iloc[0]["feature_dice"])
                    add_image_text(ax, 0.02, 0.95, rf"$\mathrm{{Dice}}(u_i^{{f}},\tilde{{v}}_i^{{f}})={fdice:.2f}$")

    ax_feat = fig.add_subplot(gs_bottom[0, 0])
    ax_corr = fig.add_subplot(gs_bottom[0, 1])

    ax_feat.plot(
        ssim_sub["slice_idx"],
        ssim_sub["ssim"],
        marker="s",
        linewidth=1.6,
        color="#4a4a4a",
        markerfacecolor="none",
        markeredgecolor="#4a4a4a",
        label="SSIM",
    )
    ax_feat.plot(
        feature_sub["slice_idx"],
        feature_sub["feature_dice"],
        marker="o",
        linewidth=1.6,
        color=dftcolor,
        markerfacecolor="none",
        markeredgecolor=dftcolor,
        label="Feature Dice",
    )
    ax_feat.set_ylim(0.0, 1.05)
    ax_feat.set_xlabel("Slice index")
    ax_feat.set_ylabel("Similarity")
    ax_feat.grid(alpha=0.25)
    ax_feat.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax_feat.legend(loc="best", fontsize=8, frameon=False, ncol=2)

    ax_corr.plot(
        corr_sub["pair_idx"],
        corr_sub["corr_original"],
        marker="o",
        linewidth=1.6,
        label="Original",
        color=simcolor,
        markerfacecolor="none",
        markeredgecolor=simcolor,
    )
    ax_corr.plot(
        corr_sub["pair_idx"],
        corr_sub["corr_translated"],
        marker="p",
        linewidth=1.6,
        label="Translated",
        color=expcolor,
        markerfacecolor="none",
        markeredgecolor=expcolor,
    )
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.set_xlabel("Adjacent pair index (i vs i+1)")
    ax_corr.set_ylabel("Correlation")
    ax_corr.grid(alpha=0.25)
    ax_corr.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax_corr.legend(loc="lower right", frameon=False, ncol=2)

    # Panel labels for the four sub-panels in the combined figure.
    fig.text(0.03, 0.97, "a", color="#000000", fontsize=20, va="top", ha="left")
    fig.text(0.03, 0.60, "b", color="#000000", fontsize=20, va="top", ha="left")
    fig.text(0.03, 0.23, "c", color="#000000", fontsize=20, va="top", ha="left")
    fig.text(0.50, 0.23, "d", color="#000000", fontsize=20, va="top", ha="left")

    fig.subplots_adjust(left=0.055, right=0.995, top=0.965, bottom=0.055)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create combined sanity figures: top AFM comparison + bottom SSIM/correlation trends."
    )
    parser.add_argument("--tar-name", default=DEFAULT_TAR_NAME)
    parser.add_argument("--original-root", default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--translated-root", default=DEFAULT_TRANSLATED_ROOT)
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--sample-ids", default="0,1,2,3,4,5,6,7", help="Comma-separated sample IDs")
    parser.add_argument("--slices", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13", help="Comma-separated 14 slice indices for top panel")
    parser.add_argument("--name-prefix", default="sample_sanity_check_set")
    parser.add_argument("--gradient-figure", action="store_true", help="Create separate Gradient-SSIM sanity figures with preprocessed image rows")
    parser.add_argument("--gradient-name-suffix", default="_gradient", help="Suffix appended to output filename stem in --gradient-figure mode")
    parser.add_argument("--feature-figure", action="store_true", help="Create separate feature-metric sanity figures")
    parser.add_argument("--feature-name-suffix", default="_feature", help="Suffix appended to output filename stem in --feature-figure mode")
    parser.add_argument("--feature-top-percent", type=float, default=5.0, help="Top percentile for feature mask visualization")
    parser.add_argument("--feature-no-gradient", action="store_true", help="Use intensity-based top-percentile feature map instead of gradient-based")
    parser.add_argument(
        "--feature-method",
        default="blob_dark_bright",
        choices=["legacy_percentile", "blob_dark_bright"],
        help="Feature map method used for feature rows",
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

    if args.gradient_figure and args.feature_figure:
        raise ValueError("Use either --gradient-figure or --feature-figure, not both")

    analysis_dir = Path(args.analysis_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ssim_csv = analysis_dir / "ssim_values.csv"
    gssim_csv = analysis_dir / "gradient_ssim_values.csv"
    feature_csv = analysis_dir / "feature_mask_metrics.csv"
    corr_csv = analysis_dir / "adjacent_slice_correlations.csv"
    blob_csv = analysis_dir / "blob_features.csv"
    if not ssim_csv.is_file():
        raise FileNotFoundError(f"Missing metrics file: {ssim_csv}")
    if args.gradient_figure and not gssim_csv.is_file():
        raise FileNotFoundError(f"Missing gradient metrics file: {gssim_csv}")
    if args.feature_figure and not feature_csv.is_file():
        raise FileNotFoundError(f"Missing feature metrics file: {feature_csv}")
    if not corr_csv.is_file():
        raise FileNotFoundError(f"Missing metrics file: {corr_csv}")

    ssim_df = pd.read_csv(ssim_csv)
    gssim_df = pd.read_csv(gssim_csv) if args.gradient_figure else None
    feature_df = pd.read_csv(feature_csv) if args.feature_figure else None
    corr_df = pd.read_csv(corr_csv)
    blob_df = pd.read_csv(blob_csv) if (args.feature_figure and blob_csv.is_file()) else None

    sample_ids = parse_int_list(args.sample_ids)
    slice_ids = parse_int_list(args.slices)
    if not sample_ids:
        raise ValueError("No sample IDs provided.")
    if not slice_ids:
        raise ValueError("No slice indices provided.")

    orig_tar_path = Path(args.original_root) / args.tar_name
    trans_tar_path = Path(args.translated_root) / args.tar_name
    if not orig_tar_path.is_file():
        raise FileNotFoundError(f"Missing original tar: {orig_tar_path}")
    if not trans_tar_path.is_file():
        raise FileNotFoundError(f"Missing translated tar: {trans_tar_path}")

    with tarfile.open(orig_tar_path, "r") as orig_tar, tarfile.open(trans_tar_path, "r") as trans_tar:
        common_samples = sorted(set(discover_sample_ids(orig_tar)).intersection(discover_sample_ids(trans_tar)))
        for idx, sample_id in enumerate(sample_ids, start=1):
            if sample_id not in common_samples:
                print(f"Skip sample {sample_id}: not present in both tars")
                continue

            ssim_sub, corr_sub = get_sample_curves(ssim_df, corr_df, args.tar_name, sample_id)
            if args.gradient_figure:
                gssim_sub = get_sample_metric_curve(gssim_df, args.tar_name, sample_id, "gssim")
                if gssim_sub.empty or corr_sub.empty:
                    print(f"Skip sample {sample_id}: missing gradient/correlation rows")
                    continue
                out_png = analysis_dir / f"{args.name_prefix}{idx}{args.gradient_name_suffix}.png"
                out_svg = analysis_dir / f"{args.name_prefix}{idx}{args.gradient_name_suffix}.svg"
                out_pdf = analysis_dir / f"{args.name_prefix}{idx}{args.gradient_name_suffix}.pdf"
                build_gradient_combined_figure(
                    orig_tar,
                    trans_tar,
                    args.tar_name,
                    sample_id,
                    slice_ids,
                    gssim_sub,
                    corr_sub,
                    out_png,
                    out_svg,
                    out_pdf,
                )
            elif args.feature_figure:
                feature_sub = feature_df[(feature_df["tar_file"] == args.tar_name) & (feature_df["sample_id"] == sample_id)].copy().sort_values("slice_idx")
                if ssim_sub.empty or feature_sub.empty or corr_sub.empty:
                    print(f"Skip sample {sample_id}: missing feature/correlation rows")
                    continue
                feature_method = args.feature_method
                feature_use_gradient = (not args.feature_no_gradient)
                if args.feature_no_gradient and feature_method == "blob_dark_bright":
                    feature_method = "legacy_percentile"
                    feature_use_gradient = False
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
                out_png = analysis_dir / f"{args.name_prefix}{idx}{args.feature_name_suffix}.png"
                out_svg = analysis_dir / f"{args.name_prefix}{idx}{args.feature_name_suffix}.svg"
                out_pdf = analysis_dir / f"{args.name_prefix}{idx}{args.feature_name_suffix}.pdf"
                build_feature_combined_figure(
                    orig_tar,
                    trans_tar,
                    args.tar_name,
                    sample_id,
                    slice_ids,
                    ssim_sub,
                    feature_sub,
                    corr_sub,
                    out_png,
                    out_svg,
                    out_pdf,
                    args.feature_top_percent,
                    feature_use_gradient,
                    feature_method,
                    blob_cfg,
                    blob_df=blob_df,
                )
            else:
                if ssim_sub.empty or corr_sub.empty:
                    print(f"Skip sample {sample_id}: missing metric rows")
                    continue

                out_png = analysis_dir / f"{args.name_prefix}{idx}.png"
                out_svg = analysis_dir / f"{args.name_prefix}{idx}.svg"
                out_pdf = analysis_dir / f"{args.name_prefix}{idx}.pdf"
                build_combined_figure(
                    orig_tar,
                    trans_tar,
                    args.tar_name,
                    sample_id,
                    slice_ids,
                    ssim_sub,
                    corr_sub,
                    out_png,
                    out_svg,
                    out_pdf,
                )
            print(out_png)
            print(out_svg)
            print(out_pdf)


if __name__ == "__main__":
    main()
