import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.colorbar as mcolorbar
import io
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def display_overlay(original_img_file, mask_pil, alpha=0.5):
    original = Image.open(original_img_file).convert("RGB").resize((512, 512))
    mask_rgb = mask_pil.convert("RGB")
    blended = Image.blend(original, mask_rgb, alpha=alpha)
    return blended



def display_distribution_chart(class_distribution, class_colors_dict, class_labels):
    exclude_labels = ["background"]

    filtered_labels = []
    filtered_values = []
    filtered_colors = []

    name_to_id = {name: idx for idx, name in class_labels.items()}

    for label, value in class_distribution.items():
        if label.lower() not in exclude_labels:
            filtered_labels.append(label)
            filtered_values.append(value)

            label_id = name_to_id[label]
            color = np.array(class_colors_dict[label_id]) / 255.0
            filtered_colors.append(color)

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(filtered_labels, filtered_values, color=filtered_colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel("No. of pixels")
    ax.set_title("Damage Level Distribution")

    plt.xticks(rotation=30, ha='right')


    plt.tight_layout()
    return fig


# def merge_mask_with_image(base_img: str, rgba_mask: Image.Image) -> Image.Image:
#     base_rgba = Image.open(base_img).convert("RGBA")
#     if base_rgba.size != rgba_mask.size:
#         rgba_mask = rgba_mask.resize(base_rgba.size, Image.NEAREST)
#
#     base_rgba.paste(rgba_mask, (0, 0), rgba_mask)
#
#     return base_rgba.convert("RGB")


def generate_cam_visualization(image_source, grayscale_cam: np.ndarray) -> np.ndarray:
    raw_img = Image.open(image_source).convert('RGB')

    rgb_img_float = np.array(raw_img, dtype=np.float32) / 255.0
    height, width = rgb_img_float.shape[:2]

    cam_resized = cv2.resize(grayscale_cam, (width, height))

    visualization = show_cam_on_image(rgb_img_float, cam_resized, use_rgb=True)

    return visualization


from PIL import Image
import io


def merge_mask_with_image(base_image, mask_image, target_size=(512, 512)):
    if isinstance(base_image, (str, bytes, io.BytesIO)):
        base_rgba = Image.open(base_image).convert("RGBA")
    elif hasattr(base_image, "convert"):
        base_rgba = base_image.convert("RGBA")
    else:
        base_rgba = Image.open(base_image).convert("RGBA")

    if not hasattr(mask_image, "convert"):
        mask_rgba = Image.open(mask_image).convert("RGBA")
    else:
        mask_rgba = mask_image.convert("RGBA")

    if base_rgba.size != mask_rgba.size:
        mask_rgba = mask_rgba.resize(base_rgba.size, Image.Resampling.NEAREST)

    blended_image = Image.alpha_composite(base_rgba, mask_rgba)
    return blended_image.resize(target_size, Image.Resampling.LANCZOS)




def merge_high_contrast_heatmap(base_image, heatmap_rgba, dim_factor=0.4, target_size=(512, 512)):
    if not hasattr(base_image, "convert"):
        base = Image.open(base_image).convert("RGBA")
    else:
        base = base_image.convert("RGBA")

    enhancer = ImageEnhance.Brightness(base)
    base_dimmed = enhancer.enhance(dim_factor)

    mask = heatmap_rgba.convert("RGBA")

    if base_dimmed.size != mask.size:
        mask = mask.resize(base_dimmed.size, Image.Resampling.NEAREST)


    combined = Image.alpha_composite(base_dimmed, mask)

    return combined.resize(target_size, Image.Resampling.LANCZOS)



def to_heatmap(data_np: np.ndarray, alpha_strength: float = 0.5) -> Image.Image:
    cm = plt.get_cmap('magma')
    colored_data = cm(data_np)

    image_data = (colored_data * 255).astype(np.uint8)
    alpha_channel = np.where(data_np > 0, int(255 * alpha_strength), 0).astype(np.uint8)

    image_data[:, :, 3] = alpha_channel

    return Image.fromarray(image_data, mode="RGBA")


def get_heatmap_legend():
    fig, ax = plt.subplots(figsize=(6, 1.1))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    cb = mcolorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal'
    )

    cb.set_label('Predictive Entropy (Uncertainty Level)', fontsize=10, fontweight='bold', color='white')
    label_obj = cb.ax.xaxis.get_label()

    stroke = [path_effects.withStroke(linewidth=3, foreground='black')]
    label_obj.set_path_effects(stroke)

    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_xticklabels(['0.0 (Certain)', '0.25', '0.50', '0.75', '1.0 (Uncertain)'])

    for tick in cb.ax.get_xticklabels():
        tick.set_color('white')
        tick.set_path_effects(stroke)
        tick.set_fontsize(9)

    cb.outline.set_edgecolor('white')
    cb.outline.set_linewidth(1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return Image.open(buf)



def display_distribution_chart_statistics_page(disaster_info: dict):
    if not disaster_info:
        return None

    class_distribution = disaster_info.get("damage_distribution", {})
    exclude_labels = ["background", "un-classified"]

    filtered_labels = []
    filtered_values = []

    for label, value in class_distribution.items():
        if label.lower() not in exclude_labels:
            pretty_label = label.replace("-", " ").title()
            filtered_labels.append(pretty_label)
            filtered_values.append(value*100)

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(filtered_labels, filtered_values, color='#4FACFE', edgecolor='black', linewidth=0.5)

    ax.set_ylabel("Distribution (%)")
    ax.set_title("Damage Level Distribution")

    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    return fig



def display_building_pie_chart(disaster_info: dict):
    if not disaster_info:
        return None

    building_counts = disaster_info.get("building_counts", {})

    labels = []
    sizes = []
    legend_labels = []

    total_buildings = sum(count for count in building_counts.values() if count > 0)

    for label, count in building_counts.items():
        if count > 0:
            pretty_label = label.replace("-", " ").title()
            labels.append(pretty_label)
            sizes.append(count)

            legend_labels.append(f"{pretty_label}: {count}")

    if not sizes:
        return None

    random_colors = [np.random.rand(3, ) for _ in range(len(sizes))]

    fig, ax = plt.subplots(figsize=(7, 4))

    def custom_autopct(pct):
        val = int(round(pct * total_buildings / 100.0))
        return str(val) if pct >= 5.0 else ''

    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=random_colors,
        autopct=custom_autopct,
        startangle=140,
        textprops={'fontsize': 10, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}
    )

    ax.legend(
        wedges,
        legend_labels,
        title="Damage Levels & Counts",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    ax.set_title("Building Damage Counts")
    plt.tight_layout()

    return fig


def add_realistic_smoke(img, max_opacity=0.6, blur_kernel_size=61):
    aug_img = img.clone()
    _, h, w = aug_img.shape

    downscale_h, downscale_w = h // 8, w // 8
    noise = torch.rand((1, downscale_h, downscale_w))
    noise = torch.nn.functional.interpolate(
        noise.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
    ).squeeze(0)

    noise = TF.gaussian_blur(noise, kernel_size=[blur_kernel_size, blur_kernel_size], sigma=[15.0, 15.0])
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    alpha = noise * max_opacity
    smoke_color = 0.8
    aug_img = aug_img * (1 - alpha) + (smoke_color * alpha)

    return aug_img