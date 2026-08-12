import base64
import io

import numpy as np
import requests
from flask import Flask, jsonify, request
from PIL import Image
from sklearn.cluster import KMeans

app = Flask(__name__)

N_COLORS = 5
MAX_DIMENSION = 300  # downscale for speed, doesn't affect color accuracy meaningfully


def load_image(payload: dict) -> Image.Image:
    """Accepts either {'image_url': ...} or {'image_base64': ...}."""
    if "image_url" in payload:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ChromatureApp/1.0; +https://emkaan.com)"
        }
        resp = requests.get(payload["image_url"], headers=headers, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    elif "image_base64" in payload:
        raw = base64.b64decode(payload["image_base64"])
        img = Image.open(io.BytesIO(raw))
    else:
        raise ValueError("Request must include 'image_url' or 'image_base64'")
    return img.convert("RGB")


def downscale(img: Image.Image) -> Image.Image:
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
    return img


def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(round(rgb[0])), int(round(rgb[1])), int(round(rgb[2]))
    )


def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def stylize_vivid(palette: list, factor: float = 1.45) -> list:
    """Pushes each color's saturation up, away from its own gray value."""
    styled = []
    for c in palette:
        rgb = hex_to_rgb(c["hex"])
        gray = rgb.mean()
        rgb = gray + (rgb - gray) * factor
        rgb = np.clip(rgb, 0, 255)
        styled.append({"hex": rgb_to_hex(rgb), "pct": c["pct"]})
    return styled


def stylize_contrast(palette: list, factor: float = 1.35) -> list:
    """Pushes each color's brightness away from mid-gray, deepening shadows and lifting highlights."""
    styled = []
    for c in palette:
        rgb = hex_to_rgb(c["hex"])
        rgb = 127.5 + (rgb - 127.5) * factor
        rgb = np.clip(rgb, 0, 255)
        styled.append({"hex": rgb_to_hex(rgb), "pct": c["pct"]})
    return styled


def compute_saturation_weight(pixels: np.ndarray) -> np.ndarray:
    """Higher weight for more colorful/vivid pixels, lower for washed-out ones."""
    max_c = pixels.max(axis=1)
    min_c = pixels.min(axis=1)
    saturation = max_c - min_c  # 0-255
    # squared so vivid pixels dominate clustering much more strongly;
    # +1 floor avoids an all-zero weight vector on a fully grayscale image
    return saturation**2 + 1.0


def compute_contrast_weight(pixels: np.ndarray) -> np.ndarray:
    """Higher weight for pixels far from mid-gray (deep shadows and bright highlights)."""
    brightness = pixels.mean(axis=1)
    contrast = np.abs(brightness - 127.5)
    return contrast**2 + 1.0


def run_kmeans(pixels: np.ndarray, sample_weight: np.ndarray = None, seed: int = 42):
    """
    Clusters the photo's actual RGB pixels, optionally weighting some pixels
    more heavily so the resulting palette leans toward a particular visual
    character (e.g. more vivid, or more high-contrast). Percentages always
    reflect real, unweighted pixel counts (true image composition); only
    the *clustering itself* is influenced by the weights, which is what
    changes which colors get picked out.
    """
    km = KMeans(n_clusters=N_COLORS, random_state=seed, n_init=4)
    labels = km.fit_predict(pixels, sample_weight=sample_weight)

    palette = []
    for cluster_id in range(N_COLORS):
        mask = labels == cluster_id
        count = mask.sum()
        if count == 0:
            continue
        avg_rgb = pixels[mask].mean(axis=0)
        pct = count / len(pixels) * 100
        palette.append({"hex": rgb_to_hex(avg_rgb), "pct": round(float(pct), 1)})

    # Normalize rounding so percentages sum to exactly 100
    total = sum(c["pct"] for c in palette)
    if total != 0 and palette:
        diff = round(100 - total, 1)
        palette[0]["pct"] = round(palette[0]["pct"] + diff, 1)

    palette.sort(key=lambda c: c["pct"], reverse=True)
    return palette


@app.route("/extract-colors", methods=["POST"])
def extract_colors():
    try:
        payload = request.get_json(force=True)
        img = load_image(payload)
        img = downscale(img)
        pixels = np.array(img).reshape(-1, 3).astype(np.float64)

        # Three passes: different pixel weighting changes which colors get
        # grouped together (helps on photos with real variety), and a style
        # shift on top guarantees the 3 are visibly distinct even on photos
        # dominated by one narrow color range (e.g. plain green foliage),
        # where reweighting alone isn't always enough to separate them.
        chromature_1 = run_kmeans(pixels, sample_weight=None)  # balanced/accurate
        chromature_2 = stylize_vivid(
            run_kmeans(pixels, sample_weight=compute_saturation_weight(pixels))
        )
        chromature_3 = stylize_contrast(
            run_kmeans(pixels, sample_weight=compute_contrast_weight(pixels))
        )

        return jsonify(
            {
                "chromature_1": chromature_1,
                "chromature_2": chromature_2,
                "chromature_3": chromature_3,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
