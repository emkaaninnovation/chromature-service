import base64
import io
import colorsys

import numpy as np
import requests
from flask import Flask, jsonify, request
from PIL import Image
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

app = Flask(__name__)

N_COLORS = 5
MAX_DIMENSION = 300  # downscale for speed, doesn't affect color accuracy meaningfully


def load_image(payload: dict) -> Image.Image:
    """Accepts either {'image_url': ...} or {'image_base64': ...}."""
    if "image_url" in payload:
        resp = requests.get(payload["image_url"], timeout=15)
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


def rgb_to_hsv_features(pixels: np.ndarray) -> np.ndarray:
    hsv = np.array(
        [colorsys.rgb_to_hsv(*(p / 255.0)) for p in pixels]
    )
    # scale hue to 0-255 range so it carries comparable weight to s,v in k-means distance
    hsv[:, 0] *= 255.0
    hsv[:, 1] *= 255.0
    hsv[:, 2] *= 255.0
    return hsv


def rgb_to_lab_features(pixels: np.ndarray) -> np.ndarray:
    # rgb2lab expects an image-shaped array in 0-1 range
    normalized = (pixels / 255.0).reshape(-1, 1, 3)
    lab = rgb2lab(normalized).reshape(-1, 3)
    return lab


def run_kmeans(pixels: np.ndarray, space: str, seed: int = 42):
    """
    Clusters the photo's pixels in a given color space, then reports each
    cluster's actual average RGB color (not the raw space's center, which
    keeps output colors always valid/displayable).

    Clustering in a different color space genuinely changes which pixels
    get grouped together, so this reliably produces 3 different-looking
    chromatures from one photo, instead of relying on random seeds alone
    (which often converge to near-identical results on real photos).

    space: 'rgb', 'hsv', or 'lab'
    """
    if space == "rgb":
        features = pixels
    elif space == "hsv":
        features = rgb_to_hsv_features(pixels)
    elif space == "lab":
        features = rgb_to_lab_features(pixels)
    else:
        raise ValueError(f"Unknown color space: {space}")

    km = KMeans(n_clusters=N_COLORS, random_state=seed, n_init=4)
    labels = km.fit_predict(features)

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

        # Three passes in three different color spaces. This is what
        # guarantees genuinely different groupings (and therefore different
        # colors/percentages) from the same photo, rather than hoping
        # different random seeds happen to diverge.
        chromature_1 = run_kmeans(pixels, space="rgb")
        chromature_2 = run_kmeans(pixels, space="hsv")
        chromature_3 = run_kmeans(pixels, space="lab")

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
