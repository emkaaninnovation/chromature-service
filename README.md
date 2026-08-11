# Chromature color extraction service

Takes a plant photo and returns 3 distinct 5-color palettes with percentages,
matching the "chromature" format from the reference design.

## What it does

- Downscales the image for speed
- Runs k-means clustering 3 separate times with different seeds/weighting
  to produce 3 genuinely different-looking palettes from the same photo
- Returns each palette as 5 hex colors + percentage of the photo they cover

## Deploy to Render (free tier, ~10 minutes)

1. Push this folder to a new GitHub repo (or upload directly if Render
   supports it in your plan)
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Settings:
   - **Runtime**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app`
5. Deploy. Render gives you a URL like `https://your-service.onrender.com`
6. Test it's live: visit `https://your-service.onrender.com/health` —
   should return `{"status": "ok"}`

Note: Render's free tier spins down after inactivity, so the first request
after idle time takes ~30-50 seconds to "wake up." Fine for testing; if this
matters for production, upgrade to a paid instance later.

## Wire into FlutterFlow

Same pattern as your Plant ID API call:

1. **Settings → API Calls → + Add API Call**, name it `ExtractColors`
2. **Method**: `POST`
3. **URL**: `https://your-service.onrender.com/extract-colors`
4. **Body type**: JSON
5. **Body**:
   ```json
   {
     "image_url": "[photo_url]"
   }
   ```
   Use the storage URL from your Upload Media action here (simpler than
   base64 — no extra conversion step needed in FlutterFlow). If you'd
   rather send base64 directly, use `"image_base64"` instead.
6. **Test** it in FlutterFlow's API tester with a real photo URL
7. FlutterFlow will parse the response into a data structure — you'll get
   `chromature_1`, `chromature_2`, `chromature_3`, each a list of
   `{hex, pct}` objects

## Response format

```json
{
  "chromature_1": [
    {"hex": "#6b7a4a", "pct": 60.0},
    {"hex": "#9aa678", "pct": 20.0},
    {"hex": "#d4537e", "pct": 10.0},
    {"hex": "#ed93b1", "pct": 7.0},
    {"hex": "#3b4d28", "pct": 3.0}
  ],
  "chromature_2": [...],
  "chromature_3": [...]
}
```

## Rendering the bar in FlutterFlow (no image generation needed)

For each chromature, use a `Row` widget containing 5 `Container` widgets:

- Each container's `width` = `parentWidth * (pct / 100)`
- Each container's `color` = hex value from the response (FlutterFlow has
  a "color from hex" function in its variable/expression picker)
- No rounded corners on inner containers, rounded corners only on the
  outer Row wrapper, to match the reference design's pill-shaped bar

This produces the exact segmented bar from your reference image, built
entirely from native widgets — nothing to generate or download as an image.
