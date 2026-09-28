import numpy as np
import pandas as pd
import colorsys

# -----------------------------------------------------------------
# STEP 1: Anchor colors from published Marquis reagent color
# response literature. Approximate RGB estimates from documented
# descriptions, not spectrophotometer-measured — stated explicitly
# in the pitch as a known limitation pending real reagent data.
# -----------------------------------------------------------------
category_anchors_rgb = {
    "no_reaction":        (245, 245, 220),  # colorless / pale yellow
    "opiates_purple":     (100, 30, 90),    # heroin/morphine - purple/violet
    "mdma_purple_black":  (60, 20, 50),     # MDMA - purple/black
    "amphetamine_orange": (200, 90, 30),    # amphetamine - orange/red
    "cocaine_peach":      (240, 190, 160),  # cocaine - pale peach/pink
}

def rgb_to_hsv_scaled(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 179, s * 255, v * 255

category_anchors_hsv = {
    label: rgb_to_hsv_scaled(*rgb)
    for label, rgb in category_anchors_rgb.items()
}

# -----------------------------------------------------------------
# STEP 2: Base noisy samples around each anchor (sample/sensor noise)
# -----------------------------------------------------------------
def generate_samples(anchor_hsv, n=200, hue_std=4, sat_std=15, val_std=20):
    h, s, v = anchor_hsv
    hs = np.clip(np.random.normal(h, hue_std, n), 0, 179)
    ss = np.clip(np.random.normal(s, sat_std, n), 0, 255)
    vs = np.clip(np.random.normal(v, val_std, n), 0, 255)
    return np.stack([hs, ss, vs], axis=1)

rows = []
for label, anchor_hsv in category_anchors_hsv.items():
    samples = generate_samples(anchor_hsv, n=200)
    for h, s, v in samples:
        rows.append({"H": h, "S": s, "V": v, "label": label})

df = pd.DataFrame(rows)

# -----------------------------------------------------------------
# STEP 3: Lighting-condition variance (real, applicable augmentation
# for scalar color data — models actual sensor/lighting confounds,
# not image-level transforms like flip/rotate/crop which don't apply
# to single averaged color triplets)
# -----------------------------------------------------------------
lighting_shift = {
    "daylight":    (0, 0, 0),
    "fluorescent": (2, -10, 15),
    "warm_indoor": (-3, 10, -10),
    "overcast":    (1, -5, -20),
    "harsh_direct":(0, 5, 25),
}

augmented = []
for _, row in df.iterrows():
    for cond, (dh, ds, dv) in lighting_shift.items():
        h = np.clip(row.H + dh + np.random.normal(0, 2), 0, 179)
        s = np.clip(row.S + ds + np.random.normal(0, 8), 0, 255)
        v = np.clip(row.V + dv + np.random.normal(0, 8), 0, 255)
        augmented.append({"H": h, "S": s, "V": v, "label": row.label, "lighting": cond})

aug_df = pd.DataFrame(augmented)

# -----------------------------------------------------------------
# STEP 4: Additional brightness/contrast/saturation jitter pass
# (legitimate scalar-data augmentation — simulates camera exposure
# and white-balance variance on top of fixed lighting conditions)
# -----------------------------------------------------------------
def jitter_row(row, brightness_std=10, contrast_scale_range=(0.9, 1.1), sat_jitter_std=10):
    v = row.V + np.random.normal(0, brightness_std)
    contrast_factor = np.random.uniform(*contrast_scale_range)
    v = 128 + (v - 128) * contrast_factor  # contrast around midpoint
    s = row.S + np.random.normal(0, sat_jitter_std)
    return pd.Series({
        "H": row.H,
        "S": np.clip(s, 0, 255),
        "V": np.clip(v, 0, 255),
        "label": row.label,
        "lighting": row.lighting
    })

jittered_df = aug_df.apply(jitter_row, axis=1)

final_df = pd.concat([aug_df, jittered_df], ignore_index=True)

# -----------------------------------------------------------------
# STEP 5: Convert to RGB too, save, sanity check
# -----------------------------------------------------------------
def hsv_to_rgb_row(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h / 179, s / 255, v / 255)
    return int(r * 255), int(g * 255), int(b * 255)

rgb_vals = final_df.apply(lambda row: hsv_to_rgb_row(row.H, row.S, row.V), axis=1)
final_df[["R", "G", "B"]] = pd.DataFrame(rgb_vals.tolist(), index=final_df.index)

final_df.to_csv("synthetic_marquis_dataset.csv", index=False)
print(final_df.label.value_counts())
print(final_df.head())