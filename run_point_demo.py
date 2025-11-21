import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- VISUALIZATION HELPER FUNCTIONS ---
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Blue color default
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# --- 1. CONFIGURATION ---
# Change these two lines to match your specific image and desired point
IMAGE_PATH = "image.jpg" 
TARGET_X = 1195  # The X coordinate of your object
TARGET_Y = 632   # The Y coordinate of your object

# --- 2. LOAD MODEL ---
print("Loading SAM 3 model...")
model = build_sam3_image_model()
processor = Sam3Processor(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}.")

# --- 3. LOAD IMAGE ---
print(f"Loading image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")
inference_state = processor.set_image(image)

# --- 4. RUN INFERENCE (POINT SIMULATION) ---
print(f"Segmenting object at [{TARGET_X}, {TARGET_Y}]...")

# TRICK: The API expects a box, so we create a 0-size box (degenerate box)
# [x_min, y_min, x_max, y_max] -> [x, y, x, y]
point_as_box = [TARGET_X, TARGET_Y, TARGET_X, TARGET_Y]

# label=True means "Foreground" (Positive click)
output = processor.add_geometric_prompt(
    state=inference_state, 
    box=point_as_box, 
    label=True
)

# --- 5. EXTRACT RESULTS ---
masks = output["masks"]
scores = output["scores"]

# SAM 3 usually returns multiple masks (multimask output). 
# We pick the one with the highest predicted IoU score.
best_mask_idx = scores.argmax()
best_mask = masks[best_mask_idx]
best_score = scores[best_mask_idx]

print(f"Success! Best Mask Score: {best_score:.3f}")

# --- 6. VISUALIZE & SAVE ---
plt.figure(figsize=(10, 10))
plt.imshow(image)

# Show the best mask
show_mask(best_mask, plt.gca())

# Show the point you clicked (Green star)
show_points(np.array([[TARGET_X, TARGET_Y]]), np.array([1]), plt.gca())

plt.axis('off')
output_filename = "output_mask.png"
plt.savefig(output_filename, bbox_inches='tight')
print(f"Visualization saved to: {output_filename}")