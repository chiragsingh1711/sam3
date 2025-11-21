import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# 1. SETUP: Define your image path and point coordinates
IMAGE_PATH = "image.jpg"  # <--- CHANGE THIS
INPUT_POINT = [[1195, 632]]            # <--- CHANGE THIS: [x, y] coordinates
INPUT_LABEL = [1]                     # 1 = Foreground (Target), 0 = Background

# 2. LOAD MODEL
print("Loading SAM 3 model...")
model = build_sam3_image_model()
processor = Sam3Processor(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3. LOAD IMAGE & PREPARE STATE
image = Image.open(IMAGE_PATH).convert("RGB")
inference_state = processor.set_image(image)

# 4. RUN INFERENCE WITH POINT PROMPT
# Note: The API uses lists for points and labels.
# Check examples/sam3_image_predictor_example.ipynb if 'set_click_prompt' differs.
print(f"Segmenting object at {INPUT_POINT}...")

# The method for points is typically 'set_click_prompt' or similar in the native API.
# Based on the text prompt pattern, we pass the state.
output = processor.set_click_prompt(
    state=inference_state, 
    point_coords=INPUT_POINT, 
    point_labels=INPUT_LABEL
)

# 5. EXTRACT & VISUALIZE RESULTS
masks = output["masks"]
scores = output["scores"]

# Pick the best mask (highest score)
best_mask_idx = scores.argmax()
best_mask = masks[best_mask_idx]
best_score = scores[best_mask_idx]

print(f"Segmentation complete. Best score: {best_score:.3f}")

# specific visualization
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(best_mask, plt.gca())
show_points(np.array(INPUT_POINT), np.array(INPUT_LABEL), plt.gca())
plt.axis('off')
plt.savefig("output_mask.png")
print("Result saved to output_mask.png")
# plt.show()
