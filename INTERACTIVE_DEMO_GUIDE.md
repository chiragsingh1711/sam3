# Interactive Point-Based Segmentation Demo Guide

This guide explains how to use the interactive segmentation demos for SAM3.

## 📋 Overview

Two demo interfaces are provided:

1. **Matplotlib Demo** (`interactive_segmentation_demo.py`) - Desktop GUI application
2. **Gradio Web Demo** (`gradio_segmentation_demo.py`) - Browser-based interface

Both demos allow you to segment objects by clicking points on an image, similar to Meta's Segment Anything demo.

---

## 🚀 Quick Start

### Option 1: Matplotlib Desktop Demo

**Run the demo:**
```bash
python interactive_segmentation_demo.py path/to/your/image.jpg
```

**Controls:**
- **Left Click**: Add foreground point (include in mask) - Green ✓
- **Right Click**: Add background point (exclude from mask) - Red ✗
- **'r' Key**: Reset points and start over
- **'s' Key**: Save current mask to `./outputs/`
- **'q' Key**: Quit the application

**Example:**
```bash
# Segment objects in an image
python interactive_segmentation_demo.py examples/dog.jpg

# Use custom checkpoint
python interactive_segmentation_demo.py image.jpg --checkpoint path/to/checkpoint.pt
```

---

### Option 2: Gradio Web Demo

**Install Gradio (if not already installed):**
```bash
pip install gradio
```

**Run the demo:**
```bash
python gradio_segmentation_demo.py
```

**Open your browser to:** `http://localhost:7860`

**Usage:**
1. Upload an image or use example images
2. Select "Foreground (include)" or "Background (exclude)" point type
3. Click on the image to add points
4. Segmentation updates automatically
5. Click "Reset Points" to start over
6. Click "Download Mask" to save the result

**Options:**
```bash
# Create a public share link (accessible from anywhere)
python gradio_segmentation_demo.py --share

# Run on different port
python gradio_segmentation_demo.py --port 8080

# Use custom checkpoint
python gradio_segmentation_demo.py --checkpoint path/to/checkpoint.pt
```

---

## 💡 How It Works

### Point-Based Segmentation Workflow

1. **Load Image**: The image is processed once to compute embeddings (this is the slow part)
2. **Click Points**: Each click is fast (~50-100ms) because embeddings are cached
3. **Get Masks**: The model returns segmentation masks based on your points
4. **Refine**: Add more points to refine the segmentation

### Point Types

| Point Type | Label | Color | Purpose | Example |
|------------|-------|-------|---------|---------|
| Foreground | 1 | Green | "Include this area" | Click on the object you want |
| Background | 0 | Red | "Exclude this area" | Click on areas to remove |

### Multi-Mask Output

- **First click**: Returns 3 candidate masks, automatically selects the best one
- **Additional clicks**: Refines the selected mask based on your feedback

### Iterative Refinement

```
Click 1: Foreground point on dog's head
  → Gets rough dog segmentation

Click 2: Background point on the grass
  → Refines to exclude grass from mask

Click 3: Foreground point on dog's tail
  → Ensures tail is included

Result: Precise dog segmentation!
```

---

## 🎯 Tips for Best Results

### Getting Good Segmentations

1. **Start with a clear foreground point**
   - Click on the center of the object you want to segment

2. **Add background points for refinement**
   - If unwanted areas are included, right-click them to exclude

3. **Use multiple foreground points for complex objects**
   - For objects with disconnected parts (e.g., person with arms spread)

4. **Reset and try different points**
   - If you're not getting good results, press 'r' to reset and try again

### Common Scenarios

**Scenario 1: Simple Object (e.g., a cup)**
- Usually 1 foreground point is enough
- The model will return 3 candidates, pick the best automatically

**Scenario 2: Object with Background Clutter**
- Start with 1 foreground point on the object
- Add background points on cluttered areas to exclude them

**Scenario 3: Multiple Connected Objects**
- Use background points to separate objects
- Example: Two people standing close - use background points between them

**Scenario 4: Complex Object (e.g., bicycle)**
- Use multiple foreground points on different parts (frame, wheels, seat)
- Ensures all parts are included in the mask

---

## 📦 Output Files

### Matplotlib Demo
Saves to `./outputs/` directory:
- `mask_001.png` - Binary mask (white = object, black = background)
- `overlay_001.png` - Visualization with mask overlay

### Gradio Demo
- Download button provides binary mask as PNG
- Mask values: 255 = object, 0 = background

---

## 🔧 Advanced Usage

### Using in Your Own Code

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
import numpy as np
from PIL import Image

# Initialize once
model = build_sam3_image_model()
predictor = SAM3InteractiveImagePredictor(model)

# Load image (do once per image)
image = np.array(Image.open("image.jpg"))
predictor.set_image(image)

# Segment with points (fast, can do many times)
points = np.array([[500, 375], [450, 300]])  # (x, y) coordinates
labels = np.array([1, 0])  # 1=foreground, 0=background

masks, scores, low_res_masks = predictor.predict(
    point_coords=points,
    point_labels=labels,
    multimask_output=True
)

# Use best mask
best_mask = masks[np.argmax(scores)]
```

### Integration with Other Tools

**OpenCV:**
```python
import cv2

# Convert mask to OpenCV format
mask_cv = (best_mask * 255).astype(np.uint8)

# Find contours
contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw on image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
```

**PIL/Pillow:**
```python
from PIL import Image

# Create RGBA image with transparency
rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
rgba[..., :3] = image
rgba[..., 3] = best_mask * 255  # Alpha channel

rgba_image = Image.fromarray(rgba)
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Install required dependencies
pip install numpy pillow matplotlib scipy

# For Gradio demo
pip install gradio
```

### Issue: Model takes too long to load

**Solution:**
- First load is always slower (downloads checkpoint from HuggingFace)
- Subsequent runs use cached checkpoint
- Use `--checkpoint` to specify local checkpoint file

### Issue: Predictions are slow

**Solution:**
- `set_image()` is slow (computes embeddings) - only call once per image
- `predict()` should be fast (~50-100ms)
- Check if running on GPU: `torch.cuda.is_available()`

### Issue: Segmentation quality is poor

**Tips:**
1. Try different point placements
2. Add more points for complex objects
3. Use background points to exclude unwanted areas
4. Make sure points are clearly on/off the object

### Issue: Gradio demo won't start

**Solution:**
```bash
# Install gradio
pip install gradio

# Try different port
python gradio_segmentation_demo.py --port 8080

# Check if port is already in use
lsof -i :7860  # On Linux/Mac
```

---

## 📚 Additional Resources

- **SAM3 Paper**: [Segment Anything 3](https://arxiv.org/abs/...)
- **Original SAM Demo**: https://segment-anything.com/
- **Meta's 3D Demo**: https://aidemos.meta.com/segment-anything/editor/convert-image-to-3d

---

## 🤝 Contributing

Found a bug or have a feature request? Please open an issue!

**Possible enhancements:**
- [ ] Multi-object segmentation (multiple masks at once)
- [ ] Video segmentation interface
- [ ] 3D visualization export
- [ ] Mobile-friendly interface
- [ ] Batch processing mode

---

## 📄 License

These demos are provided as examples for using SAM3. See the main repository LICENSE for details.
