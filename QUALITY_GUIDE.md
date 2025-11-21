# SAM3 Segmentation Quality Guide

## Understanding Mask Quality Parameters

The quality of your segmentation masks is controlled by **two critical thresholds**:

### 1. Mask Threshold (Default: 0.6)

**What it does:** Controls how aggressively background pixels are excluded from the mask.

**How it works:**
- SAM3 produces a probability map (0.0 to 1.0) for each pixel
- The mask threshold determines the cutoff: pixels above this value = object, below = background
- **Original issue:** Was hardcoded at 0.5, which included pixels that were only 50% likely to be the object

**Recommended values:**
- **0.7-0.8**: Clean, precise masks (recommended for production)
  - Minimal background noise
  - Matches Meta's playground quality
  - Best for transparent overlays and cutouts

- **0.5-0.6**: Balanced quality
  - Some background pixels may leak through
  - Good for getting complete objects

- **0.3-0.4**: Loose, inclusive masks
  - Captures uncertain edges
  - May include background noise
  - Use when objects have fuzzy boundaries

### 2. Confidence Threshold (Default: 0.5)

**What it does:** Filters which detected objects to keep based on detection confidence.

**How it works:**
- SAM3 assigns a confidence score to each detected object
- Only objects scoring above this threshold are returned
- Does NOT affect mask quality, only which objects are detected

**Recommended values:**
- **0.7-0.9**: High confidence only
  - Fewer detections, but all are reliable
  - Reduces false positives

- **0.4-0.6**: Balanced detection
  - Good mix of recall and precision
  - Default setting

- **0.1-0.3**: Detect everything
  - Maximum recall
  - May include false positives
  - Use when you want to be sure nothing is missed

## Why Meta's Playground Looks Better

Meta's official SAM3 playground achieves cleaner results by:

1. **Using a higher mask threshold (0.6-0.7 instead of 0.5)**
   - This is now the default in our tool!

2. **Post-processing** (potential, not confirmed):
   - Connected component analysis
   - Morphological operations (erosion/dilation)
   - Edge smoothing

3. **Interactive refinement**:
   - Ability to add positive/negative points
   - Iterative mask improvement
   - Human-in-the-loop refinement

## Quick Settings Guide

### For Product Photography (Clean Cutouts)
```
Mask Threshold: 0.75
Confidence Threshold: 0.6
```

### For People/Portraits
```
Mask Threshold: 0.65
Confidence Threshold: 0.5
```

### For Complex Scenes
```
Mask Threshold: 0.6
Confidence Threshold: 0.4
```

### For Fuzzy/Hairy Objects
```
Mask Threshold: 0.5
Confidence Threshold: 0.5
```

## API Usage

### Python Example
```python
import requests

response = requests.post('http://136.112.178.210:8000/segment', json={
    'box': {
        'center_x': 0.5,
        'center_y': 0.5,
        'width': 0.3,
        'height': 0.3,
        'label': True
    },
    'mask_threshold': 0.7,        # Cleaner masks
    'confidence_threshold': 0.6,  # Higher confidence
    'text_prompt': 'person'       # Optional
})
```

### JavaScript Example
```javascript
const response = await fetch('http://136.112.178.210:8000/segment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        box: {
            center_x: 0.5,
            center_y: 0.5,
            width: 0.3,
            height: 0.3,
            label: true
        },
        mask_threshold: 0.7,
        confidence_threshold: 0.6,
        text_prompt: 'car'
    })
});
```

## Troubleshooting

### "Too much background in my masks"
- **Solution:** Increase mask threshold to 0.7-0.8
- This is the most common issue

### "Missing parts of the object"
- **Solution:** Decrease mask threshold to 0.5-0.6
- Or draw a larger bounding box

### "No objects detected"
- **Solution:** Decrease confidence threshold to 0.3-0.4
- Or adjust your bounding box

### "Too many false detections"
- **Solution:** Increase confidence threshold to 0.6-0.8
- Use a text prompt to specify the object

## Technical Details

### Mask Generation Pipeline

1. **Feature Extraction**: Image → Vision Encoder → Feature Maps
2. **Detection**: Detector predicts boxes + confidence scores
3. **Mask Generation**: Decoder generates probability maps
4. **Confidence Filtering**: Keep objects with `score > confidence_threshold`
5. **Mask Binarization**: Convert probabilities to binary with `mask > mask_threshold`
6. **Post-processing**: Resize to original image dimensions

### The 0.5 Problem

The original SAM3 processor hardcoded the mask threshold at 0.5:

```python
# OLD CODE (sam3_image_processor.py line 219)
state["masks"] = out_masks > 0.5  # Too permissive!
```

This meant pixels with only 50% probability were included, leading to noisy masks.

### Our Solution

We now allow custom thresholds:

```python
# NEW CODE (backend/app.py)
mask_threshold = request.mask_threshold if request.mask_threshold is not None else 0.6
current_state["masks"] = masks_logits > mask_threshold  # Configurable!
```

Default increased to 0.6, and fully adjustable from the UI.

## Best Practices

1. **Start with defaults (0.6, 0.5)** and adjust as needed
2. **Use text prompts** for better accuracy when multiple objects are present
3. **Draw tight boxes** around your object of interest
4. **Experiment with thresholds** - every image is different
5. **For production**, consider post-processing (OpenCV morphological operations)

## Future Enhancements

Potential improvements to match Meta's playground:

- [ ] Interactive point refinement (add positive/negative points)
- [ ] Morphological post-processing (erosion, dilation)
- [ ] Edge smoothing with bilateral filtering
- [ ] Multiple mask merging
- [ ] Mask editing tools
- [ ] Batch processing with consistent thresholds

---

**Need help?** Check the main [DEPLOYMENT.md](DEPLOYMENT.md) for setup instructions.
