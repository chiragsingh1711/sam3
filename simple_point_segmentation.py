#!/usr/bin/env python3
"""
Simple Point-Based Segmentation Script
======================================

A minimal example showing how to use point prompts with SAM3.
No GUI - just processes an image with specified points and saves the result.

Usage:
    python simple_point_segmentation.py <image_path> <x> <y> [--output result.png]

Example:
    # Segment object at point (500, 375)
    python simple_point_segmentation.py dog.jpg 500 375

    # Multiple points: foreground at (500,375), background at (450,300)
    python simple_point_segmentation.py dog.jpg 500 375 1 450 300 0

    # Custom output path
    python simple_point_segmentation.py dog.jpg 500 375 --output my_mask.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor


def segment_with_points(image_path, points, labels, output_path="output_mask.png",
                        save_visualization=True):
    """
    Segment an image using point prompts.

    Args:
        image_path: Path to input image
        points: List of [x, y] coordinates
        labels: List of point labels (1=foreground, 0=background)
        output_path: Where to save the mask
        save_visualization: Whether to save a visualization overlay

    Returns:
        The segmentation mask as a numpy array
    """
    # Load model
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    predictor = SAM3InteractiveImagePredictor(model)

    # Load and set image
    print(f"Loading image: {image_path}")
    image = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(image)

    # Prepare points
    point_coords = np.array(points)
    point_labels = np.array(labels)

    print(f"\nSegmenting with {len(points)} point(s):")
    for (x, y), label in zip(points, labels):
        point_type = "foreground" if label == 1 else "background"
        print(f"  - {point_type} point at ({x}, {y})")

    # Run prediction
    print("\nRunning segmentation...")
    masks, scores, low_res_masks = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Select best mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    print(f"\nSelected mask {best_idx + 1}/3 (score: {scores[best_idx]:.3f})")
    print(f"Mask covers {mask.sum() / mask.size * 100:.1f}% of image")

    # Save binary mask
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(output_path)
    print(f"\n✓ Saved binary mask to: {output_path}")

    # Save visualization
    if save_visualization:
        viz_path = output_path.parent / f"{output_path.stem}_overlay{output_path.suffix}"

        # Create colored overlay
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[mask] = [30, 144, 255]  # Blue

        overlay = (overlay * 0.6 + mask_colored * 0.4).astype(np.uint8)

        # Add contour
        try:
            from scipy import ndimage
            contour = ndimage.binary_dilation(mask) & ~mask
            overlay[contour] = [255, 255, 0]
        except ImportError:
            pass

        # Draw points
        for (x, y), label in zip(points, labels):
            color = [0, 255, 0] if label == 1 else [255, 0, 0]
            # Draw circle
            y_min, y_max = max(0, y-8), min(overlay.shape[0], y+9)
            x_min, x_max = max(0, x-8), min(overlay.shape[1], x+9)
            yy, xx = np.ogrid[-8:9, -8:9]
            circle = xx**2 + yy**2 <= 64
            circle_crop = circle[8-(y-y_min):8+(y_max-y), 8-(x-x_min):8+(x_max-x)]
            overlay[y_min:y_max, x_min:x_max][circle_crop] = color

        overlay_image = Image.fromarray(overlay)
        overlay_image.save(viz_path)
        print(f"✓ Saved visualization to: {viz_path}")

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Simple point-based segmentation with SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single foreground point
  python simple_point_segmentation.py image.jpg 500 375

  # Multiple points (x1 y1 label1 x2 y2 label2 ...)
  # Labels: 1=foreground, 0=background
  python simple_point_segmentation.py image.jpg 500 375 1 450 300 0

  # Custom output
  python simple_point_segmentation.py image.jpg 500 375 --output results/mask.png

  # Disable visualization overlay
  python simple_point_segmentation.py image.jpg 500 375 --no-viz
        """
    )

    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("coordinates", nargs="+", type=int,
                        help="Point coordinates and labels: x1 y1 [label1] x2 y2 [label2] ...")
    parser.add_argument("--output", "-o", type=str, default="output_mask.png",
                        help="Output mask path (default: output_mask.png)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Don't save visualization overlay")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")

    args = parser.parse_args()

    # Check image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Parse coordinates
    coords = args.coordinates
    points = []
    labels = []

    if len(coords) == 2:
        # Simple case: just x, y (assume foreground)
        points.append([coords[0], coords[1]])
        labels.append(1)
    elif len(coords) % 3 == 0:
        # Format: x1 y1 label1 x2 y2 label2 ...
        for i in range(0, len(coords), 3):
            x, y, label = coords[i], coords[i+1], coords[i+2]
            if label not in [0, 1]:
                print(f"Error: Label must be 0 or 1, got {label}")
                sys.exit(1)
            points.append([x, y])
            labels.append(label)
    else:
        print("Error: Invalid coordinate format")
        print("Use: x y  OR  x1 y1 label1 x2 y2 label2 ...")
        print("Labels: 1=foreground, 0=background")
        sys.exit(1)

    # Run segmentation
    try:
        segment_with_points(
            args.image,
            points,
            labels,
            args.output,
            save_visualization=not args.no_viz
        )
        print("\n✓ Done!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
