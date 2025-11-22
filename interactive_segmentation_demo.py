#!/usr/bin/env python3
"""
Interactive Point-Based Segmentation Demo for SAM3
===================================================

Usage:
    python interactive_segmentation_demo.py <path_to_image>

Controls:
    - Left click: Add foreground point (include in mask)
    - Right click: Add background point (exclude from mask)
    - 'r' key: Reset points and start over
    - 's' key: Save current mask
    - 'q' key: Quit

Example:
    python interactive_segmentation_demo.py examples/example_image.jpg
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Circle

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor


class InteractiveSegmentationDemo:
    """Interactive point-based segmentation demo using SAM3."""

    def __init__(self, image_path, checkpoint=None):
        """
        Initialize the interactive demo.

        Args:
            image_path: Path to the image file
            checkpoint: Optional path to model checkpoint
        """
        print("Loading SAM3 model...")
        self.model = build_sam3_image_model(checkpoint=checkpoint)
        self.predictor = SAM3InteractiveImagePredictor(self.model)

        print(f"Loading image: {image_path}")
        self.image_path = Path(image_path)
        self.image = np.array(Image.open(image_path).convert("RGB"))

        print("Computing image embeddings...")
        self.predictor.set_image(self.image)

        # State tracking
        self.points = []
        self.labels = []
        self.current_mask = None
        self.low_res_mask = None
        self.point_artists = []

        # Setup matplotlib figure
        self.setup_plot()

        print("\n" + "="*60)
        print("Interactive Segmentation Demo Ready!")
        print("="*60)
        print("Controls:")
        print("  - Left click: Add foreground point (green)")
        print("  - Right click: Add background point (red)")
        print("  - 'r' key: Reset and start over")
        print("  - 's' key: Save current mask")
        print("  - 'q' key: Quit")
        print("="*60 + "\n")

    def setup_plot(self):
        """Setup matplotlib figure and axes."""
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('SAM3 Interactive Point-Based Segmentation', fontsize=16)

        # Left axis: Original image with points
        self.ax_image = self.axes[0]
        self.ax_image.set_title('Click to add points\n(Left: foreground, Right: background)')
        self.image_display = self.ax_image.imshow(self.image)
        self.ax_image.axis('off')

        # Right axis: Segmentation result
        self.ax_mask = self.axes[1]
        self.ax_mask.set_title('Segmentation Result')
        self.mask_display = self.ax_mask.imshow(self.image)
        self.ax_mask.axis('off')

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.tight_layout()

    def on_click(self, event):
        """Handle mouse click events."""
        # Check if click is in the image axis
        if event.inaxes != self.ax_image:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Get click coordinates
        x, y = int(event.xdata), int(event.ydata)

        # Determine if foreground or background point
        if event.button == 1:  # Left click
            is_positive = True
            color = 'green'
            marker = '+'
            print(f"Added foreground point at ({x}, {y})")
        elif event.button == 3:  # Right click
            is_positive = False
            color = 'red'
            marker = 'x'
            print(f"Added background point at ({x}, {y})")
        else:
            return

        # Add point
        self.add_point(x, y, is_positive, color, marker)

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'r':
            print("Resetting...")
            self.reset()
        elif event.key == 's':
            self.save_mask()
        elif event.key == 'q':
            print("Quitting...")
            plt.close(self.fig)

    def add_point(self, x, y, is_positive, color, marker):
        """Add a point and update segmentation."""
        # Store point
        self.points.append([x, y])
        self.labels.append(1 if is_positive else 0)

        # Draw point on image
        point_artist = self.ax_image.plot(
            x, y,
            marker=marker,
            color=color,
            markersize=15,
            markeredgewidth=3,
            markeredgecolor='white' if is_positive else 'black',
            markerfacecolor=color
        )[0]
        self.point_artists.append(point_artist)

        # Run prediction
        self.update_segmentation()

    def update_segmentation(self):
        """Run SAM3 prediction with current points."""
        if len(self.points) == 0:
            return

        point_coords = np.array(self.points)
        point_labels = np.array(self.labels)

        print(f"Running prediction with {len(self.points)} point(s)...")

        # First point: get multiple masks to choose best
        # Additional points: refine with single mask
        multimask = (len(self.points) == 1)

        masks, scores, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=self.low_res_mask if self.low_res_mask is not None else None,
            multimask_output=multimask,
        )

        # Select best mask
        if multimask:
            best_idx = np.argmax(scores)
            self.current_mask = masks[best_idx]
            self.low_res_mask = low_res_masks[best_idx]
            print(f"Selected mask {best_idx+1}/3 (score: {scores[best_idx]:.3f})")
        else:
            self.current_mask = masks[0]
            self.low_res_mask = low_res_masks[0]
            print(f"Refined mask (score: {scores[0]:.3f})")

        # Display result
        self.display_mask()

    def display_mask(self):
        """Display the current segmentation mask."""
        if self.current_mask is None:
            return

        # Create colored overlay
        overlay = self.image.copy()

        # Create semi-transparent mask overlay (blue with alpha)
        mask_overlay = np.zeros_like(self.image)
        mask_overlay[self.current_mask] = [0, 114, 189]  # Blue color

        # Blend with original image
        alpha = 0.5
        overlay = (overlay * (1 - alpha) + mask_overlay * alpha).astype(np.uint8)

        # Add mask contour
        from scipy import ndimage
        contour = ndimage.binary_dilation(self.current_mask) & ~self.current_mask
        overlay[contour] = [255, 255, 0]  # Yellow contour

        # Update display
        self.mask_display.set_data(overlay)
        self.ax_mask.set_title(f'Segmentation Result ({len(self.points)} point(s))')

        self.fig.canvas.draw_idle()

    def reset(self):
        """Reset all points and masks."""
        self.points = []
        self.labels = []
        self.current_mask = None
        self.low_res_mask = None

        # Remove point markers
        for artist in self.point_artists:
            artist.remove()
        self.point_artists = []

        # Reset mask display
        self.mask_display.set_data(self.image)
        self.ax_mask.set_title('Segmentation Result')

        self.fig.canvas.draw_idle()
        print("Reset complete. Click to start new segmentation.")

    def save_mask(self):
        """Save the current mask to a file."""
        if self.current_mask is None:
            print("No mask to save. Add points first.")
            return

        # Generate filename
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)

        # Find next available number
        existing = list(output_dir.glob("mask_*.png"))
        if existing:
            numbers = [int(f.stem.split('_')[1]) for f in existing]
            next_num = max(numbers) + 1
        else:
            next_num = 1

        mask_path = output_dir / f"mask_{next_num:03d}.png"
        overlay_path = output_dir / f"overlay_{next_num:03d}.png"

        # Save binary mask
        mask_image = Image.fromarray((self.current_mask * 255).astype(np.uint8))
        mask_image.save(mask_path)

        # Save overlay visualization
        overlay = self.image.copy()
        mask_overlay = np.zeros_like(self.image)
        mask_overlay[self.current_mask] = [0, 114, 189]
        overlay = (overlay * 0.5 + mask_overlay * 0.5).astype(np.uint8)

        from scipy import ndimage
        contour = ndimage.binary_dilation(self.current_mask) & ~self.current_mask
        overlay[contour] = [255, 255, 0]

        overlay_image = Image.fromarray(overlay)
        overlay_image.save(overlay_path)

        print(f"Saved mask to: {mask_path}")
        print(f"Saved overlay to: {overlay_path}")

    def run(self):
        """Run the interactive demo."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive point-based segmentation demo for SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_segmentation_demo.py image.jpg
  python interactive_segmentation_demo.py image.jpg --checkpoint path/to/checkpoint.pt

Controls:
  Left click:  Add foreground point (include in mask)
  Right click: Add background point (exclude from mask)
  'r' key:     Reset points and start over
  's' key:     Save current mask to ./outputs/
  'q' key:     Quit
        """
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Run demo
    demo = InteractiveSegmentationDemo(args.image, checkpoint=args.checkpoint)
    demo.run()


if __name__ == "__main__":
    main()
