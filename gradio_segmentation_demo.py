#!/usr/bin/env python3
"""
Web-based Interactive Segmentation Demo using Gradio
=====================================================

A browser-based point-click segmentation interface similar to Meta's demo.

Usage:
    python gradio_segmentation_demo.py

Then open your browser to the displayed URL (typically http://localhost:7860)

Features:
    - Upload any image
    - Click to add positive points (segment object)
    - Use "Add Negative Point" mode to exclude areas
    - Real-time segmentation updates
    - Download segmented masks
"""

import argparse
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor


class GradioSegmentationApp:
    """Web-based interactive segmentation using Gradio."""

    def __init__(self, checkpoint=None):
        """Initialize the app with SAM3 model."""
        print("Loading SAM3 model...")
        self.model = build_sam3_image_model(checkpoint=checkpoint)
        self.predictor = SAM3InteractiveImagePredictor(self.model)

        # Session state
        self.current_image = None
        self.points = []
        self.labels = []
        self.current_mask = None
        self.low_res_mask = None

        print("Model loaded successfully!")

    def set_image(self, image):
        """Set a new image and reset state."""
        if image is None:
            return None

        # Convert to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Store and set image in predictor
        self.current_image = image
        print(f"Setting image of shape: {image.shape}")
        self.predictor.set_image(image)

        # Reset state
        self.points = []
        self.labels = []
        self.current_mask = None
        self.low_res_mask = None

        return image

    def add_point_to_mask(self, image, point_type, evt: gr.SelectData):
        """Handle point clicks on the image."""
        if self.current_image is None:
            return image, "Please upload an image first"

        # Get click coordinates
        x, y = evt.index[0], evt.index[1]

        # Determine point type
        is_positive = (point_type == "Foreground (include)")
        label = 1 if is_positive else 0

        # Add point
        self.points.append([x, y])
        self.labels.append(label)

        point_type_str = "foreground" if is_positive else "background"
        status = f"Added {point_type_str} point at ({x}, {y}). Total points: {len(self.points)}"
        print(status)

        # Run prediction
        result_image = self.segment()

        return result_image, status

    def segment(self):
        """Run segmentation with current points."""
        if len(self.points) == 0:
            return self.current_image

        point_coords = np.array(self.points)
        point_labels = np.array(self.labels)

        # Run prediction
        multimask = (len(self.points) == 1)

        masks, scores, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=self.low_res_mask,
            multimask_output=multimask,
        )

        # Select best mask
        if multimask:
            best_idx = np.argmax(scores)
            self.current_mask = masks[best_idx]
            self.low_res_mask = low_res_masks[best_idx]
        else:
            self.current_mask = masks[0]
            self.low_res_mask = low_res_masks[0]

        # Create visualization
        return self.visualize_result()

    def visualize_result(self):
        """Create visualization with mask overlay and points."""
        if self.current_mask is None:
            return self.current_image

        # Create overlay
        overlay = self.current_image.copy()

        # Add semi-transparent mask (blue)
        mask_colored = np.zeros_like(self.current_image)
        mask_colored[self.current_mask] = [30, 144, 255]  # Dodger blue

        # Blend
        alpha = 0.4
        overlay = (overlay * (1 - alpha) + mask_colored * alpha).astype(np.uint8)

        # Add contour
        try:
            from scipy import ndimage
            contour = ndimage.binary_dilation(self.current_mask) & ~self.current_mask
            overlay[contour] = [255, 255, 0]  # Yellow contour
        except ImportError:
            pass  # Skip contour if scipy not available

        # Draw points
        for (x, y), label in zip(self.points, self.labels):
            color = [0, 255, 0] if label == 1 else [255, 0, 0]  # Green/Red
            # Draw a small circle (5 pixel radius)
            y_min, y_max = max(0, y-5), min(overlay.shape[0], y+6)
            x_min, x_max = max(0, x-5), min(overlay.shape[1], x+6)

            # Create circle mask
            yy, xx = np.ogrid[-5:6, -5:6]
            circle = xx**2 + yy**2 <= 25

            # Adjust circle size based on actual bounds
            circle_h = y_max - y_min
            circle_w = x_max - x_min
            circle_crop = circle[5-(y-y_min):5+(y_max-y), 5-(x-x_min):5+(x_max-x)]

            overlay[y_min:y_max, x_min:x_max][circle_crop] = color

            # White border
            border = (xx**2 + yy**2 <= 36) & (xx**2 + yy**2 > 25)
            border_crop = border[5-(y-y_min):5+(y_max-y), 5-(x-x_min):5+(x_max-x)]
            overlay[y_min:y_max, x_min:x_max][border_crop] = [255, 255, 255]

        return overlay

    def reset(self):
        """Reset all points and mask."""
        self.points = []
        self.labels = []
        self.current_mask = None
        self.low_res_mask = None

        if self.current_image is not None:
            return self.current_image, "Reset complete. Click to add new points."
        return None, "Reset complete. Upload an image to start."

    def get_mask_download(self):
        """Get the current mask as a downloadable file."""
        if self.current_mask is None:
            return None

        # Convert mask to PIL Image
        mask_img = Image.fromarray((self.current_mask * 255).astype(np.uint8))
        return mask_img

    def build_interface(self):
        """Build the Gradio interface."""
        with gr.Blocks(title="SAM3 Interactive Segmentation") as demo:
            gr.Markdown("""
            # 🎯 SAM3 Interactive Point-Based Segmentation

            Upload an image and click on it to segment objects!

            **Instructions:**
            1. Upload an image or use the example
            2. Select point type (Foreground to include, Background to exclude)
            3. Click on the image to add points
            4. The segmentation updates automatically
            5. Click "Reset Points" to start over or "Download Mask" to save
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    # Input image upload
                    input_image = gr.Image(
                        type="numpy",
                        label="Upload Image",
                        height=400
                    )

                    # Controls
                    point_type = gr.Radio(
                        choices=["Foreground (include)", "Background (exclude)"],
                        value="Foreground (include)",
                        label="Point Type"
                    )

                    with gr.Row():
                        reset_btn = gr.Button("🔄 Reset Points", variant="secondary")
                        download_btn = gr.Button("💾 Download Mask", variant="primary")

                    status_text = gr.Textbox(
                        label="Status",
                        value="Upload an image to start",
                        interactive=False
                    )

                with gr.Column(scale=1):
                    # Output image with segmentation
                    output_image = gr.Image(
                        type="numpy",
                        label="Segmentation Result (Click to add points)",
                        height=400
                    )

                    # Download output
                    download_file = gr.File(label="Download Mask", visible=False)

            # Example images
            gr.Examples(
                examples=[
                    ["examples/example_image.jpg"] if Path("examples/example_image.jpg").exists() else None
                ],
                inputs=input_image,
                label="Example Images"
            )

            # Event handlers
            input_image.change(
                fn=self.set_image,
                inputs=[input_image],
                outputs=[output_image]
            ).then(
                fn=lambda: "Image loaded! Click on the image to segment objects.",
                outputs=[status_text]
            )

            output_image.select(
                fn=self.add_point_to_mask,
                inputs=[output_image, point_type],
                outputs=[output_image, status_text]
            )

            reset_btn.click(
                fn=self.reset,
                outputs=[output_image, status_text]
            )

            download_btn.click(
                fn=self.get_mask_download,
                outputs=[download_file]
            ).then(
                fn=lambda: gr.File(visible=True),
                outputs=[download_file]
            )

            gr.Markdown("""
            ---
            ### Tips:
            - **Green dots** = Foreground points (include in mask)
            - **Red dots** = Background points (exclude from mask)
            - Use background points to refine the segmentation
            - The first click returns the best of 3 candidate masks
            - Additional clicks refine the segmentation
            """)

        return demo


def main():
    parser = argparse.ArgumentParser(description="Web-based SAM3 segmentation demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--share", action="store_true",
                        help="Create a public share link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server on")

    args = parser.parse_args()

    # Create app
    app = GradioSegmentationApp(checkpoint=args.checkpoint)
    demo = app.build_interface()

    # Launch
    print("\n" + "="*60)
    print("Starting Gradio web interface...")
    print("="*60)

    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
