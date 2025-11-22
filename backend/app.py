"""
FastAPI backend for SAM3 Image Annotation Tool
Provides endpoints for image upload, segmentation, and mask download
"""
import io
import base64
from typing import List, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

# Initialize FastAPI app
app = FastAPI(title="SAM3 Image Annotation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for GCP deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for model and processor
model = None
processor = None
interactive_predictor = None  # For point-based segmentation
current_state = {}
current_image = None
current_image_np = None  # Numpy array for interactive predictor
point_masks = None  # Store masks from point-based segmentation
point_scores = None  # Store scores from point-based segmentation


class BoxPrompt(BaseModel):
    """Box prompt in normalized coordinates [0-1]"""
    center_x: float
    center_y: float
    width: float
    height: float
    label: bool = True  # True for positive, False for negative


class SegmentRequest(BaseModel):
    """Request for segmentation with box prompt"""
    box: BoxPrompt
    text_prompt: Optional[str] = None
    confidence_threshold: Optional[float] = 0.5  # Filter detections by confidence
    mask_threshold: Optional[float] = 0.6  # Binarize masks (higher = cleaner, try 0.6-0.8)


class PointPrompt(BaseModel):
    """Single point prompt"""
    x: float  # X coordinate in pixels
    y: float  # Y coordinate in pixels
    label: int  # 1 for foreground, 0 for background


class PointSegmentRequest(BaseModel):
    """Request for point-based segmentation"""
    points: List[PointPrompt]
    multimask_output: bool = True  # Return 3 candidate masks
    mask_threshold: Optional[float] = 0.0  # Threshold for binarizing mask


@app.on_event("startup")
async def startup_event():
    """Initialize SAM3 model on startup"""
    global model, processor, interactive_predictor
    print("Loading SAM3 model...")
    try:
        # Enable instance interactivity for point-based segmentation
        model = build_sam3_image_model(enable_inst_interactivity=True)
        # Use higher default confidence threshold for better quality
        processor = Sam3Processor(model, device="cuda" if torch.cuda.is_available() else "cpu", confidence_threshold=0.5)
        # Use the built-in interactive predictor from the model
        interactive_predictor = model.inst_interactive_predictor
        if interactive_predictor is None:
            print("Warning: Model does not have interactive predictor. Point-based segmentation will not be available.")
        else:
            print("✓ Interactive predictor enabled for point-based segmentation")
            # CRITICAL FIX: Move the tracker model to the same device as the main model
            if hasattr(interactive_predictor, 'model') and interactive_predictor.model is not None:
                device = processor.device
                interactive_predictor.model = interactive_predictor.model.to(device)
                interactive_predictor.model.eval()  # Set to eval mode

                # CRITICAL FIX: Exit the global BFloat16 context that the tracker entered
                # The tracker's __init__ enters a bf16 autocast context globally, which breaks
                # other parts of the code that don't support bf16
                if hasattr(interactive_predictor.model, 'bf16_context'):
                    try:
                        interactive_predictor.model.bf16_context.__exit__(None, None, None)
                        print("✓ Disabled tracker's global BFloat16 context to prevent conflicts")
                    except:
                        pass

                print(f"✓ Interactive predictor's tracker moved to device: {device}")
            else:
                print("ERROR: Interactive predictor's internal model is None!")
                interactive_predictor = None  # Disable it if broken
        print(f"SAM3 model loaded successfully on {processor.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model will be loaded on first request")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "SAM3 Image Annotation API",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": processor.device if processor else "not initialized",
        "has_image": current_image is not None
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for annotation"""
    global current_state, current_image, current_image_np, model, processor, interactive_predictor

    # Lazy load model if not loaded during startup
    if model is None:
        print("Loading SAM3 model...")
        # Enable instance interactivity for point-based segmentation
        model = build_sam3_image_model(enable_inst_interactivity=True)
        processor = Sam3Processor(model, device="cuda" if torch.cuda.is_available() else "cpu", confidence_threshold=0.5)
        interactive_predictor = model.inst_interactive_predictor
        if interactive_predictor is None:
            print("Warning: Model does not have interactive predictor. Point-based segmentation will not be available.")
        else:
            print("✓ Interactive predictor enabled for point-based segmentation")
            # CRITICAL FIX: Move the tracker model to the same device as the main model
            if hasattr(interactive_predictor, 'model') and interactive_predictor.model is not None:
                device = processor.device
                interactive_predictor.model = interactive_predictor.model.to(device)
                interactive_predictor.model.eval()  # Set to eval mode

                # CRITICAL FIX: Exit the global BFloat16 context that the tracker entered
                # The tracker's __init__ enters a bf16 autocast context globally, which breaks
                # other parts of the code that don't support bf16
                if hasattr(interactive_predictor.model, 'bf16_context'):
                    try:
                        interactive_predictor.model.bf16_context.__exit__(None, None, None)
                        print("✓ Disabled tracker's global BFloat16 context to prevent conflicts")
                    except:
                        pass

                print(f"✓ Interactive predictor's tracker moved to device: {device}")
            else:
                print("ERROR: Interactive predictor's internal model is None!")
                interactive_predictor = None  # Disable it if broken
        print(f"SAM3 model loaded successfully on {processor.device}")

    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Store image (PIL for box-based, numpy for point-based)
        current_image = image
        current_image_np = np.array(image)

        # Set image in box-based processor
        current_state = processor.set_image(image)

        # Set image in interactive predictor (computes embeddings) - if available
        if interactive_predictor is not None:
            print("Computing image embeddings for interactive segmentation...")
            interactive_predictor.set_image(current_image_np)
            print("✓ Image embeddings computed successfully")
        else:
            print("Note: Interactive predictor not available, point-based segmentation disabled")

        return {
            "status": "success",
            "width": image.width,
            "height": image.height,
            "message": "Image uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/segment")
async def segment_image(request: SegmentRequest):
    """Segment image with box prompt"""
    global current_state, processor

    if current_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded. Please upload an image first.")

    try:
        # Set confidence threshold if provided
        if request.confidence_threshold is not None:
            processor.set_confidence_threshold(request.confidence_threshold)

        # Reset previous prompts
        processor.reset_all_prompts(current_state)

        # Re-set image to get fresh state
        current_state = processor.set_image(current_image, current_state)

        # Add text prompt if provided
        if request.text_prompt:
            current_state = processor.set_text_prompt(request.text_prompt, current_state)

        # Add box prompt
        box = [
            request.box.center_x,
            request.box.center_y,
            request.box.width,
            request.box.height
        ]
        current_state = processor.add_geometric_prompt(box, request.box.label, current_state)

        # Apply custom mask threshold if provided (for cleaner masks)
        mask_threshold = request.mask_threshold if request.mask_threshold is not None else 0.6

        # Get results with custom mask threshold
        masks_logits = current_state.get("masks_logits")
        if masks_logits is not None:
            # Re-binarize masks with custom threshold
            current_state["masks"] = masks_logits > mask_threshold

        masks = current_state.get("masks")
        boxes = current_state.get("boxes")
        scores = current_state.get("scores")

        if masks is None or len(masks) == 0:
            return {
                "status": "no_objects",
                "message": "No objects detected",
                "num_masks": 0
            }

        # Convert to CPU and numpy
        masks_np = masks.cpu().numpy()
        boxes_np = boxes.cpu().numpy() if boxes is not None else None
        scores_np = scores.cpu().numpy() if scores is not None else None

        return {
            "status": "success",
            "num_masks": len(masks_np),
            "scores": scores_np.tolist() if scores_np is not None else [],
            "boxes": boxes_np.tolist() if boxes_np is not None else [],
            "message": f"Segmented {len(masks_np)} object(s)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")


@app.post("/segment_points")
async def segment_with_points(request: PointSegmentRequest):
    """Segment image using point prompts (interactive segmentation)"""
    global point_masks, point_scores, interactive_predictor, current_image_np, model

    if current_image_np is None:
        raise HTTPException(status_code=400, detail="No image uploaded. Please upload an image first.")

    if interactive_predictor is None:
        raise HTTPException(status_code=400, detail="Interactive predictor not available. Point-based segmentation is not supported.")

    if len(request.points) == 0:
        raise HTTPException(status_code=400, detail="At least one point is required.")

    try:
        # Prepare points
        point_coords = np.array([[p.x, p.y] for p in request.points], dtype=np.float32)
        point_labels = np.array([p.label for p in request.points], dtype=np.int32)

        print(f"\nSegmenting with {len(request.points)} point(s):")
        print(f"  Image shape: {current_image_np.shape}")
        for p in request.points:
            point_type = "foreground" if p.label == 1 else "background"
            print(f"  - {point_type} point at ({p.x:.1f}, {p.y:.1f})")

        # Run prediction with normalize_coords=True (expects coordinates in original image space)
        # The coordinates from frontend are in canvas pixel space, which matches image dimensions
        masks, scores, low_res_masks = interactive_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=request.multimask_output,
            normalize_coords=True,  # Explicitly set to True (this is the default)
        )

        print(f"  → Generated {len(masks)} masks with scores: {[f'{s:.3f}' for s in scores]}")

        # Store results globally
        point_masks = masks
        point_scores = scores

        # Select best mask if multimask output
        if request.multimask_output and len(scores) > 0:
            best_idx = np.argmax(scores)
            print(f"  → Selected mask {best_idx + 1}/{len(scores)} (score: {scores[best_idx]:.3f})")
        else:
            best_idx = 0

        return {
            "status": "success",
            "num_masks": len(masks),
            "scores": scores.tolist(),
            "best_mask_index": int(best_idx),
            "message": f"Segmented with {len(request.points)} point(s)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during point-based segmentation: {str(e)}")


@app.get("/download_mask")
async def download_mask(mask_index: int = 0, source: str = "box"):
    """Download a specific mask as a binary image

    Args:
        mask_index: Index of the mask to download
        source: 'box' for box-based masks or 'point' for point-based masks
    """
    global current_state, current_image, point_masks

    if current_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")

    # Get masks based on source
    if source == "point":
        masks = point_masks
        if masks is None or len(masks) == 0:
            raise HTTPException(status_code=400, detail="No point-based masks available. Please segment with points first.")
    else:
        masks = current_state.get("masks")
        if masks is None or len(masks) == 0:
            raise HTTPException(status_code=400, detail="No masks available. Please segment first.")

    if mask_index >= len(masks):
        raise HTTPException(status_code=400, detail=f"Invalid mask index. Only {len(masks)} masks available.")

    try:
        # Get the mask
        if source == "point":
            # Point-based masks are already numpy arrays
            mask = masks[mask_index].squeeze()
        else:
            # Box-based masks are tensors
            mask = masks[mask_index].squeeze().cpu().numpy()

        # Convert to PIL Image (binary)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

        # Save to bytes
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=mask_{mask_index}.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating mask: {str(e)}")


@app.get("/download_masked_image")
async def download_masked_image(mask_index: int = 0, source: str = "box"):
    """Download the original image with mask applied (masked region shown, rest transparent)

    Args:
        mask_index: Index of the mask to download
        source: 'box' for box-based masks or 'point' for point-based masks
    """
    global current_state, current_image, point_masks

    if current_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")

    # Get masks based on source
    if source == "point":
        masks = point_masks
        if masks is None or len(masks) == 0:
            raise HTTPException(status_code=400, detail="No point-based masks available. Please segment with points first.")
    else:
        masks = current_state.get("masks")
        if masks is None or len(masks) == 0:
            raise HTTPException(status_code=400, detail="No masks available. Please segment first.")

    if mask_index >= len(masks):
        raise HTTPException(status_code=400, detail=f"Invalid mask index. Only {len(masks)} masks available.")

    try:
        # Get the mask
        if source == "point":
            # Point-based masks are already numpy arrays
            mask = masks[mask_index].squeeze()
        else:
            # Box-based masks are tensors
            mask = masks[mask_index].squeeze().cpu().numpy()

        # Convert original image to RGBA
        img_rgba = current_image.convert("RGBA")
        img_array = np.array(img_rgba)

        # Apply mask (keep masked region, make rest transparent)
        img_array[:, :, 3] = (mask * 255).astype(np.uint8)

        # Create masked image
        masked_image = Image.fromarray(img_array, mode='RGBA')

        # Save to bytes
        img_byte_arr = io.BytesIO()
        masked_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=masked_image_{mask_index}.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating masked image: {str(e)}")


@app.get("/preview_masks")
async def preview_masks(source: str = "box"):
    """Get base64 encoded preview of all masks overlaid on the image

    Args:
        source: 'box' for box-based masks or 'point' for point-based masks
    """
    global current_state, current_image, point_masks, point_scores

    if current_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")

    # Get masks and scores based on source
    if source == "point":
        masks = point_masks
        scores = point_scores
        if masks is None or len(masks) == 0:
            return {"status": "no_masks", "masks": []}
    else:
        masks = current_state.get("masks")
        scores = current_state.get("scores")
        if masks is None or len(masks) == 0:
            return {"status": "no_masks", "masks": []}

    try:
        mask_previews = []

        # Sort masks by score (highest quality first) for point-based segmentation
        if source == "point" and scores is not None and len(scores) > 0:
            # Create list of (index, mask, score) tuples
            mask_score_pairs = list(zip(range(len(masks)), masks, scores))
            # Sort by score descending (best first)
            mask_score_pairs.sort(key=lambda x: x[2], reverse=True)
            sorted_indices = [idx for idx, _, _ in mask_score_pairs]
            sorted_masks = [mask for _, mask, _ in mask_score_pairs]
            sorted_scores = [score for _, _, score in mask_score_pairs]
        else:
            sorted_indices = list(range(len(masks)))
            sorted_masks = masks
            sorted_scores = scores if scores is not None else [None] * len(masks)

        for i, (orig_idx, mask, score) in enumerate(zip(sorted_indices, sorted_masks, sorted_scores)):
            # Get the mask
            if source == "point":
                mask_np = mask.squeeze()
            else:
                mask_np = mask.squeeze().cpu().numpy()

            # Convert original image to RGBA
            img_rgba = current_image.convert("RGBA")
            img_array = np.array(img_rgba).copy()

            # Create colored overlay
            color = np.array([255, 0, 0, 128])  # Red with 50% opacity
            overlay = np.zeros_like(img_array)
            overlay[mask_np > 0.5] = color

            # Blend
            result = img_array.copy()
            mask_bool = mask_np > 0.5
            result[mask_bool] = (img_array[mask_bool] * 0.6 + overlay[mask_bool] * 0.4).astype(np.uint8)

            # Convert to PIL
            preview_image = Image.fromarray(result, mode='RGBA')

            # Encode to base64
            img_byte_arr = io.BytesIO()
            preview_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()

            # Get score (already sorted)
            score_value = float(score) if score is not None else None

            # Convert score to Python float if it's a tensor
            if hasattr(score_value, 'item'):
                score_value = score_value.item()

            mask_previews.append({
                "index": orig_idx,  # Original index for downloading
                "rank": i + 1,  # Quality rank (1 = best)
                "score": score_value,
                "preview": f"data:image/png;base64,{img_base64}"
            })

        return {
            "status": "success",
            "masks": mask_previews
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating previews: {str(e)}")


@app.post("/segment_direct")
async def segment_direct(
    file: UploadFile = File(...),
    box_center_x: float = 0.5,
    box_center_y: float = 0.5,
    box_width: float = 0.3,
    box_height: float = 0.4,
    mask_threshold: float = 0.6,
    confidence_threshold: float = 0.5,
    text_prompt: Optional[str] = None,
):
    """
    All-in-one endpoint: Upload image + segment + return best masked image

    Args:
        file: Image file to segment
        box_center_x: Normalized center X (0.0 - 1.0)
        box_center_y: Normalized center Y (0.0 - 1.0)
        box_width: Normalized width (0.0 - 1.0)
        box_height: Normalized height (0.0 - 1.0)
        mask_threshold: Quality threshold (0.3 - 0.9, default 0.6)
        confidence_threshold: Detection confidence (0.1 - 0.9, default 0.5)
        text_prompt: Optional text description of object

    Returns:
        PNG image with transparent background (best mask applied)
    """
    global model, processor

    # Lazy load model if needed
    if model is None or processor is None:
        print("Loading SAM3 model for direct segmentation...")
        try:
            model = build_sam3_image_model(enable_inst_interactivity=True)
            processor = Sam3Processor(model, device="cuda" if torch.cuda.is_available() else "cpu", confidence_threshold=0.5)

            # Exit BFloat16 context if exists
            if hasattr(model, 'inst_interactive_predictor') and model.inst_interactive_predictor is not None:
                if hasattr(model.inst_interactive_predictor.model, 'bf16_context'):
                    try:
                        model.inst_interactive_predictor.model.bf16_context.__exit__(None, None, None)
                    except:
                        pass

            print(f"✓ Model loaded on {processor.device}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        print(f"\n=== DIRECT SEGMENTATION REQUEST ===")
        print(f"Image: {image.size[0]}x{image.size[1]} pixels")
        print(f"Normalized box: center=({box_center_x:.3f}, {box_center_y:.3f}), size=({box_width:.3f}, {box_height:.3f})")
        if text_prompt:
            print(f"Text prompt: '{text_prompt}'")

        # EXACT SAME WORKFLOW AS /upload + /segment

        # Step 1: Initial image set (like /upload)
        state = processor.set_image(image)

        # Step 2: Set confidence threshold (like /segment)
        processor.set_confidence_threshold(confidence_threshold)

        # Step 3: Reset prompts (CRITICAL - was missing!)
        processor.reset_all_prompts(state)

        # Step 4: Re-set image to get fresh state (CRITICAL - was missing!)
        state = processor.set_image(image, state)

        # Step 5: Add text prompt if provided
        if text_prompt:
            state = processor.set_text_prompt(text_prompt, state)

        # Step 6: Add box prompt (as list: [center_x, center_y, width, height])
        box = [box_center_x, box_center_y, box_width, box_height]
        state = processor.add_geometric_prompt(box, True, state)

        # Step 7: Apply custom mask threshold (CRITICAL - store back in state!)
        masks_logits = state.get("masks_logits")
        if masks_logits is not None:
            state["masks"] = masks_logits > mask_threshold

        # Step 8: Get final results from state
        masks = state.get("masks")
        scores = state.get("scores")

        if masks is None or len(masks) == 0:
            raise HTTPException(status_code=404, detail="No objects detected. Try adjusting the box or confidence threshold.")

        # Find best mask (highest score)
        if scores is not None and len(scores) > 0:
            best_idx = torch.argmax(scores).item()
            best_score = scores[best_idx].item()
            print(f"✓ Selected best mask: {best_idx + 1}/{len(masks)} (score: {best_score:.3f})")
        else:
            best_idx = 0
            best_score = None
            print(f"✓ Using first mask (no scores available)")

        # Get best mask
        best_mask = masks[best_idx].squeeze().cpu().numpy()

        # Create masked image (transparent background)
        img_array = np.array(image)

        # Create RGBA image
        rgba_image = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = img_array  # RGB channels
        rgba_image[:, :, 3] = (best_mask * 255).astype(np.uint8)  # Alpha channel from mask

        # Convert to PIL
        result_image = Image.fromarray(rgba_image, mode='RGBA')

        # Save to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        print(f"✓ Returning masked image: {result_image.size[0]}x{result_image.size[1]} RGBA")

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=segmented_{file.filename}",
                "X-Mask-Score": str(best_score) if best_score is not None else "N/A",
                "X-Num-Masks": str(len(masks))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
