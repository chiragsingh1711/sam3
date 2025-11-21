# SAM3 Image Annotation Tool - Deployment Guide

This is a full-stack web application for interactive image segmentation using Meta's SAM3 model.

## Features

- 🖼️ Upload images via drag-and-drop or file picker
- 🎯 Draw bounding boxes to select objects
- 🤖 AI-powered segmentation using SAM3
- 💬 Optional text prompts for better targeting
- 📥 Download masked images with transparent backgrounds
- 🎨 Visual preview of all detected masks

## Architecture

- **Backend**: FastAPI (Python) - Port 8000
- **Frontend**: HTML/CSS/JavaScript - Port 5173
- **Model**: SAM3 (Segment Anything Model 3)
- **External Access**: http://136.112.178.210

## Prerequisites

1. **SAM3 Environment**:
   ```bash
   conda activate sam3
   ```

2. **SAM3 Model Access**:
   - Request access to SAM3 checkpoints on [Hugging Face](https://huggingface.co/facebook/sam3)
   - Authenticate: `huggingface-cli login`

3. **Backend Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

## Quick Start

### Option 1: Run Both Servers (Recommended)

Open two terminal windows:

**Terminal 1 - Backend**:
```bash
conda activate sam3
bash start_backend.sh
```

**Terminal 2 - Frontend**:
```bash
bash start_frontend.sh
```

### Option 2: Manual Start

**Backend**:
```bash
cd backend
conda activate sam3
python app.py
```

**Frontend**:
```bash
cd frontend
python3 serve.py
```

## Access the Application

Once both servers are running:

- **Frontend**: http://136.112.178.210:5173
- **Backend API**: http://136.112.178.210:8000
- **API Docs**: http://136.112.178.210:8000/docs

## Usage Instructions

1. **Upload Image**: Click the upload area or drag-and-drop an image
2. **Draw Box**: Click and drag on the image to draw a bounding box around the object
3. **Add Text** (Optional): Enter a description like "cat", "person", etc.
4. **Segment**: Click "Segment Image" button
5. **Download**: Click "Download Masked Image" for any detected object

## API Endpoints

### `POST /upload`
Upload an image for annotation.

**Request**: Multipart form data with image file
**Response**: Image dimensions and upload status

### `POST /segment`
Segment the image with box and optional text prompt.

**Request Body**:
```json
{
  "box": {
    "center_x": 0.5,
    "center_y": 0.5,
    "width": 0.3,
    "height": 0.3,
    "label": true
  },
  "text_prompt": "cat"
}
```

**Response**: Number of masks, scores, and bounding boxes

### `GET /download_masked_image?mask_index=0`
Download a specific masked image.

**Response**: PNG image with transparent background

### `GET /preview_masks`
Get base64-encoded previews of all masks.

**Response**: Array of mask previews with overlays

## Troubleshooting

### Backend won't start
- Ensure SAM3 environment is activated: `conda activate sam3`
- Check if model is accessible: Test with example notebook first
- Verify CUDA/GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Frontend can't connect to backend
- Check if backend is running on port 8000: `curl http://localhost:8000/health`
- Verify firewall rules allow external access on ports 5173 and 8000
- Check GCP firewall settings if needed

### Model loading is slow
- First load downloads the model from Hugging Face (can take several minutes)
- Subsequent loads are faster as the model is cached locally
- GPU is recommended for faster inference

### CORS errors
- Both servers are configured to allow cross-origin requests
- If you see CORS errors, check that both servers are running

## Performance Notes

- **First Inference**: Slower due to model initialization (~10-30 seconds)
- **Subsequent Inferences**: Much faster (~1-3 seconds)
- **GPU Recommended**: For production use, GPU significantly speeds up inference
- **Memory**: Requires ~8GB GPU memory or ~16GB RAM for CPU inference

## Security Considerations

⚠️ **This is a development deployment**:
- CORS is set to allow all origins (`*`)
- No authentication or rate limiting
- For production, add proper security measures

## File Structure

```
sam3/
├── backend/
│   ├── app.py              # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Web UI
│   └── serve.py           # Frontend server
├── start_backend.sh        # Backend startup script
├── start_frontend.sh       # Frontend startup script
└── DEPLOYMENT.md          # This file
```

## Model Information

- **Model**: SAM3 (Segment Anything Model 3)
- **Provider**: Meta AI
- **Size**: 848M parameters
- **Capabilities**:
  - Text-prompted segmentation
  - Box-prompted segmentation
  - Point-prompted segmentation
  - Open-vocabulary object detection

## License

This deployment uses SAM3 which is licensed under the SAM License. See the main SAM3 repository for details.

## Support

For SAM3 model issues, see: https://github.com/facebookresearch/sam3
For deployment issues, check the logs from both backend and frontend servers.
