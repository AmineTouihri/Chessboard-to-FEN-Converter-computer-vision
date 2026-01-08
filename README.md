# Chess Piece Detector 🎯♟️

A Python-based chess piece detection and board mapping system using ONNX models from the CameraChessWeb project. Detects and classifies chess pieces, identifies chessboard boundaries, and maps pieces to their corresponding board squares.

## Features

- **Piece Detection**: Detect and classify all 12 chess pieces with confidence scores
- **Corner Detection**: Find chessboard corner points for grid construction
- **Board Mapping**: Automatically map detected pieces to chess notation (e.g., "a1", "e4")
- **Homography Transform**: Accurate piece-to-square mapping using perspective transformation
- **High Accuracy**: Uses LeYOLO-based models optimized for chess
- **Easy to Use**: Simple Python scripts with minimal setup
- **GPU Support**: Optional CUDA acceleration for faster inference
- **Visualization**: Draw bounding boxes, grid lines, and square labels

## Detected Pieces

- **White pieces**: Pawn (P), Knight (N), Bishop (B), Rook (R), Queen (Q), King (K)
- **Black pieces**: pawn (p), knight (n), bishop (b), rook (r), queen (q), king (k)

## Board Notation

The system uses the following notation:
- **Horizontal (columns)**: 1 (left) to 8 (right)
- **Vertical (rows)**: a (top) to h (bottom)
- **Example**: Top-left square is "a1", bottom-right is "h8"

## Installation

### 1. Clone or Download This Project

```bash
cd chess_piece_detector
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8 or higher
- opencv-python
- numpy
- onnxruntime

### 3. Download ONNX Models

You need to download the AI models before using the detector.

**Option A: Manual Download**
1. Download the pieces model: [480M_leyolo_pieces.onnx](https://drive.google.com/file/d/1-80xp_nly9i6s3o0mF0mU9OZGEzUAlGj/view?usp=sharing)
2. (Optional) Download the xcorners model: [480L_leyolo_xcorners.onnx](https://drive.google.com/file/d/1-2wodbiXag9UQ44e2AYAmoRN6jVpxy83/view?usp=sharing)
3. Place the downloaded files in the `models/` directory

**Option B: Using gdown (Automated)**
```bash
# Install gdown
pip install gdown

# Download models
gdown "https://drive.google.com/uc?id=1-80xp_nly9i6s3o0mF0mU9OZGEzUAlGj" -O models/480M_leyolo_pieces.onnx
gdown "https://drive.google.com/uc?id=1-2wodbiXag9UQ44e2AYAmoRN6jVpxy83" -O models/480L_leyolo_xcorners.onnx
```

See [models/README.md](models/README.md) for detailed download instructions.

## Usage

### Quick Start: Full Board Analysis (Recommended)

The main script detects pieces, maps them to squares, and visualizes everything:

```bash
python main.py
```

This will:
1. Load `examples/chessboard.jpg`
2. Detect all chess pieces with bounding boxes
3. Detect chessboard corner points
4. Map each piece to its board square (e.g., "a1", "e4")
5. Save annotated visualization to `output/final_mapped_board.jpg`
6. Display a table of pieces and their squares

**Sample Output:**
```
Piece  Square
---------------
P      a1
N      b1
B      c1
Q      d1
K      e1
...
```

### Individual Detection Scripts

**Detect Pieces Only:**
```bash
python detect_pieces.py
```

**Detect Board Corners Only:**
```bash
python detect_board.py
```

### Using Your Own Images

Simply replace `examples/chessboard.jpg` with your own image, or modify the `input_image_path` variable in `main.py`:

```python
input_image_path = "examples/your_image.jpg"
```

### Customizing Detection Parameters

Edit the parameters in the scripts:

```python
# In main.py or detect_pieces.py
pieces = piece_detector.detect_pieces(image, score_threshold=0.3)  # Lower for more detections

# In detect_board.py
corners = corner_detector.detect_all_corners(image, conf_threshold=0.3)
```

## Examples

The `examples/` directory contains sample chess images for testing:
- `chessboard.jpg` - Standard chess position
- `green_chess.jpg` - Green board variant

**Add your own images:**
- Photos of physical chess boards
- Screenshots from online chess games
- Chess diagrams or puzzles

## Output

### Console Output
```
============================================================
Chess Board Logic Fix - Corrected Coordinates
============================================================
Loading image: examples/chessboard.jpg
Detecting pieces...
-> Found 32 pieces.
Detecting board corners...
-> Found 49 raw grid points.
Mapping pieces to squares...
----------------------------------------
Piece  Square
---------------
P      a2
P      b2
P      c2
...
K      e1

Saved result to output/final_mapped_board.jpg
```

### Visual Output
The system creates an annotated image showing:
- **Green boxes**: White pieces
- **Red boxes**: Black pieces
- **Yellow lines**: Board grid
- **Purple dots**: Piece center points
- **Square labels**: Chess notation for each piece

## Project Structure

```
chess_piece_detector/
├── main.py                        # Full board analysis
├── detect_pieces.py               # Piece detection only
├── detect_board.py                # Corner detection only
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore
│
├── models/                        # ONNX models (download required)
│   ├── 480M_leyolo_pieces.onnx   # Pieces model (3.98 MB)
│   └── 480L_leyolo_xcorners.onnx # Corners model (3.98 MB)
│
├── examples/                      # Test images
│   ├── chessboard.jpg
│   └── green_chess.jpg
│
└── output/                        # Generated outputs
    └── final_mapped_board.jpg
```

## How It Works

### Piece Detection Pipeline
1. **Preprocessing**: Image resized to 480×288 and normalized to float16
2. **Inference**: LeYOLO ONNX model processes the image
3. **Post-processing**: Non-Maximum Suppression (NMS) filters overlapping detections
4. **Output**: List of pieces with bounding boxes, class labels, and confidence scores

### Corner Detection Pipeline
1. **Preprocessing**: Image resized to 480×288 pixels
2. **Inference**: LeYOLO ONNX model detects corner keypoints
3. **Post-processing**: Extracts raw corner coordinates with confidence filtering
4. **Grid Construction**: Identifies the 4 boundary corners using convex hull

### Board Mapping (main.py)
1. **Corner Detection**: Detects corner points on the board
2. **Boundary Extraction**: Creates boundary from detected corners
3. **Homography Transform**: Calculates perspective transformation
4. **Piece Mapping**: Maps piece positions to board coordinates
5. **Square Conversion**: Converts to chess notation (a1-h8)

**Key Feature**: Uses perspective transformation to handle board angles accurately.

## Troubleshooting

### "Model not found" Error
- Download the ONNX models (see Installation step 3)
- Place them in the `models/` directory
- Verify filenames:
  - `480M_leyolo_pieces.onnx` (3.98 MB)
  - `480L_leyolo_xcorners.onnx` (3.98 MB)

### Incorrect Square Mappings
- Ensure adequate lighting and clear board boundaries
- The system expects standard 8×8 chessboard
- Works best with straight-on or slight angle views
- Verify corner detection found enough grid points (should be ~49)

### Low Detection Accuracy
- Adjust `score_threshold` in the scripts:
  - Lower (0.2) for more detections (may include false positives)
  - Higher (0.5) for fewer, more confident detections
- Ensure good lighting and clear view of pieces
- Avoid extreme angles, occlusions, or reflections

### Slow Performance
- Install `onnxruntime-gpu` for GPU acceleration (requires CUDA)
- Process lower resolution images
- Close other resource-intensive applications

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Corner Detection Issues
- Check that the board has visible grid lines
- Ensure model is correctly downloaded
- Try different `conf_threshold` values (0.2-0.4)

## Technical Details

### Coordinate System
- **Image coordinates**: (0,0) at top-left, (width, height) at bottom-right
- **Board coordinates**: a-h (top to bottom), 1-8 (left to right)
- **Transformation**: Uses OpenCV's `getPerspectiveTransform` for accurate mapping

### Detection Models
Both models use the LeYOLO architecture optimized for chess:

**Pieces Model (480M_leyolo_pieces.onnx)**:
- Input: 480×288×3 RGB image
- Output: Bounding boxes + class probabilities (12 classes)
- Post-processing: Non-Maximum Suppression with IoU threshold 0.45

**Corners Model (480L_leyolo_xcorners.onnx)**:
- Input: 480×288×3 RGB image  
- Output: Corner keypoint coordinates with confidence scores
- Post-processing: Convex hull extraction for boundary detection

The system uses homography (perspective transformation) to accurately map piece positions to board squares, even when the board is viewed at an angle.

## Credits

Models from [CameraChessWeb](https://github.com/CSSLab/CameraChess) project using LeYOLO architecture.

## License

MIT License - see [LICENSE](LICENSE) file.

Models from CameraChessWeb - refer to their repository for model licensing.

## Future Ideas

- FEN notation generation
- Position validation
- Video/webcam support
- Move detection
- **Original models**: Visit [CameraChessWeb](https://github.com/CSSLab/CameraChess)

---

**Happy Chess Detecting! ♟️**
