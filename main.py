"""
Chess Piece Detection with Board Mapping - FIXED COORDINATES VERSION
Corrects coordinate mapping: 
- Vertical: a (top) to h (bottom)
- Horizontal: 1 (left) to 8 (right)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Import our detectors
from detect_pieces import ChessPieceDetector
from detect_board import ChessboardDetector

def create_board_grid(corners: np.ndarray) -> np.ndarray:
    """
    Creates the boundary of the internal grid from raw corner points.
    Returns array of 4 corners: [TopLeft, TopRight, BottomRight, BottomLeft]
    """
    if len(corners) < 4:
        print(f"Error: Only found {len(corners)} corners. Need at least 4.")
        return None

    # Just take the x,y coordinates
    pts = corners[:, :2].astype(np.float32)

    # Find the 4 extreme corners of the convex hull
    hull = cv2.convexHull(pts)
    hull = hull.squeeze()
    
    if len(hull) < 4:
        return None

    # Standard corner sorting
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1)
    
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(diff)]
    bl = hull[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_square_from_position_homography(x: float, y: float, grid: np.ndarray) -> str:
    """
    Maps pixel coordinates to your specific board orientation:
    - Left->Right = 1->8
    - Top->Bottom = a->h
    """
    if grid is None or len(grid) != 4:
        return "unknown"

    # 1. Source Points: The 4 corners of the detected INTERNAL grid
    src_pts = grid.reshape(-1, 1, 2)

    # 2. Destination Points: Logical coordinates 1.0 to 7.0
    # We map the internal grid boundary (which is between squares)
    dst_pts = np.float32([
        [1, 1], # Top-Left
        [7, 1], # Top-Right
        [7, 7], # Bottom-Right
        [1, 7]  # Bottom-Left
    ]).reshape(-1, 1, 2)

    # 3. Calculate Homography
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 4. Transform Piece Position
    piece_point = np.float32([[[x, y]]])
    transformed_point = cv2.perspectiveTransform(piece_point, M)
    
    logical_x = transformed_point[0][0][0]
    logical_y = transformed_point[0][0][1]

    # 5. Convert to Index (0-7)
    col_idx = int(np.floor(logical_x)) # Horizontal Index
    row_idx = int(np.floor(logical_y)) # Vertical Index

    # Clamp to 0-7
    col_idx = max(0, min(7, col_idx))
    row_idx = max(0, min(7, row_idx))

    # 6. Convert to Notation (YOUR SPECIFIC ORIENTATION)
    
    # Horizontal: 1 (Left) to 8 (Right)
    # col_idx 0 -> "1", col_idx 7 -> "8"
    rank_number = col_idx + 1

    # Vertical: a (Top) to h (Bottom)
    # row_idx 0 -> "a", row_idx 7 -> "h"
    files = "abcdefgh"
    file_letter = files[row_idx]

    return f"{file_letter}{rank_number}"


def map_pieces_to_squares(pieces: List[Dict], grid: np.ndarray) -> List[Dict]:
    pieces_with_squares = []
    
    for piece in pieces:
        bbox = piece['bbox']
        
        center_x = (bbox[0] + bbox[2]) / 2
        
        # Use "feet" position (90% down the box)
        center_y = bbox[1] + (bbox[3] - bbox[1]) * 0.90 
        
        square = get_square_from_position_homography(center_x, center_y, grid)
        
        p_copy = piece.copy()
        p_copy['square'] = square
        p_copy['center'] = (center_x, center_y)
        pieces_with_squares.append(p_copy)
    
    return pieces_with_squares

def visualize_board_with_pieces(image: np.ndarray, pieces: List[Dict], grid: np.ndarray) -> np.ndarray:
    vis_image = image.copy()
    
    # Draw the boundary grid
    if grid is not None and len(grid) == 4:
        pts = grid.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_image, [pts], True, (0, 255, 255), 2)

    # Draw pieces
    for piece in pieces:
        bbox = piece['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Green box for White, Red box for Black
        color = (0, 255, 0) if piece['class'].isupper() else (0, 0, 255)
        
        # Draw Box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Use piece['class'] (e.g., 'p') instead of piece['name']
        label = f"{piece['class']} {piece['square']}" 
        # --------------------------
        
        # Draw Text Background (Black box for readability)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis_image, (x1, y1 - 25), (x1 + w, y1), color, -1)
        
        # Draw Text (White or Black depending on box color for contrast)
        text_color = (0, 0, 0) if piece['class'].isupper() else (255, 255, 255)
        cv2.putText(vis_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Draw the "Feet" point
        cx, cy = piece['center']
        cv2.circle(vis_image, (int(cx), int(cy)), 4, (255, 0, 255), -1)
    
    return vis_image

def main():
    # Hardcoded paths
    pieces_model_path = "models/480M_leyolo_pieces.onnx"
    xcorners_model_path = "models/480L_leyolo_xcorners.onnx"
    input_image_path = "examples/green_chess.jpg"
    output_image_path = "output/final_mapped_board.jpg"
    
    print("=" * 60)
    print("Chess Board Logic Fix - Corrected Coordinates")
    print("=" * 60)
    
    if not Path(input_image_path).exists():
        print(f"Error: Image not found: {input_image_path}")
        return
    
    print(f"Loading image: {input_image_path}")
    image = cv2.imread(input_image_path)
    if image is None: return
    
    # 1. Detect Pieces
    print("Detecting pieces...")
    piece_detector = ChessPieceDetector(pieces_model_path)
    pieces = piece_detector.detect_pieces(image, score_threshold=0.3)
    print(f"-> Found {len(pieces)} pieces.")

    # 2. Detect Corners
    print("Detecting board corners...")
    corner_detector = ChessboardDetector(xcorners_model_path)
    
    try:
        # Preprocess
        input_tensor, metadata = corner_detector.preprocess_image(image)
        # Run Model
        outputs = corner_detector.session.run(None, {corner_detector.session.get_inputs()[0].name: input_tensor})
        preds = outputs[0]
        
        # Postprocess manually to get ALL valid corners
        preds = np.transpose(preds, (0, 2, 1))
        valid_preds = preds[0][preds[0, :, 4] >= 0.3]
        
        # Scale back to original image
        raw_corners = []
        pad_l, pad_r, pad_t, pad_b = metadata['padding']
        scale_x = metadata['width'] / (480 - pad_l - pad_r)
        scale_y = metadata['height'] / (288 - pad_t - pad_b)
        
        for p in valid_preds:
            xc = (p[0] - pad_l) * scale_x
            yc = (p[1] - pad_t) * scale_y
            raw_corners.append([xc, yc, p[4]]) 
            
        all_corners = np.array(raw_corners)
        print(f"-> Found {len(all_corners)} raw grid points.")
        
    except Exception as e:
        print(f"Error extracting corners: {e}")
        return

    # 3. Create Grid Boundary
    grid = create_board_grid(all_corners)
    
    if grid is None:
        print("Failed to build grid structure.")
        return

    # 4. Map Pieces
    print("Mapping pieces to squares...")
    mapped_pieces = map_pieces_to_squares(pieces, grid)
    
    # 5. Output
    print("-" * 40)
    # Sort for easier reading (by square)
    mapped_pieces.sort(key=lambda x: x['square'])
    
    print(f"{'Piece':<6} {'Square':<6}")
    print("-" * 15)

    for p in mapped_pieces:
        # p['class'] holds the single letter (p, P, r, R, etc.)
        # p['square'] holds the coordinate (1a, 4e, etc.)
        print(f"{p['class']:<6} {p['square']}")
    
    vis = visualize_board_with_pieces(image, mapped_pieces, grid)
    cv2.imwrite(output_image_path, vis)
    print(f"\nSaved result to {output_image_path}")
    cv2.imshow("Final Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()