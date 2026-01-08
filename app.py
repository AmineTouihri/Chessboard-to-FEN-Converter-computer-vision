import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Import your logic functions from main_logic.py
from main_logic import (
    ChessPieceDetector, 
    ChessboardDetector, 
    create_board_grid, 
    map_pieces_to_squares, 
    generate_fen
)

# --- CONFIGURATION ---
# Models must be in a 'models' folder next to this file
PIECES_MODEL = os.path.join("models", "480M_leyolo_pieces.onnx")
CORNERS_MODEL = os.path.join("models", "480L_leyolo_xcorners.onnx")

# --- UI SETUP ---
st.set_page_config(
    page_title="Chess2FEN",
    page_icon="♟️",
    layout="centered"
)

st.title("♟️ Chessboard to FEN Converter")
st.markdown("Upload a photo of a chessboard to get the FEN string and analyze it on Lichess.")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Settings")
    input_method = st.radio("Input Source", ["Upload Image", "Use Camera"])
    confidence = st.slider("AI Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

# --- MAIN LOGIC WRAPPER ---
def process_chess_image(image_path):
    """
    Wraps the logic from your main() function to work within Streamlit.
    """
    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Could not load image file."

    # 2. Detect Pieces
    try:
        piece_detector = ChessPieceDetector(PIECES_MODEL)
        pieces = piece_detector.detect_pieces(image, score_threshold=confidence)
    except Exception as e:
        return None, f"Error loading Piece Model: {e}"

    # 3. Detect Board Corners
    try:
        corner_detector = ChessboardDetector(CORNERS_MODEL)
        
        # Run raw session to get all points (logic from your main.py)
        input_tensor, metadata = corner_detector.preprocess_image(image)
        outputs = corner_detector.session.run(None, {corner_detector.session.get_inputs()[0].name: input_tensor})
        preds = np.transpose(outputs[0], (0, 2, 1))
        
        # Filter raw predictions manually
        valid_preds = preds[0][preds[0, :, 4] >= confidence]
        
        raw_corners = []
        pad_l, pad_r, pad_t, pad_b = metadata['padding']
        scale_x = metadata['width'] / (480 - pad_l - pad_r)
        scale_y = metadata['height'] / (288 - pad_t - pad_b)
        
        for p in valid_preds:
            xc = (p[0] - pad_l) * scale_x
            yc = (p[1] - pad_t) * scale_y
            raw_corners.append([xc, yc])
            
        all_corners = np.array(raw_corners)
        
    except Exception as e:
        return None, f"Error detecting corners: {e}"

    # 4. Build Grid
    grid = create_board_grid(all_corners)
    if grid is None:
        return None, "Could not detect a valid 4-corner grid. Try a clearer image."

    # 5. Map & Generate FEN
    mapped_data = map_pieces_to_squares(pieces, grid)
    fen = generate_fen(mapped_data)
    
    # 6. Visualization (Optional but nice)
    # We draw the grid on the image to show the user what was detected
    vis_img = image.copy()
    pts = grid.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis_img, [pts], True, (0, 255, 0), 3)
    
    # Convert BGR to RGB for Streamlit display
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return (fen, vis_img), None

# --- APP EXECUTION ---

# 1. Get Image
image_file = None
if input_method == "Upload Image":
    image_file = st.file_uploader("Upload a chessboard image", type=['jpg', 'png', 'jpeg'])
elif input_method == "Use Camera":
    image_file = st.camera_input("Take a picture")

# 2. Process
if image_file is not None:
    # Save temp file because your logic uses cv2.imread(path)
    # It is safer to keep your logic intact than rewriting it for memory buffers
    temp_filename = "temp_streamlit_upload.jpg"
    with open(temp_filename, "wb") as f:
        f.write(image_file.getbuffer())
    
    st.image(image_file, caption="Input Image", use_container_width=True)
    
    if st.button("Detect Board Position"):
        with st.spinner("🤖 AI is analyzing the board..."):
            result, error = process_chess_image(temp_filename)
            
            if error:
                st.error(error)
            else:
                fen, vis_image = result
                
                # Clean up FEN (replace spaces for URL)
                clean_fen = fen.replace(" ", "_")
                lichess_url = f"https://lichess.org/analysis/fromPosition/{clean_fen}"
                
                st.success("Analysis Complete!")
                
                # Show Result
                st.subheader("FEN String")
                st.code(fen, language="text")
                
                # Lichess Button
                st.link_button("♟️ Analyze on Lichess", lichess_url)
                
                # Show Visualization
                st.image(vis_image, caption="AI Detection Overlay", use_container_width=True)

    # Cleanup (Optional)
    # os.remove(temp_filename)