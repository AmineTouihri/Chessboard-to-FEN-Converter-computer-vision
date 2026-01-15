import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

class ChessboardDetector:
    def __init__(self, debug=True, output_dir="chessboard_detection"):
        self.debug = debug
        self.output_dir = output_dir
        
        if debug and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_image(self, image_path):
        """Load and preprocess the image"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        if self.debug:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.image_rgb)
            plt.title("Original Image")
            plt.axis('off')
            plt.savefig(os.path.join(self.output_dir, "01_original.jpg"))
            plt.close()
        
        return self.image_rgb
    
    def compute_gradients(self):
        """Compute gradients and find edge orientations"""
        # Use Sobel for better edge detection
        grad_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and angle
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        angle = np.mod(angle, 180)  # Normalize to 0-180 degrees
        
        if self.debug:
            plt.figure(figsize=(15, 10))
            plt.subplot(231)
            plt.imshow(np.abs(grad_x), cmap='gray')
            plt.title("Gradient X (dI/dx)")
            plt.axis('off')
            
            plt.subplot(232)
            plt.imshow(np.abs(grad_y), cmap='gray')
            plt.title("Gradient Y (dI/dy)")
            plt.axis('off')
            
            plt.subplot(233)
            plt.imshow(magnitude, cmap='gray')
            plt.title("Gradient Magnitude")
            plt.axis('off')
            
            plt.subplot(234)
            plt.imshow(angle, cmap='hsv')
            plt.title("Gradient Angle (HSV)")
            plt.colorbar()
            plt.axis('off')
            
            plt.subplot(235)
            # Create binary edge map
            edge_threshold = np.percentile(magnitude, 85)
            edges = (magnitude > edge_threshold).astype(np.uint8) * 255
            plt.imshow(edges, cmap='gray')
            plt.title(f"Edge Map (threshold: {edge_threshold:.1f})")
            plt.axis('off')
            
            plt.subplot(236)
            # Combine x and y gradients for visualization
            combined = np.abs(grad_x) + np.abs(grad_y)
            plt.imshow(combined, cmap='gray')
            plt.title("Combined Gradient (|X| + |Y|)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "02_gradients.jpg"), dpi=100)
            plt.close()
        
        return grad_x, grad_y, magnitude, angle
    
    def find_edge_orientations(self, magnitude, angle):
        """Find dominant edge orientations (chessboard has 2 perpendicular directions)"""
        # Threshold to get strong edges
        mag_threshold = np.percentile(magnitude, 80)
        strong_edges = magnitude > mag_threshold
        
        # Get angles of strong edges
        strong_angles = angle[strong_edges]
        
        # Create histogram of edge orientations
        hist, bins = np.histogram(strong_angles, bins=90, range=(0, 180))
        
        # Find peaks in histogram
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist, height=np.max(hist)*0.3, distance=20)
        
        # Sort peaks by height
        peak_heights = hist[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        peak_angles = bins[peaks][sorted_indices]
        
        print(f"  Found {len(peaks)} dominant edge orientations")
        
        # Chessboard should have 2 main perpendicular orientations
        if len(peak_angles) >= 2:
            # Take top 2 peaks
            angle1, angle2 = peak_angles[0], peak_angles[1]
            
            # Ensure they're approximately perpendicular (90° ± 20°)
            angle_diff = min(
                abs(abs(angle1 - angle2) - 90),
                abs(abs(angle1 - angle2 + 180) - 90),
                abs(abs(angle1 - angle2 - 180) - 90)
            )
            
            print(f"  Top 2 orientations: {angle1:.1f}° and {angle2:.1f}°")
            print(f"  Difference from 90°: {angle_diff:.1f}°")
            
            if angle_diff < 25:  # Reasonably perpendicular
                return angle1, angle2
        
        # If not found, use default perpendicular orientations
        print("  Using default orientations: 0° and 90°")
        return 0, 90
    
    def extract_edge_lines(self, grad_x, grad_y, angle1, angle2):
        """Extract lines along the dominant edge orientations - VECTORIZED VERSION"""
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Vectorized angle calculation
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        angle = np.mod(angle, 180)
        
        # Threshold for strong edges
        mag_threshold = np.percentile(magnitude, 70)
        strong_mask = magnitude > mag_threshold
        
        # Tolerance for angle matching
        tolerance = 10
        
        # Vectorized angle difference calculation
        # Calculate minimum angular difference for both target angles
        diff1 = np.minimum(
            np.abs(angle - angle1),
            np.minimum(
                np.abs(angle - angle1 - 180),
                np.abs(angle - angle1 + 180)
            )
        )
        
        diff2 = np.minimum(
            np.abs(angle - angle2),
            np.minimum(
                np.abs(angle - angle2 - 180),
                np.abs(angle - angle2 + 180)
            )
        )
        
        # Create masks
        mask1 = strong_mask & (diff1 < tolerance)
        mask2 = strong_mask & (diff2 < tolerance)
        
        print(f"  Found {np.sum(mask1)} pixels at {angle1:.1f}°")
        print(f"  Found {np.sum(mask2)} pixels at {angle2:.1f}°")
        
        if self.debug:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            orientation_map = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
            orientation_map[mask1] = [255, 0, 0]  # Red for orientation 1
            orientation_map[mask2] = [0, 255, 0]  # Green for orientation 2
            plt.imshow(orientation_map)
            plt.title(f"Edge Orientations\nRed: {angle1:.1f}°, Green: {angle2:.1f}°")
            plt.axis('off')
            
            # Find lines using Hough Transform on each orientation
            for orientation_idx, (mask, angle_val) in enumerate([(mask1, angle1), (mask2, angle2)]):
                binary_mask = (mask * 255).astype(np.uint8)
                
                # Apply Hough Transform
                lines = cv2.HoughLinesP(
                    binary_mask, 
                    rho=1, 
                    theta=np.pi/180, 
                    threshold=20,
                    minLineLength=20,  # Reduced for better detection
                    maxLineGap=15
                )
                
                if lines is not None:
                    line_image = self.image_rgb.copy()
                    for line in lines[:30]:  # Draw first 30 lines
                        x1, y1, x2, y2 = line[0]
                        color = [255, 0, 0] if orientation_idx == 0 else [0, 255, 0]
                        cv2.line(line_image, (x1, y1), (x2, y2), color, 2)
                    
                    plt.subplot(1, 3, 2 + orientation_idx)
                    plt.imshow(line_image)
                    plt.title(f"Lines at {angle_val:.1f}°: {len(lines)} found")
                    plt.axis('off')
                else:
                    # Create empty subplot if no lines found
                    plt.subplot(1, 3, 2 + orientation_idx)
                    plt.imshow(self.image_rgb)
                    plt.title(f"No lines at {angle_val:.1f}°")
                    plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "03_edge_lines.jpg"))
            plt.close()
        
        return mask1, mask2
    
    def find_grid_points(self, mask1, mask2):
        """Find intersection points of the edge grid"""
        # Convert masks to binary images
        binary1 = (mask1 * 255).astype(np.uint8)
        binary2 = (mask2 * 255).astype(np.uint8)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated1 = cv2.dilate(binary1, kernel, iterations=1)
        dilated2 = cv2.dilate(binary2, kernel, iterations=1)
        
        # Find contours in each orientation
        contours1, _ = cv2.findContours(dilated1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(dilated2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract line segments from contours
        lines1 = []
        lines2 = []
        
        for contour in contours1:
            if len(contour) >= 2:
                # Fit line to contour
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Get two points on the line
                lefty = int((-x*vy/vx) + y)
                righty = int(((self.image.shape[1]-x)*vy/vx) + y)
                
                lines1.append(((0, lefty), (self.image.shape[1]-1, righty)))
        
        for contour in contours2:
            if len(contour) >= 2:
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x*vy/vx) + y)
                righty = int(((self.image.shape[1]-x)*vy/vx) + y)
                lines2.append(((0, lefty), (self.image.shape[1]-1, righty)))
        
        # Find intersections
        intersections = []
        for line1 in lines1[:20]:  # Limit to first 20 lines each
            for line2 in lines2[:20]:
                (x1, y1), (x2, y2) = line1
                (x3, y3), (x4, y4) = line2
                
                # Compute intersection
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) > 1e-6:
                    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                    
                    if 0 <= px < self.image.shape[1] and 0 <= py < self.image.shape[0]:
                        intersections.append((int(px), int(py)))
        
        # Remove duplicates
        unique_intersections = []
        for pt in intersections:
            if not any(np.sqrt((pt[0]-u[0])**2 + (pt[1]-u[1])**2) < 10 for u in unique_intersections):
                unique_intersections.append(pt)
        
        print(f"  Found {len(unique_intersections)} grid intersection points")
        
        if self.debug and unique_intersections:
            points_image = self.image_rgb.copy()
            
            # Draw lines
            for (x1, y1), (x2, y2) in lines1[:10]:
                cv2.line(points_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            for (x1, y1), (x2, y2) in lines2[:10]:
                cv2.line(points_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Draw intersection points
            for px, py in unique_intersections:
                cv2.circle(points_image, (px, py), 5, (0, 0, 255), -1)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(points_image)
            plt.title(f"Grid Lines and {len(unique_intersections)} Intersections")
            plt.axis('off')
            plt.savefig(os.path.join(self.output_dir, "04_grid_points.jpg"))
            plt.close()
        
        return unique_intersections
    
    def find_chessboard_corners(self, grid_points):
        """Find the 4 outer corners of the chessboard from grid points"""
        if len(grid_points) < 4:
            return None
        
        points = np.array(grid_points)
        
        # Find convex hull
        hull = cv2.convexHull(points)
        
        # Simplify to quadrilateral
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
            
            # Order corners: top-left, top-right, bottom-right, bottom-left
            sums = corners.sum(axis=1)
            diffs = corners[:, 0] - corners[:, 1]
            
            top_left = corners[np.argmin(sums)]
            bottom_right = corners[np.argmax(sums)]
            top_right = corners[np.argmax(diffs)]
            bottom_left = corners[np.argmin(diffs)]
            
            ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])
            
            if self.debug:
                corners_image = self.image_rgb.copy()
                for i, (x, y) in enumerate(ordered_corners):
                    cv2.circle(corners_image, (int(x), int(y)), 10, (0, 255, 0), -1)
                    cv2.putText(corners_image, f"C{i}", (int(x)+15, int(y)+15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                plt.figure(figsize=(8, 6))
                plt.imshow(corners_image)
                plt.title("Chessboard Corners")
                plt.axis('off')
                plt.savefig(os.path.join(self.output_dir, "05_corners.jpg"))
                plt.close()
            
            return ordered_corners
        
        return None
    
    def warp_chessboard(self, corners):
        """Warp the chessboard to a rectified view"""
        if corners is None or len(corners) != 4:
            return None, None
        
        # Output size for 8x8 chessboard
        square_size = 60
        output_size = square_size * 8
        
        # Destination points
        dst_points = np.array([
            [0, 0],
            [output_size, 0],
            [output_size, output_size],
            [0, output_size]
        ], dtype=np.float32)
        
        # Compute perspective transform
        transform_matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32), 
            dst_points
        )
        
        # Warp image
        warped = cv2.warpPerspective(
            self.image_rgb, 
            transform_matrix, 
            (output_size, output_size)
        )
        
        if self.debug:
            plt.figure(figsize=(10, 8))
            plt.imshow(warped)
            plt.title("Rectified Chessboard")
            plt.axis('off')
            plt.savefig(os.path.join(self.output_dir, "06_warped.jpg"))
            plt.close()
        
        return warped, transform_matrix
    
    def display_final_result(self, corners, transform_matrix):
        """Create final visualization"""
        result_image = self.image_rgb.copy()
        
        # Draw corners
        for i, (x, y) in enumerate(corners):
            cv2.circle(result_image, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(result_image, f"C{i}", (int(x)+15, int(y)+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw quadrilateral
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(result_image, [pts], True, (255, 0, 0), 3)
        
        # Draw grid
        square_size = 60
        grid_size = 8
        
        # Draw horizontal grid lines
        for i in range(grid_size + 1):
            y = i * square_size
            src_points = np.array([[[0, y], [grid_size * square_size, y]]], 
                                 dtype=np.float32)
            dst_points = cv2.perspectiveTransform(src_points, 
                                                np.linalg.inv(transform_matrix))
            pt1 = tuple(dst_points[0][0].astype(int))
            pt2 = tuple(dst_points[0][1].astype(int))
            cv2.line(result_image, pt1, pt2, (0, 0, 255), 1)
        
        # Draw vertical grid lines
        for i in range(grid_size + 1):
            x = i * square_size
            src_points = np.array([[[x, 0], [x, grid_size * square_size]]], 
                                 dtype=np.float32)
            dst_points = cv2.perspectiveTransform(src_points, 
                                                np.linalg.inv(transform_matrix))
            pt1 = tuple(dst_points[0][0].astype(int))
            pt2 = tuple(dst_points[0][1].astype(int))
            cv2.line(result_image, pt1, pt2, (0, 0, 255), 1)
        
        if self.debug:
            plt.figure(figsize=(12, 10))
            plt.imshow(result_image)
            plt.title("Final Detection Result")
            plt.axis('off')
            plt.savefig(os.path.join(self.output_dir, "07_final_result.jpg"))
            plt.close()
        
        # Save as PNG
        cv2.imwrite(os.path.join(self.output_dir, "final_result.png"), 
                   cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    def detect_chessboard(self, image_path):
        """Main detection pipeline"""
        print("=" * 60)
        print("CHESSBOARD DETECTION FROM GRADIENTS")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Load image
            print("1. Loading image...")
            self.load_image(image_path)
            print(f"   Size: {self.image.shape[1]}x{self.image.shape[0]}")
            
            # Step 2: Compute gradients
            print("2. Computing gradients...")
            grad_x, grad_y, magnitude, angle = self.compute_gradients()
            
            # Step 3: Find edge orientations
            print("3. Finding edge orientations...")
            angle1, angle2 = self.find_edge_orientations(magnitude, angle)
            
            # Step 4: Extract edge lines
            print("4. Extracting edge lines...")
            mask1, mask2 = self.extract_edge_lines(grad_x, grad_y, angle1, angle2)
            
            # Step 5: Find grid points
            print("5. Finding grid intersection points...")
            grid_points = self.find_grid_points(mask1, mask2)
            
            if not grid_points or len(grid_points) < 9:  # Need at least 3x3 grid
                print(f"   ✗ Not enough grid points: {len(grid_points) if grid_points else 0}")
                return None
            
            # Step 6: Find corners
            print("6. Finding chessboard corners...")
            corners = self.find_chessboard_corners(grid_points)
            
            if corners is None:
                print("   ✗ Could not find 4 corners")
                return None
            
            # Step 7: Warp chessboard
            print("7. Warping chessboard...")
            warped, transform_matrix = self.warp_chessboard(corners)
            
            if warped is None:
                print("   ✗ Could not warp image")
                return None
            
            # Step 8: Display results
            print("8. Creating final visualization...")
            self.display_final_result(corners, transform_matrix)
            
            elapsed_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("✓ CHESSBOARD DETECTED!")
            print(f"  Time: {elapsed_time:.2f} seconds")
            print(f"  Results saved to: {self.output_dir}/")
            print("=" * 60)
            
            return {
                'corners': corners,
                'warped_image': warped,
                'transform_matrix': transform_matrix,
                'elapsed_time': elapsed_time
            }
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    # Test with your image
    image_path = "/home/amine/workspace/python/yolov8/chessModel/dataset/images/test/4e3117459d759798537eb52cf5bf534d_jpg.rf.ec961b62d4b0e131fae760ed1f80836b.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    detector = ChessboardDetector(debug=True, output_dir="gradient_detection")
    result = detector.detect_chessboard(image_path)
    
    if result:
        print(f"\nSuccess! Corners found at:")
        for i, (x, y) in enumerate(result['corners']):
            print(f"  Corner {i}: ({x:.0f}, {y:.0f})")
        
        # Show summary
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(132)
        result_img = cv2.imread("gradient_detection/final_result.png")
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        plt.imshow(result_img)
        plt.title("Detection")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(result['warped_image'])
        plt.title("Rectified")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("detection_summary.jpg")
        plt.close()
        
        print("\nCheck 'gradient_detection/' folder for all intermediate steps")
        print("Check 'detection_summary.jpg' for overview")
    else:
        print("\nDetection failed. The image might not contain a clear chessboard.")


if __name__ == "__main__":
    main()