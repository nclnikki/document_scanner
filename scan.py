import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
import img2pdf

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped
def enhance_document(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7
    )
    
    # Apply some morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def find_document_contour(image, min_area_percentage=0.1, max_area_percentage=0.95):
    # Get image dimensions
    height, width = image.shape[:2]
    image_area = height * width
    
    # Make a copy of the image for display
    display = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while reducing noise
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Create CLAHE object for adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    
    # Auto-detect appropriate threshold values using Otsu's method
    high_thresh, _ = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = high_thresh * 0.5
    
    # Apply Canny edge detection
    edged = cv2.Canny(equalized, low_thresh, high_thresh)
    
    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Debugging: Draw all contours
    contour_img = np.zeros_like(image)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Define area constraints
    min_area = image_area * min_area_percentage
    max_area = image_area * max_area_percentage
    
    document_contour = None
    
    # Loop through contours
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip if contour is too small or too large
        if area < min_area or area > max_area:
            continue
        
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if it's a quadrilateral
        if len(approx) == 4:
            # Check if the contour is convex
            if cv2.isContourConvex(approx):
                # Additional validation: angles should be close to 90 degrees
                if validate_rectangle(approx):
                    document_contour = approx
                    break
    
    # If no 4-point contour is found, try with the largest contour that has 4-8 points
    if document_contour is None:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if 4 <= len(approx) <= 8:
                # If we have more than 4 points, get the 4 corners
                if len(approx) > 4:
                    approx = get_corner_points(approx)
                
                document_contour = approx
                break
    
    # If still no contour found, try Hough lines method as fallback
    if document_contour is None:
        document_contour = detect_document_with_hough_lines(dilated, image_area)
    
    return document_contour, {
        "edge_image": edged,
        "contour_image": contour_img,
        "preprocessed": equalized
    }

def validate_rectangle(points):
    """Check if the quadrilateral is roughly rectangular by verifying angles."""
    points = points.reshape(4, 2)
    
    # Calculate vectors between consecutive points
    vectors = [
        points[1] - points[0],
        points[2] - points[1],
        points[3] - points[2],
        points[0] - points[3]
    ]
    
    # Calculate angles between vectors
    angles = []
    for i in range(4):
        v1 = vectors[i]
        v2 = vectors[(i + 1) % 4]
        
        # Calculate the angle using dot product
        dot = np.dot(v1, v2)
        det = v1[0] * v2[1] - v1[1] * v2[0]
        angle = np.abs(np.degrees(np.arctan2(det, dot)))
        angles.append(angle)
    
    # Check if angles are roughly 90 degrees (with tolerance)
    for angle in angles:
        if abs(angle - 90) > 20:  # 20 degree tolerance
            return False
    
    return True

def get_corner_points(polygon):
    # Extract the four corner points from a polygon with more than 4 points.
    # Convert to a simpler representation
    points = polygon.reshape(-1, 2)
    
    # Find the convex hull
    hull = cv2.convexHull(points)
    
    # Approximate to get the corners
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    
    # If we still have more than 4 points, get the 4 most extreme ones
    if len(approx) > 4:
        # Get extreme points (top-left, top-right, bottom-right, bottom-left)
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        top_left = points[np.argmin(s)]
        bottom_right = points[np.argmax(s)]
        top_right = points[np.argmin(diff)]
        bottom_left = points[np.argmax(diff)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left]).reshape(-1, 1, 2)
    
    return approx

def detect_document_with_hough_lines(edge_image, image_area):
    # Fallback method to detect document using Hough lines.
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edge_image, 1, np.pi/180, threshold=100)
    
    if lines is None or len(lines) < 4:
        return None
    
    # Group lines by similar angles
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        rho, theta = line[0]
        if 0 <= theta < np.pi/4 or 3*np.pi/4 <= theta < np.pi:
            # This is a vertical line
            vertical_lines.append((rho, theta))
        else:
            # This is a horizontal line
            horizontal_lines.append((rho, theta))
    
    # Get the two strongest horizontal and vertical lines
    horizontal_lines = sorted(horizontal_lines, key=lambda x: abs(x[0]))[:2]
    vertical_lines = sorted(vertical_lines, key=lambda x: abs(x[0]))[:2]
    
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None
    
    # Find intersections to get corners
    corners = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            corner = get_intersection(h_line, v_line)
            if corner is not None:
                corners.append(corner)
    
    if len(corners) != 4:
        return None
    
    # Sort corners to get them in correct order
    corners = np.array(corners)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    
    # Order: top-left, top-right, bottom-right, bottom-left
    ordered_corners = np.zeros((4, 2), dtype=np.float32)
    ordered_corners[0] = corners[np.argmin(s)]
    ordered_corners[2] = corners[np.argmax(s)]
    ordered_corners[1] = corners[np.argmin(diff)]
    ordered_corners[3] = corners[np.argmax(diff)]
    
    return ordered_corners.reshape(-1, 1, 2).astype(np.int32)

def get_intersection(line1, line2):
    # Find intersection point of two lines in rho-theta form.
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    
    try:
        x, y = np.linalg.solve(A, b)
        return [x, y]
    except np.linalg.LinAlgError:
        return None
    

def main():
    st.title("Document Scanner App")
    
    # Initialize session state
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'scanned_image' not in st.session_state:
        st.session_state.scanned_image = None
    
    # Sidebar settings
    st.sidebar.title("Settings")
    
    # Advanced settings expander
    with st.sidebar.expander("Advanced Detection Settings"):
        min_area_pct = st.slider("Min Document Area (%)", 1, 50, 10, help="Minimum document size as % of image")
        max_area_pct = st.slider("Max Document Area (%)", 50, 99, 95, help="Maximum document size as % of image")
        debug_mode = st.checkbox("Processing Steps", value=False, help="Show intermediate processing steps")
    
    # Camera input
    img_file = st.camera_input("Take a picture of your document")
    
    if img_file is not None:
        # Convert to OpenCV format
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.session_state.current_image = cv_img
        
        # Document scanning with improved detection
        document_contour, debug_images = find_document_contour(
            cv_img, 
            min_area_percentage=min_area_pct/100, 
            max_area_percentage=max_area_pct/100
        )
        
        # Show debug images if enabled
        if debug_mode:
            st.subheader("Process Visualization")
            debug_col1, debug_col2, debug_col3 = st.columns(3)
            
            with debug_col1:
                st.caption("Original Image")
                st.image(cv_img, channels="BGR")
            
            with debug_col2:
                st.caption("Preprocessed")
                st.image(debug_images["preprocessed"])
            
            with debug_col3:
                st.caption("Edge Detection")
                st.image(debug_images["edge_image"])
            
            st.caption("Detected Contours")
            st.image(debug_images["contour_image"], channels="BGR")
        
        if document_contour is not None:
            # Draw contour on copy of original image
            display_img = cv_img.copy()
            cv2.drawContours(display_img, [document_contour], -1, (0, 255, 0), 2)
            
            # Transform the image to get the document perspective
            document_pts = document_contour.reshape(4, 2)
            warped = four_point_transform(cv_img, document_pts)
            
            # Image enhancement options
            st.sidebar.subheader("Document Enhancement")
            enhancement_option = st.sidebar.radio(
                "Enhancement Mode",
                ["Original", "Standard Enhancement", "Adaptive Enhancement", "High Contrast"],
                index=1
            )
            
            if enhancement_option == "Original":
                st.session_state.scanned_image = warped
                result_image = warped
                display_channels = "BGR"
            elif enhancement_option == "Standard Enhancement":
                enhanced = enhance_document(warped)
                st.session_state.scanned_image = enhanced
                result_image = enhanced
                display_channels = "GRAY"
            elif enhancement_option == "Adaptive Enhancement":
                enhanced = adaptive_enhance_document(warped)
                st.session_state.scanned_image = enhanced
                result_image = enhanced
                display_channels = "GRAY"
            else:  # High Contrast
                enhanced = high_contrast_enhance(warped)
                st.session_state.scanned_image = enhanced
                result_image = enhanced
                display_channels = "GRAY"
            
            # Show original with contour and processed images
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detected Document")
                st.image(display_img, channels="BGR")
            with col2:
                st.subheader("Scanned Result")
                if display_channels == "BGR":
                    st.image(result_image, channels="BGR")
                else:
                    st.image(result_image)

            if st.button("Save Image"):
                if display_channels == "BGR":
                    img_to_save = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                else:
                    img_to_save = Image.fromarray(result_image)
            
                # Add to session state
                img_byte_arr = io.BytesIO()
                img_to_save.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                st.session_state.captured_images.append(img_byte_arr)
                st.success(f"Image saved! Total images: {len(st.session_state.captured_images)}")
        
        else:
            st.error("No document detected. Please try the following:")
            st.markdown("""
            - Ensure your document is clearly visible against a contrasting background
            - Make sure there is adequate lighting
            - Hold the camera steady and try to capture the full document
            - Try adjusting the minimum and maximum document area in settings
            """)
            st.image(cv_img, channels="BGR")

    # Show captured images
    if st.session_state.captured_images:
        st.subheader("Captured Images")
        cols = st.columns(3)
        for idx, img_bytes in enumerate(st.session_state.captured_images):
            with cols[idx % 3]:
                st.image(img_bytes, caption=f"Image {idx + 1}")

        # Generate PDF
        if st.button("Generate PDF"):
            try:
                # Create PDF
                pdf_bytes = io.BytesIO()
                
                # Convert images to PIL format for PDF creation
                pil_images = []
                for img_bytes in st.session_state.captured_images:
                    pil_images.append(Image.open(io.BytesIO(img_bytes)))
                
                # Save as PDF
                pdf_bytes.write(img2pdf.convert([io.BytesIO(img_bytes) for img_bytes in st.session_state.captured_images]))
                
                # Offer download
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes.getvalue(),
                    file_name=f"scanned_document_{timestamp}.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

        # Clear all images
        if st.button("Clear All Images"):
            st.session_state.captured_images = []
            st.success("All images cleared!")

def adaptive_enhance_document(image):
    # Enhanced document processing with adaptive thresholding.
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Create CLAHE object for adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    # Apply adaptive thresholding with larger block size for better handling of shadows
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def high_contrast_enhance(image):
    # High contrast enhancement for documents with faint text.
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

if __name__ == "__main__":
    main()
