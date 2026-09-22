import cv2
import numpy as np
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import DeepFace
from deepface import DeepFace

# Initialize OpenCV Haar Cascade for real-time face detection (fast and lightweight)
# This will be used for smooth real-time detection
haar_cascade_path = 'haarcascade_frontalface_default.xml'
try:
    if os.path.exists(haar_cascade_path):
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        if not face_cascade.empty():
            OPENCV_DETECTION_AVAILABLE = True
        else:
            OPENCV_DETECTION_AVAILABLE = False
    else:
        OPENCV_DETECTION_AVAILABLE = False
except Exception:
    OPENCV_DETECTION_AVAILABLE = False

def capture_and_save_face(name, roll_no):
    """
    Capture and save a face image from webcam.
    Uses MediaPipe for real-time detection (fast) and DeepFace for final capture (accurate).
    
    Args:
        name (str): Name of the student
        roll_no (str): Roll number of the student (used for filename)
    """
    # Create faces directory if it doesn't exist
    if not os.path.exists('faces'):
        os.makedirs('faces')
    
    # Open webcam (camera index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    print(f"Capturing face for {name} (Roll No: {roll_no})")
    print("Press SPACEBAR to capture face, 'q' to quit")
    
    if OPENCV_DETECTION_AVAILABLE:
        print("Using OpenCV Haar Cascade for real-time detection (smooth)")
        print("Using DeepFace + RetinaFace for final capture (accurate)")
    else:
        print("Using DeepFace for all operations (may be slower)")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Use OpenCV Haar Cascade for real-time face detection (fast)
        if OPENCV_DETECTION_AVAILABLE:
            # Convert to grayscale for Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using Haar Cascade
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Draw face detections on the frame
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add label with name
                    label = f"{name} ({roll_no})"
                    cv2.putText(display_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw guide box if no face detected
                height_frame, width_frame = display_frame.shape[:2]
                center_x, center_y = width_frame // 2, height_frame // 2
                box_size = 200
                cv2.rectangle(display_frame, (center_x - box_size//2, center_y - box_size//2), 
                             (center_x + box_size//2, center_y + box_size//2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{name} ({roll_no}) - Position face in center", 
                           (center_x - box_size//2, center_y - box_size//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Fallback to DeepFace if MediaPipe not available (slower)
            try:
                faces = DeepFace.extract_faces(
                    frame, 
                    detector_backend='retinaface',
                    enforce_detection=False,
                    align=True
                )
                
                if faces:
                    for face in faces:
                        facial_area = face['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                        confidence = face['confidence']
                        
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{name} ({roll_no}) - {confidence:.2f}"
                        cv2.putText(display_frame, label, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Draw guide box
                    height_frame, width_frame = display_frame.shape[:2]
                    center_x, center_y = width_frame // 2, height_frame // 2
                    box_size = 200
                    cv2.rectangle(display_frame, (center_x - box_size//2, center_y - box_size//2), 
                                 (center_x + box_size//2, center_y + box_size//2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{name} ({roll_no}) - Position face in center", 
                               (center_x - box_size//2, center_y - box_size//2 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                # Show guide box on error
                height_frame, width_frame = display_frame.shape[:2]
                center_x, center_y = width_frame // 2, height_frame // 2
                box_size = 200
                cv2.rectangle(display_frame, (center_x - box_size//2, center_y - box_size//2), 
                             (center_x + box_size//2, center_y + box_size//2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{name} ({roll_no}) - Position face in center", 
                           (center_x - box_size//2, center_y - box_size//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        window_name = 'Face Capture - OpenCV (Smooth)' if OPENCV_DETECTION_AVAILABLE else 'Face Capture - DeepFace'
        cv2.imshow(window_name, display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Press SPACEBAR to capture
        if key == ord(' '):
            print("Processing final capture with DeepFace + RetinaFace...")
            
            # Use DeepFace with RetinaFace for accurate final capture
            try:
                faces = DeepFace.extract_faces(
                    frame, 
                    detector_backend='retinaface',
                    enforce_detection=False,
                    align=True
                )
                
                if faces:
                    # Crop the first detected face with high accuracy
                    facial_area = faces[0]['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    
                    face_image = frame[y:y + h, x:x + w]
                    print("Face detected and cropped successfully")
                else:
                    # Capture center region when face detection fails
                    height_frame, width_frame = frame.shape[:2]
                    center_x, center_y = width_frame // 2, height_frame // 2
                    box_size = 200
                    x = max(0, center_x - box_size // 2)
                    y = max(0, center_y - box_size // 2)
                    w = min(box_size, width_frame - x)
                    h = min(box_size, height_frame - y)
                    face_image = frame[y:y + h, x:x + w]
                    print("Face detection failed - captured center region")
            except Exception as e:
                # Fallback to center capture on error
                height_frame, width_frame = frame.shape[:2]
                center_x, center_y = width_frame // 2, height_frame // 2
                box_size = 200
                x = max(0, center_x - box_size // 2)
                y = max(0, center_y - box_size // 2)
                w = min(box_size, width_frame - x)
                h = min(box_size, height_frame - y)
                face_image = frame[y:y + h, x:x + w]
                print(f"Error during capture: {str(e)}")
            
            # Save the face image
            save_path = os.path.join('faces', f"{roll_no}.jpg")
            cv2.imwrite(save_path, face_image)
            print(f"Face saved as {save_path}")
            
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            return True
        
        # Press 'q' to quit without saving
        elif key == ord('q'):
            print("Capture cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    return False

def get_face_encoding(image_path):
    """
    Get face encoding from an image using DeepFace with Facenet512 model.
    Returns a 512-dimensional embedding vector.
    
    Args:
        image_path (str): Path to the face image
        
    Returns:
        numpy.ndarray: 512-dimensional face embedding vector
        
    Raises:
        ValueError: If no face is detected in the image
        Exception: If there's an error processing the image
    """
    try:
        # Use DeepFace to get face embeddings with Facenet512 model
        # Using retinaface detector which is more reliable than opencv
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name='Facenet512',
            detector_backend='retinaface',
            enforce_detection=True  # Raise error if no face detected
        )
        
        if embedding_objs and len(embedding_objs) > 0:
            # Get the first face embedding
            embedding = embedding_objs[0]['embedding']
            embedding_array = np.array(embedding)
            
            # Verify it's a 512-dimensional vector
            if len(embedding_array) != 512:
                raise ValueError(f"Expected 512-dimensional embedding, got {len(embedding_array)}")
            
            return embedding_array
        else:
            raise ValueError(f"No face detected in {image_path}")
            
    except ValueError as e:
        raise ValueError(f"Face detection failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {str(e)}")

def compare_faces(face_encoding1, face_encoding2, threshold=0.4):
    """
    Compare two face encodings and return if they match.
    Uses Euclidean distance for Facenet512 embeddings.
    
    Args:
        face_encoding1 (numpy.ndarray): First face embedding (512-dimensional)
        face_encoding2 (numpy.ndarray): Second face embedding (512-dimensional)
        threshold (float): Distance threshold for matching (lower = stricter)
                           Typical range for Facenet512: 0.3-0.5
        
    Returns:
        tuple: (bool indicating match, float distance)
    """
    try:
        # Calculate Euclidean distance between embeddings
        distance = np.linalg.norm(face_encoding1 - face_encoding2)
        
        # Lower distance means more similar faces
        is_match = distance < threshold
        
        return is_match, distance
        
    except Exception as e:
        raise Exception(f"Error comparing faces: {str(e)}")

def count_faces(image_path):
    """
    Count the number of faces detected in an image using DeepFace with RetinaFace detector.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        int: Number of faces detected
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend='retinaface',
            enforce_detection=False,
            align=False
        )
        if not faces:
            return 0
        # DeepFace with enforce_detection=False returns a dummy face with confidence 0 if no face is found
        real_faces = [f for f in faces if f.get('confidence', 0) > 0]
        return len(real_faces)
    except ValueError:
        return 0
    except Exception as e:
        return 0

if __name__ == "__main__":
    # Test the functions
    print("Face utilities module loaded successfully!")
    if OPENCV_DETECTION_AVAILABLE:
        print("Using OpenCV Haar Cascade for real-time detection and DeepFace + RetinaFace for accurate capture")
    else:
        print("Using DeepFace for all operations")
    print("Facenet512 model for 512-dimensional embeddings")