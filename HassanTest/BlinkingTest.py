import cv2
import numpy as np

def detect_face_and_eyes(image_source=0, show_video=True):
    """
    Detect faces and eyes using Haar Cascades
    
    Args:
        image_source: Can be:
            - 0 or 1 for webcam
            - Path to an image file
            - Path to a video file
        show_video: If True, shows real-time detection for video/webcam
    """
    
    # Load the Haar Cascade classifiers for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Check if cascades loaded successfully
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return
    if eye_cascade.empty():
        print("Error: Could not load eye cascade classifier")
        return
    
    # Check if source is webcam (integer) or file (string)
    if isinstance(image_source, int):
        # Webcam mode
        cap = cv2.VideoCapture(image_source)
        process_video(cap, face_cascade, eye_cascade, show_video)
    elif image_source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Image mode
        process_image(image_source, face_cascade, eye_cascade)
    else:
        # Video file mode
        cap = cv2.VideoCapture(image_source)
        process_video(cap, face_cascade, eye_cascade, show_video)


def process_image(image_path, face_cascade, eye_cascade):
    """Process a single image for face and eye detection"""
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"Found {len(faces)} face(s)")
    
    # Draw rectangles around faces and detect eyes within each face
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Define region of interest for eyes (within the face)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=10,
            minSize=(15, 15)
        )
        
        print(f"  Found {len(eyes)} eye(s) in this face")
        
        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display the result
    cv2.imshow('Face and Eye Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    output_path = 'detected_' + image_path.split('/')[-1]
    cv2.imwrite(output_path, img)
    print(f"Result saved as {output_path}")


def process_video(cap, face_cascade, eye_cascade, show_video=True):
    """Process video stream for face and eye detection"""
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Press 'q' to quit, 's' to save current frame")
    frame_count = 0
    
    while True:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Define region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=10,
                minSize=(15, 15),
                maxSize=(int(w/3), int(h/3))  # Eyes shouldn't be too large relative to face
            )
            
            # Draw rectangles around eyes (limit to 2 eyes per face)
            eye_count = 0
            for (ex, ey, ew, eh) in eyes:
                if eye_count < 2:  # Limit to 2 eyes per face
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    eye_count += 1
        
        # Display FPS
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if show_video:
            # Display the frame
            cv2.imshow('Face and Eye Detection - Press Q to quit', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f'captured_frame_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
                frame_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function with example usage"""
    
    print("Face and Eye Detection using Haar Cascades")
    print("=" * 50)
    print("\nOptions:")
    print("1. Use webcam (real-time detection)")
    print("2. Use image file")
    print("3. Use video file")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        # Webcam mode
        print("\nStarting webcam... Press 'q' to quit, 's' to save frame")
        detect_face_and_eyes(0, show_video=True)
        
    elif choice == '2':
        # Image mode
        image_path = input("Enter path to image file: ").strip()
        detect_face_and_eyes(image_path, show_video=True)
        
    elif choice == '3':
        # Video file mode
        video_path = input("Enter path to video file: ").strip()
        detect_face_and_eyes(video_path, show_video=True)
        
    else:
        print("Invalid choice!")
        
        # Run with default webcam
        print("\nRunning with default webcam...")
        detect_face_and_eyes(0, show_video=True)


# Alternative simplified version for quick testing
def simple_face_eye_detection():
    """Simplified version for quick testing with webcam"""
    
    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Get face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                # Draw eye rectangles
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Face and Eye Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
    # Or run the simplified version directly:
    # simple_face_eye_detection()