from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login
from django.shortcuts import render, redirect
from django.contrib import messages

def login(request):
    if request.method == "POST": 
        username = request.POST['username']
        pass1 = request.POST['pass1']

        user = authenticate(request, username=username, password=pass1)

        if user is not None:
            auth_login(request, user)
            return render(request, "")
        else:
            messages.error(request, "Invalid Credentials")
            return redirect('login') 

    return render(request, 'LOGIN.html')

def register(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        if pass1 != pass2:
            messages.error(request, "Passwords do not match")
            return redirect('register')
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken. Please choose a different one.")
            return redirect('register')
        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return redirect('register')

        myuser = User.objects.create_user(username, email, pass1)
        myuser.save()

        messages.success(request, "Your Account has been successfully created")
        return redirect('login') 
    return render(request, 'register.html')

def index(request):
    return render(request, 'index.html')


'''from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define teacher zone (x1, y1, x2, y2) as a rectangular area on the frame
teacher_zone = (50, 100, 200, 300, 200)  # Adjust these values as per your camera setup

def detect_gaze(frame):
    # Convert frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Draw the teacher zone rectangle
    cv2.rectangle(frame, (teacher_zone[0], teacher_zone[1]), (teacher_zone[2], teacher_zone[3]), (255, 0, 0), 2)
    
    # Check if any faces are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates for the nose tip (landmark for face position)
            nose_tip = [face_landmarks.landmark[1].x, face_landmarks.landmark[1].y]
            
            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            nose_tip = (int(nose_tip[0] * w), int(nose_tip[1] * h))
            
            # Skip gaze detection for faces within the teacher zone
            if teacher_zone[0] <= nose_tip[0] <= teacher_zone[2] and teacher_zone[1] <= nose_tip[1] <= teacher_zone[3]:
                continue  # Skip this face as it is within the teacher zone

            # Otherwise, proceed with gaze detection for faces outside the teacher zone
            left_eye = [face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]
            right_eye = [face_landmarks.landmark[263].x, face_landmarks.landmark[263].y]

            # Convert eye coordinates to pixel values
            left_eye = (int(left_eye[0] * w), int(left_eye[1] * h))
            right_eye = (int(right_eye[0] * w), int(right_eye[1] * h))
            
            # Determine if gaze is directed at the teacher zone (optional for logging purposes)
            gaze = "Looking Outside Teacher Zone"
            
            # Draw gaze direction and label on frame
            cv2.putText(frame, gaze, (left_eye[0] - 10, left_eye[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, left_eye, 5, (0, 255, 0), -1)
            cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)
            cv2.circle(frame, nose_tip, 5, (0, 0, 255), -1)
    
    return frame

def generate_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect gaze in the frame
        frame = detect_gaze(frame)
        
        # Encode frame to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        # Stream frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    cap.release()

def video_feed(request):
    return StreamingHttpResponse(generate_video(), content_type='multipart/x-mixed-replace; boundary=frame')
'''


from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login
from django.contrib import messages
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import torch
import numpy as np
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image

# Load the pre-trained ResNet model for emotion detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = models.resnet18(weights='IMAGENET1K_V1')
emotion_model.fc = torch.nn.Linear(emotion_model.fc.in_features, 7)  # Update for 7 classes
emotion_model.load_state_dict(torch.load("emotion_detection.pth", map_location=torch.device('cpu')))
emotion_model = emotion_model.to(device)
emotion_model.eval()

# Initialize Mediapipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define teacher zone (x1, y1, x2, y2) as a rectangular area on the frame
teacher_zone = (50, 100, 200, 300)  # Adjust these values as per your camera setup

# Data transformation for emotion detection
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel (ResNet expects 3 channels)
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per pre-trained model
])

def detect_gaze(frame):
    # Convert frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Draw the teacher zone rectangle
    cv2.rectangle(frame, (teacher_zone[0], teacher_zone[1]), (teacher_zone[2], teacher_zone[3]), (255, 0, 0), 2)
    
    # Check if any faces are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates for the nose tip (landmark for face position)
            nose_tip = [face_landmarks.landmark[1].x, face_landmarks.landmark[1].y]
            
            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            nose_tip = (int(nose_tip[0] * w), int(nose_tip[1] * h))
            
            # Skip gaze detection for faces within the teacher zone
            if teacher_zone[0] <= nose_tip[0] <= teacher_zone[2] and teacher_zone[1] <= nose_tip[1] <= teacher_zone[3]:
                continue  # Skip this face as it is within the teacher zone

            # Otherwise, proceed with gaze detection for faces outside the teacher zone
            left_eye = [face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]
            right_eye = [face_landmarks.landmark[263].x, face_landmarks.landmark[263].y]

            # Convert eye coordinates to pixel values
            left_eye = (int(left_eye[0] * w), int(left_eye[1] * h))
            right_eye = (int(right_eye[0] * w), int(right_eye[1] * h))
            
            # Determine if gaze is directed at the teacher zone (optional for logging purposes)
            gaze = "Looking Outside Teacher Zone"
            
            # Draw gaze direction and label on frame
            cv2.putText(frame, gaze, (left_eye[0] - 10, left_eye[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, left_eye, 5, (0, 255, 0), -1)
            cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)
            cv2.circle(frame, nose_tip, 5, (0, 0, 255), -1)
    
    return frame

def predict_emotion(frame):
    # Convert frame to PIL image
    pil_image = Image.fromarray(frame)
    
    # Apply transformations to the image
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Get emotion prediction
    with torch.no_grad():
        output = emotion_model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    # Return the predicted class (Emotion)
    emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[predicted.item()]

def generate_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect gaze in the frame
        frame = detect_gaze(frame)
        
        # Predict emotion
        emotion = predict_emotion(frame)
        
        # Add emotion label on the frame
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode frame to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        # Stream frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    cap.release()

def video_feed(request):
    return StreamingHttpResponse(generate_video(), content_type='multipart/x-mixed-replace; boundary=frame')
