import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
from datetime import datetime
import pytz
import os
import base64
from io import BytesIO
from PIL import Image

# ------------------ DATABASE CONNECTION ------------------
def get_database():
    CONNECTION_URL = os.getenv("CONNECTION_URL")
    if not CONNECTION_URL:
        raise Exception("Missing CONNECTION_URL environment variable")
    client = MongoClient(CONNECTION_URL)
    return client['AttendEase']

# ------------------ FACE ENCODING FUNCTIONS ------------------
def getEncodings():
    """Retrieve all stored face encodings from database"""
    dbname = get_database()
    collection_name = dbname["encodings"]
    items = collection_name.find({})
    
    known_images = []
    encodings = []
    
    for i in items:
        i.pop("_id")
        for name, encoding in i.items():
            known_images.append(name)
            encodings.append(np.array(encoding))
    
    return known_images, encodings

def get_face_embedding(img_file):
    """Extract face embedding using DeepFace"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{datetime.now().timestamp()}.jpg"
        
        if hasattr(img_file, 'save'):
            img_file.save(temp_path)
        else:
            # If it's already a file path
            temp_path = img_file
        
        # Get face embedding using DeepFace
        embedding_objs = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",  # Fast and accurate
            enforce_detection=True
        )
        
        # Clean up temp file
        if os.path.exists(temp_path) and hasattr(img_file, 'save'):
            os.remove(temp_path)
        
        if embedding_objs:
            return embedding_objs[0]["embedding"]
        return None
        
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        # Clean up temp file on error
        if os.path.exists(temp_path) and hasattr(img_file, 'save'):
            os.remove(temp_path)
        return None

def update_face(imgName, addImg):
    """Add new face encoding to database"""
    try:
        embedding = get_face_embedding(addImg)
        
        if embedding is None:
            return False
        
        # Store in database
        pair = {imgName: embedding}
        dbname = get_database()
        collection_name = dbname["encodings"]
        collection_name.insert_one(pair)
        return True
        
    except Exception as e:
        print("Error while updating face:", e)
        return False

def compare_faces(baseImg, threshold=0.6):
    """Compare uploaded face with all stored faces"""
    try:
        # Get embedding for uploaded image
        test_embedding = get_face_embedding(baseImg)
        
        if test_embedding is None:
            return False
        
        # Get all known encodings
        known_images, encodings = getEncodings()
        
        if not encodings:
            return False
        
        # Calculate cosine similarity with all known faces
        test_embedding = np.array(test_embedding)
        min_distance = float('inf')
        matched_name = None
        
        for i, known_encoding in enumerate(encodings):
            known_encoding = np.array(known_encoding)
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(test_embedding - known_encoding)
            
            if distance < min_distance:
                min_distance = distance
                matched_name = known_images[i]
        
        # Check if best match is below threshold
        if min_distance < threshold:
            return matched_name.split(".")[0]
        
        return False
        
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False

# ------------------ ATTENDANCE FUNCTION ------------------
def update_attendance(id, status):
    """Record attendance in database"""
    IST = pytz.timezone('Asia/Kolkata')
    now = datetime.now(IST)
    momentDate = now.strftime("%d/%m/%Y")
    momentTime = now.strftime("%H:%M:%S")
    db = get_database()
    collection_name = db[momentDate]
    data = {"id": id, "status": status, "date": momentDate, "time": momentTime}
    try:
        collection_name.insert_one(data)
        return True
    except Exception as e:
        print("Error updating attendance:", e)
        return False

# ------------------ FLASK APP SETUP ------------------
app = Flask(__name__)
CORS(app)

@app.route('/face_match', methods=['POST'])
def face_match():
    """Match uploaded face with stored faces"""
    if 'file1' in request.files:
        file1 = request.files.get('file1')
        response = compare_faces(file1)
        if response:
            id = response
            status = file1.filename
            update_attendance(id, status)
        return jsonify({"status": response})
    return jsonify({"error": "No file provided"}), 400

@app.route('/add_face', methods=['POST'])
def add_face():
    """Add new face to database"""
    if 'file1' in request.files:
        file1 = request.files.get('file1')
        imgName = file1.filename.split(".")[0]
        response = update_face(imgName, file1)
        return jsonify({"status": response})
    return jsonify({"error": "No file provided"}), 400

@app.route('/', methods=['GET'])
def home():
    return 'AttendEase APP API is Running Successfully! ✅'

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# ------------------ MAIN ENTRY POINT ------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)