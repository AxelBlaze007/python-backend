import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='face_recognition_models')

from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition as fr
from pymongo import MongoClient
import numpy as np
from datetime import datetime
import pytz
import os

# ------------------ DATABASE CONNECTION ------------------
def get_database():
    CONNECTION_URL = os.getenv("CONNECTION_URL")
    if not CONNECTION_URL:
        raise Exception("Missing CONNECTION_URL environment variable")
    client = MongoClient(CONNECTION_URL)
    return client['AttendEase']

# ------------------ FACE ENCODING FUNCTIONS ------------------
def getEncodings():
    dbname = get_database()
    collection_name = dbname["encodings"]
    items = collection_name.find({})
    for i in items:
        i.pop("_id")
        known_images = list(i.keys())
        values = list(i.values())
        encodings = []
        for v in values:
            encodings.append(np.array(v))
        return known_images, encodings

def update_face(imgName, addImg):
    addImage = fr.load_image_file(addImg)
    try:
        image_encoding = list(fr.face_encodings(addImage)[0])
    except IndexError:
        return False
    try:
        pair = {imgName: image_encoding}
        dbname = get_database()
        collection_name = dbname["encodings"]
        collection_name.insert_one(pair)
        return True
    except Exception as e:
        print("Error while updating face:", e)
        return False

def compare_faces(baseImg):
    known_images, encodings = getEncodings()
    test = fr.load_image_file(baseImg)
    try:
        test_encoding = fr.face_encodings(test)[0]
    except IndexError:
        return False
    results = fr.compare_faces(encodings, test_encoding)
    if True in results:
        i = results.index(True)
        return known_images[i].split(".")[0]
    return False

# ------------------ ATTENDANCE FUNCTION ------------------
def update_attendance(id, status):
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
    if 'file1' in request.files:
        file1 = request.files.get('file1')
        imgName = file1.filename.split(".")[0]
        response = update_face(imgName, file1)
        return jsonify({"status": response})
    return jsonify({"error": "No file provided"}), 400

@app.route('/', methods=['GET'])
def home():
    return 'AttendEase APP API is Running Successfully! ✅'

# ------------------ MAIN ENTRY POINT ------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
