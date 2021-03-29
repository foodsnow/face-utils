from flask import Flask
from flask import request
from PIL import Image
from werkzeug.exceptions import BadRequestKeyError
from src.models.face_validation import DocumentFaceChecker

app = Flask(__name__)
face_model = DocumentFaceChecker()


@app.route('/compare-faces', methods=["POST"])
def compare():
    file = None
    try:
        file = request.files['document']
    except BadRequestKeyError:
        app.logger.debug("Bad key")
    if file is None:
        return {
            "error": "Bad request"
        }
    try:
        image = Image.open(file)
        score = face_model.check(image)
        return {
            "score": float(score),
            "error": None
        }
    except Exception as e:
        return {
            "score": 0.0,
            "error": str(e)
        }
