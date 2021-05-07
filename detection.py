from flask import Flask
from flask import request
from PIL import Image
from werkzeug.exceptions import BadRequestKeyError
from src.models.document_scan import InnChecker
from src.models.face_validation import DocumentFaceChecker

app = Flask(__name__)
face_model = DocumentFaceChecker()
iin_checker = InnChecker()
VALIDATION_KEY = "repaatafterme"


@app.route("/test")
def testServer():
    return "Success"


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


@app.route("/get-iin", methods=["POST"])
def get_iin():
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
        pil_image = Image.open(file)
        iin = iin_checker.get_iin(pil_image)
        return {
            "iin": iin,
            "error": None
        }
    except Exception as e:
        return {
            "iin": "",
            "error": str(e)
        }
