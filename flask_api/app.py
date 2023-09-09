from flask import Flask, jsonify, request, send_file, url_for  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
from segmentation import segmentation
from werkzeug.utils import secure_filename
import os
import base64
import cv2

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록
seg = segmentation()

@api.route('/hello')  # 데코레이터 이용, '/hello' 경로에 클래스 등록
class HelloWorld(Resource):
    def get(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"hello": "world!"}
    
@app.route('/segment', methods=['POST'])
def predict():
    if request.method == 'POST':
        imgFile = request.files["image"]
        imgFileName = secure_filename(imgFile.filename)

        savePath = "/static/"
        os.makedirs(savePath, exist_ok=True)

        imgPath = os.path.join(savePath, imgFileName)

        imgFile.save(imgPath)

        pointX = request.form['pointX']
        pointY = request.form['pointY']

        segmentedImg = seg.onePointSegment(imgPath, pointX, pointY)
        
        segmentedImgPath = os.path.join(savePath, "segmentedImg.jpeg")
        segmentedImg.save(segmentedImgPath)

        # print(url_for('static', filename = 'segmentedImg.jpeg'))

        return send_file(segmentedImgPath,  mimetype='image/jpeg', as_attachment=True)
    
if __name__ == "__main__":
    app.run(debug=True)