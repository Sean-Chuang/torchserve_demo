import io
import json
import handler

from flask import Flask, jsonify, request

app = Flask(__name__)
handler.init(None)


@app.route('/predictions/mnist', methods=['GET', 'POST', 'PUT'])
def predict():
    data = []
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        data.append({"data": img_bytes})
    elif request.method == 'PUT':
        data.append({"data": request.data})
    else:
        with open("test_data/0.png", "rb") as in_f:
            img_bytes = in_f.read()
            data.append({"data": img_bytes})

    res = handler.handle(data)
    # class_id, class_name = get_prediction(image_bytes=img_bytes)
    # return jsonify({'class_id': class_id, 'class_name': class_name})
    return jsonify({'class_id': res, 'class_name': '2345'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True, threaded=True)