import json
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, abort

from inferer import ModelInferer
from utils import config

load_dotenv()

API_KEY = os.getenv("APP_APIKEY")
app = Flask(__name__)
checkpoint_path = "checkpoints/roberta_80000.bin"


@app.route("/", methods=["GET", "POST"])
def predictor() -> json:
    model_infer = ModelInferer(config=config, checkpoint_path=checkpoint_path, quantize=True)
    try:
        if request.args.get("key") and request.args.get("key") == API_KEY:
            text = request.form["data"]
            prediction = model_infer.predict(text)
            return jsonify(
                category=prediction
            )
        else:
            abort(401)
    except Exception as e:
        logging.error(str(e))
        return jsonify(
            category=None
        )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
