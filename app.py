import argparse
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


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--quantize", default=True, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


@app.route("/", methods=["GET", "POST"])
def predictor() -> json:
    flags = vars(parse_flags())
    model_infer = ModelInferer(config=config, checkpoint_path=flags["checkpoint"], quantize=flags["quantize"])


    if request.args.get("key") and request.args.get("key") == API_KEY:
        text = request.form["data"]
        prediction = model_infer.predict(text)
        print(text)
        print(prediction)
        return jsonify(
            category=prediction
        )
    else:
        abort(401)
    # except Exception as e:
    #     logging.warning(str(e))
    #     return jsonify(
    #         category=None
    #     )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
