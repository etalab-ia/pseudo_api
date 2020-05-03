import logging
import os

import stopwatch
from flair.models import SequenceTagger
from flask import Flask
from flask import request, jsonify

from flask_app.data_ETL import prepare_output, sw

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

server = Flask(__name__)

# Env variables
MODEL_PATH = os.environ.get('PSEUDO_MODEL_PATH', './model/best-model.pt')
TAGGER = SequenceTagger.load(MODEL_PATH)


def run_demo_request():
    data = {"success": False}
    try:
        if request.form.get("text"):
            text = request.form.get("text")
            logging.info("Tagging text with model...")
            # Predict and return a CoNLL string to send to the web demo app
            conll_str = prepare_output(text=text, tagger=TAGGER, request_type="demo")
            data["conll_tagged_text"] = conll_str
            data["success"] = True

    except Exception as e:
        logger.error(e)
    finally:
        return jsonify(data)


def run_pseudonymize_request():
    data = {"success": False}
    try:
        if request.form.get("text"):
            text = request.form.get("text")
            logging.info("Tagging text with model...")
            # Predict and return a CoNLL string to send to the web demo app
            tagged_str, pseudonymized_str = prepare_output(text=text, tagger=TAGGER, request_type="api")
            data["tagged_text"] = tagged_str
            data["pseudonymized_text"] = pseudonymized_str
            data["success"] = True

    except Exception as e:
        logger.error(e)
    finally:
        print(stopwatch.format_report(sw.get_last_aggregated_report()))
        return jsonify(data)


@server.route('/pseudoniymize_demo', methods=['GET', 'POST'])
def pseudonymize_demo():
    if request.method == 'GET':
        return 'The model is up and running. Send a POST request'
    else:
        return run_demo_request()


@server.route('/pseudonimyze', methods=['GET', 'POST'])
def pseudonymize():
    if request.method == 'GET':
        return 'The model is up and running. Send a POST request'
    else:
        return run_pseudonymize_request()
