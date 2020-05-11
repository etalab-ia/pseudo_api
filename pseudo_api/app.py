import json
import logging
import os

import stopwatch
from flair.models import SequenceTagger
from flask import Flask
from flask import request, jsonify
from sqlitedict import SqliteDict

from data_ETL import prepare_output, sw, update_stats

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

server = Flask(__name__)

# Env variables
PSEUDO_MODEL_PATH = os.environ.get('PSEUDO_MODEL_PATH', './model/best-model.pt')
TAGGER = SequenceTagger.load(PSEUDO_MODEL_PATH)


def run_stats_request():
    stats_dict = SqliteDict('./api_stats.sqlite', autocommit=True)
    return jsonify(dict(stats_dict))


def run_pseudonymize_request():
    data = {"success": False}
    stats_dict = SqliteDict('./api_stats.sqlite', autocommit=True)
    output_types = ["pseudonymized", "tagged", "conll"]
    try:
        if not request.form.get("output_type"):
            logging.info("No tags were indicated. I will give you the text pseudonymized.")
            output_type = "pseudonymized"
        else:
            output_type = request.form.get("output_type")
            if output_type not in output_types:
                logging.warning("Your output type is not supported. I will give you the text pseudonymized.")
                output_type = "pseudonymized"

        if request.form.get("text"):
            text = request.form.get("text")
            logging.info("Tagging text with model...")
            # Predict and return a CoNLL string to send to the web demo app
            output, analysis_ner_stats = prepare_output(text=text, tagger=TAGGER, output_type=output_type)
            data[output_type] = output
            data["success"] = True
            # stats_dict[:]
    except Exception as e:
        logger.error(e)
    finally:
        logger.info(stopwatch.format_report(sw.get_last_aggregated_report()))
        if data["success"]:
            update_stats(analysis_stats=stats_dict, analysis_ner_stats=analysis_ner_stats,
                         time_info=sw.get_last_aggregated_report(), output_type=output_type)
        logger.info(json.dumps(dict(stats_dict), indent=4))
        stats_dict.close()
        return jsonify(data)


@server.route('/', methods=['GET', 'POST'])
def pseudonymize():
    if request.method == 'GET':
        return 'The model is up and running. Send a POST request'
    else:
        return run_pseudonymize_request()


@server.route('/api_stats/', methods=['GET'])
def stats():
    if request.method == 'POST':
        return 'The model is up and running. Send a GET request'
    else:
        return run_stats_request()
