'''
Pseudonymize a doc file. It takes as input a .doc file, converts it to txt, pseudonymizes it and outputs a
pseudonymized txt file.

Usage:
    doc2pseudo.py <input_file_path> <model_folder> [options]

Arguments:
    <input_file_path>       A required path parameter
    <model_folder>          A folder with a model inside
'''
import logging
import os
from glob import glob
from pathlib import Path

from argopt import argopt
from flair.models import SequenceTagger
from joblib import Parallel, delayed
from tqdm import tqdm

from data_ETL import prepare_output


def doc2txt(doc_path: Path):
    if doc_path.suffix == ".doc":
        try:
            import textract
        except ImportError:
            raise Exception("Textract is not installed. Cannot convert .doc file")
        text = textract.process(doc_path.as_posix()).decode("utf-8").replace("|", "")
        return text
    elif doc_path.suffix == ".txt":
        with open(doc_path.as_posix()) as filo:
            return filo.read()
    else:
        raise Exception("File type not handled: either .doc or .txt")


def save_text_file(text: str, output_file:Path):
    with open(output_file.as_posix(), "w") as out:
        out.write(text)


def run(doc_path: Path):
    text = doc2txt(doc_path=doc_path)
    output_text = Path(doc_path.stem + ".txt")
    output, analysis_ner_stats = prepare_output(text=text, tagger=TAGGER, output_type="pseudonymized")
    save_text_file(output, output_file=Path(output_text))
    print(output)


def main(input_file_path: Path, model_folder: Path):
    global TAGGER

    doc_paths = []
    TAGGER = SequenceTagger.load(model_folder)
    job_output = []
    tqdm.write(f"Converting file {input_file_path}")
    job_output.append(run(input_file_path))

    return doc_paths


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    input_file_path = Path(parser.input_file_path)
    model_folder = parser.model_folder
    main(input_file_path=input_file_path, model_folder=model_folder)
