# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275
import cv2
import numpy as np
import argparse
from typing import Union, List
import os
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase
import json

def grab_model(model_id_val, conn):
    conn.execute(f"select * from Models where id={model_id_val}")
    return conn.fetchone()

def grab_dataset(dataset_id_val, conn):
    conn.execute(f"select * from Datasets where id={dataset_id_val}")
    return conn.fetchone()

def grab_perturbation(perturbation_id_val, conn):
    conn.execute(f"select * from Perturbations where id={perturbation_id_val}")
    return conn.fetchone()

def grab_evaluation(evaluation_id_val, conn):
    conn.execute(f"select * from Evaluations where id={evaluation_id_val}")
    return conn.fetchone()

db_path = None

def get_database(json_paths):
    global db_path
    config = None
    for json_path in json_paths:
        config = {}
        with open(json_path, "rb") as f:
            config.update(json.load(f))

    db_path = config["metadata"]["db_root"]
    return ModelCharacterizerDatabase(db_path=db_path,
                                        database=config["metadata"]["database"],
                                        user=config["metadata"]["user"],
                                        password=config["metadata"]["password"],
                                        host=config["metadata"]["host"],
                                        port=config["metadata"]["port"])

def write_anno_to_frame(video_path, detections, output_dir):
    # Detection format: x, y, w, h, conf, cls
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dets = detections[i]
        for det in dets:
            print(dets)
            print(dets.shape)
            if dets.shape == (1, 0):
                break
            x, y, w, h, conf, thing = det
            col = (0, 0, 255)
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x + w/2), int(y + h/2)), col, 2)
        print(f"Writing to {os.path.join(output_dir, str(i))}.png")
        cv2.imwrite(os.path.join(output_dir, f"{str(i)}.png"), frame)
        i = i + 1

def display_evaluation(json_paths, eval_ids: List[int], output_dir: str, video_root_dir: str = None):
    db = get_database(json_paths)
    conn = db.get_connection()
    curr = conn.cursor()
    for eval_id in eval_ids:
        
        evaluation = grab_evaluation(eval_id, curr)
        # evaluation = (id, model_id, dataset_id, date_run, finished)
        evaluation_id = evaluation[0]
        eval_output_path = os.path.join(output_dir, str(evaluation_id))
        dataset = grab_dataset(evaluation[2], curr)
        # dataset = (id, dataset_name, video_path, date_added, perturbation_id, cached, x_res, y_res)
        # TODO Use perturbation to perturb original images
        perturbation = grab_perturbation(dataset[4], curr)
        # perturbation = (id, perturb_class, perturb_parameters)
        detection_files_path = os.path.join(db_path, "evaluations", str(eval_id), "detections")
        detection_files = os.listdir(detection_files_path)
        detection_files.sort()

        det_arrays = []
        for file in detection_files:
            detections = np.loadtxt(os.path.join(detection_files_path, file))
            if len(detections.shape) == 1:
                detections = np.expand_dims(detections, axis=0)
            det_arrays.append(detections)
            print(detections)
        # Grab the video
        video_path = dataset[2]
        # If user supplied an alternate directory to use as root for video files, use that one.
        if video_root_dir is not None:
            video_filename = os.path.basename(video_path)
            video_path = os.path.join(video_root_dir, video_filename)
        write_anno_to_frame(video_path, det_arrays, eval_output_path)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--json_paths", type=str, nargs='+',
                            help="Paths to the JSON file describing the database. Can be the same ones used to run the ModelCharacterizer job. Later json files will overwrite earlier ones with the same fields.")
    arg_parser.add_argument("--eval_ids", type=int, nargs="+", help="ID of the evaluation you wish to display")
    arg_parser.add_argument("--output_dir", type=str, default="./", help="Directory to use as root for output")
    arg_parser.add_argument("--video_root_dir", type=str, default=None, help="Directory to use as root for video files. If supplied, try looking for video files at {video_root_dir}/{file_basename}")

    # arg_parser.add_argument("--")
    args = arg_parser.parse_args()
    if args.json_paths is None:
        raise RuntimeError("Please provide at least one path for the JSON configuration file.")
    display_evaluation(args.json_paths, args.eval_ids, args.output_dir, args.video_root_dir)

