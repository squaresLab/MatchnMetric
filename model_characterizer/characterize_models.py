# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275
# Script for detection arena model running and output caching
import argparse
from job_handler import JobHandler

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--perturbation", type=str, default=None,
                            help="""Path to json config specifying which perturbation to use. To use a perturbation 
                                    that does not modify the image, specify 'no-op'.""")
    arg_parser.add_argument("--video-path", type=str, default=None,
                            help="""Path to a video file to use for input. Current expected resolution is 1920x1080.
                                    If not set, expects --pre-cached-images to be set to use as images for input. 
                                    Mutually exclusive with --pre-cached images.""")
    arg_parser.add_argument("--annotations", type=str, default=None,
                            help="Path to the annotations file related to the video dataset specified in '--video-path'")
    arg_parser.add_argument("--image-output-path", type=str, default=None,
                            help="""Output path for image caching. Mutually exclusive with --pre-cached-images.""")
    arg_parser.add_argument("--model", type=str, default=None,
                            help="""Argument used to run a single model for test. String will be passed directly to
                             deeplite to try to find the model. Mutually exclusive with '--json-job' argument.""")
    arg_parser.add_argument("--model_dataset", type=str, default=None,
                            help="If supplied, will attempt to find a model pretrained on this dataset.")
    arg_parser.add_argument("--detections-output-path", type=str, default=None,
                            help="""Root path for detection outputs. If not set, will raise an error.""")
    arg_parser.add_argument("--json-job", type=str, default=None, nargs='+',
                            help="""Paths to json files specifying the models, datasets, perturbations,
                                    and other arguments to use.  Later json files will overwrite earlier ones with the same fields.
                                    Currently the only way to leverage the database
                                    for multiple runs.""")
    args = arg_parser.parse_args()
    
    if args.json_job is not None:
        print(f"Running json job with path: {args.json_job}")
        model_characterizer = JobHandler()
        model_characterizer.handle_json_jobs(json_paths=args.json_job)
    else:
        print("Running CLI job!")
        model_characterizer = JobHandler()
        model_characterizer.handle_cli_job(
            video_path=args.video_path,
            annotations_path=args.annotations,
            perturbation_path=args.perturbation,
            model=args.model,
            model_dataset=args.model_dataset,
            image_output_path=args.image_output_path,
            detections_output_path=args.detections_output_path
        )
