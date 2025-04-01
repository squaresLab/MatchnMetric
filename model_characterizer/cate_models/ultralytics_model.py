# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275

from cate_models import Model
from typing import List
import ultralytics
import numpy
import torch
import torchvision.transforms as T
from torchvision.transforms import Resize
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase
import datetime
import json

class UltralyticsModel(Model):
    """CaTE wrapper class for the Ultralytics model provided by the Ultralytics repo."""

    TABLE = "Models"

    def instantiate(self, model_name: str, model_dictionary: dict, gpu: str = None):
        """Initialize the callable model object.
        Expected format for an Ultralytics model in a model_characterizer json file:
            "model_name": "name of a model (yolo11, yolo10, etc),
            "type": "ultralytics",
            "model_dictionary": {
                "weights": "name of weights file inside model_characterizer/ultralytics/
            }
        Arguments:
            model_name: String indicating the name of the model as will be stored in the database.
            model_dictionary: For Ultralytics models, this should be a dictionary with a single key: 'weights' with a string
                indicating the name of the weights file that should be loaded, e.g. 'yolo11m.pt'
        """
        LUT = {
            "yolo11m": ultralytics.YOLO,
            "yolo11n": ultralytics.YOLO,
            "rtdetr-l": ultralytics.RTDETR,
            "rtdetr-x": ultralytics.RTDETR,
            "yolov8n": ultralytics.YOLO,
            "yolov8l": ultralytics.YOLO,
            "yolov10l": ultralytics.YOLO,
            "yolov10m": ultralytics.YOLO
        }
        self.device = gpu
        self.model = LUT[model_name[:-8]](model_dictionary["weights"])
        if self.model == None:
            raise ValueError(f"Not able to load model {model_name} with specified weights ({model_dictionary['weights']}) in supported models.")

    def _insert(self, db: ModelCharacterizerDatabase, model_name: str, model_dictionary: dict):
        """
        Formats data dictionary (if needed) and calls insert on the database to insert the model as a row.
        Expected model dictionary for a Deeplite torch model:
           {
                "architecture": str,
                "dataset": str,
                "pretrained": bool
            }
        """

        # Get the current datetime for model inserts
        date = datetime.datetime.now()
        date_formatted = date.strftime("%m/%d/%Y %H:%M:%S")
        data = {
            "model_name": model_name,
            "model_dictionary": model_dictionary,
            "date_added": f"{date_formatted}"
        }
        conn = db.get_connection()
        success, row = ModelCharacterizerDatabase.insert(self.TABLE, data, conn)
        if row is not None:
            self.id_val = row[0]
            self.model_name = row[1]

            # This ends up being a string that is actually a dictionary.
            # We 'eval' it to turn it into a dictionary.
            if not isinstance(row[3], str):
                self.model_dictionary = json.loads(row[2])
            else:
                self.model_dictionary = json.loads(row[3])
        else:
            raise RuntimeError("Failed to insert model into database.")

    def prep_model_for_inferrence(self):
        """Moves the model onto the GPU when it's time to use it."""
        self.model.to(self.device)
        self.model.float()
        self.model.eval()

    def adjust_image_tensor(self, img: torch.tensor, tensor_change: int, dimension):
        # Yolo accepts images where both x and y resolution is divisible by 32
        # This function adjusts the image as necessary to allow it to work with yolo
        # size = 32 - sizegiot
        tensor_change = 32 - tensor_change
        if dimension == "y":
            return torch.cat((img, torch.zeros_like(img)[:,:,0:tensor_change,:]), dim=2).half()
        elif dimension == "x":
            return torch.cat((img, torch.zeros_like(img)[:,:,:,0:tensor_change,]), dim=3).half()
        else:
            raise RuntimeError(f"Invalid dimension {dimension} specified, allowed values are 'x' or 'y'")

    def prep_data_for_inferrence(self, input: torch.Tensor):
        """Resizes and normalizes images in preparation for inferrence with Ultralytics models
        Also stores the original size of the image as this is used again during output formatting.

        Arguments:
            img: Tensor containing the image to be transformed. Expected format: [B, C, H, W]

        Returns:
            Resized, floatified, and normalized image(s)
        """
        self.img_size = input.size()

        y_tensor_change = input.shape[2]%32
        x_tensor_change = input.shape[3]%32
        if y_tensor_change > 0:
            input = self.adjust_image_tensor(input, y_tensor_change, "y")
        if x_tensor_change > 0:
            input = self.adjust_image_tensor(input, x_tensor_change, "x")
        input = input.float().half()
        return input/255

    def handle_nn_output(self, outputs: List[ultralytics.engine.results.Results]):
        """Iterates over results from neural network. Reformats results into output format:
        x,y,w,h,conf,cls where xy refer to the center point of the bounding box.

        Arguments:
            outputs: """
        res = []
        for idx, output in enumerate(outputs):
            xywhn = output.boxes.xywhn
            conf = output.boxes.conf.unsqueeze(1)
            classes = output.boxes.cls.unsqueeze(1)
            output_tensor = torch.cat((xywhn, conf, classes), dim=1).cpu()
            output_tensor[:, 0] *= self.img_size[-1]
            output_tensor[:, 1] *= self.img_size[-2]
            output_tensor[:, 2] *= self.img_size[-1]
            output_tensor[:, 3] *= self.img_size[-2]
            res.append(numpy.array(output_tensor))
        return res
