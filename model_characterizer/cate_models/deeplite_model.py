# CATE Detection Evaluation code

# Copyright 2025 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# DM25-0275

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.src.object_detection.eval.utils import non_max_suppression, xyxy2xywh
from database.postgres_model_characterizer_db import ModelCharacterizerDatabase

import torchvision.transforms as T
import datetime
from cate_models import Model
import torch
import json

class DeepliteModel(Model):

    TABLE = "Models"

    def instantiate(self, model_name: str, model_dictionary: dict, gpu: str = None):
        """Initialize the model object itself.
        Expected Usage:
        self.model MUST be a callable that can take in image tensors and output an inference.
        Expected model dictionary for a Deeplite torch model:
           {
                "architecture": str,
                "dataset": str,
                "pretrained": bool
            }
        """
        self.model = get_model(
                model_name=model_name[:-4],
                dataset_name=model_dictionary["dataset"],
                pretrained=model_dictionary["pretrained"]
            )
        self.prep_model_for_inferrence(gpu)

    def prep_model_for_inferrence(self, device: str = None):
        """Perform any model-side changes required to run repeated inferences on the model.
        In this case, move the model to a gpu and set to half precision.
        """
        if device is None:
            device = "cpu"
        self.device = device
        self.half = True
        # self.model if self.half else self.model.float()
        self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()


    def adjust_image_tensor(self, img: torch.tensor, tensor_change: int, dimension):
        # Yolo accepts images where both x and y resolution is divisible by 32
        # This function adjusts the image as necessary to allow it to work with yolo
        tensor_change = 32 - tensor_change
        if dimension == "y":
            return torch.cat((img, torch.zeros_like(img)[:,:,0:tensor_change,:]), dim=2).half()
        elif dimension == "x":
            return torch.cat((img, torch.zeros_like(img)[:,:,:,0:tensor_change,]), dim=3).half()
        else:
            raise RuntimeError(f"Invalid dimension {dimension} specified, allowed values are 'x' or 'y'")

    def prep_data_for_inferrence(self, input):
        """Perform any pre-inferrence transformation on input tensors to make it compatible with this model.

        For deeplite models, the image tensor is first modified to make sure it is compatible in regards
        to X and Y resolution and then moved to half precision and onto the CUDA device associated with this model.

        Arguments:
            input:
                1x3xYxH tensor representing an RGB image read in from a video.
        """
        self.img_size = input.size()
        t = T.Compose([ T.Resize((640, 640))])
        res_img = t(input.float())
        res_img /= 255
        return res_img


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

            if not isinstance(row[3], str):
                self.model_dictionary = json.loads(row[2])
            else:
                self.model_dictionary = json.loads(row[3])
        else:
            raise RuntimeError("Failed to insert model into database.")

    def handle_nn_output(self, nn_out):
        """Expected to take in the direct output of the neural network. In the case of the deeplite models,
           this is a tuple: (predictions, train_out). We only care about the predictions."""
        preds = nn_out[0]
        ret_res = []
        for image_preds in preds:
            # Each iteration is all the boxes on a single image
            nms_preds = non_max_suppression(image_preds.unsqueeze(0),
                                            conf_thres=0.001,
                                            iou_thres=0.5,
                                            labels=[],
                                            multi_label=True,
                                            agnostic=False)
            predsxywh = xyxy2xywh(nms_preds[0].numpy(force=True))
            
            x_scale = self.img_size[-1]/640
            y_scale = self.img_size[-2]/640
            predsxywh[:, 0] *= x_scale
            predsxywh[:, 1] *= y_scale
            predsxywh[:, 2] *= x_scale
            predsxywh[:, 3] *= y_scale
            ret_res.append(predsxywh)
        return ret_res

    def format_output(self, nn_output):
        """Taking in the direct output of the neural network, convert the model output into our standardized format (2d tensory )."""
        pass