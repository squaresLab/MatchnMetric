# CATE Detection Evaluation code
 
# Copyright 2025 Carnegie Mellon University.
 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
 
# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
 
# DM25-0275

import pandas as pd # type: ignore
import numpy as np
import os
from tqdm import tqdm # type: ignore
import cv2 # type: ignore
from typing import Collection, Hashable, Tuple, Optional, Union, List
import numpy.typing as npt
import torch
from collections import namedtuple
from dataclasses import dataclass

from deeplite_torch_zoo.src.object_detection.datasets.utils import xywh2xyxy # type: ignore
from deeplite_torch_zoo.src.object_detection.eval.utils import box_iou # type: ignore

from scipy.optimize import linear_sum_assignment # type: ignore



class BoundingBoxesDict(dict):
    """ The BoundingBoxesDict base class

    This object stores a dictionary of pandas DataFrames that contain
    bounding boxes for a number of images. This is meant to be extended
    into different types of bounding boxes for different uses.

    Use this as a dict, with all dict operations

    Attributes:
        COLUMN_NAMES Tuple[str,...]: Names expected for columns of DataFrames, 
                                     when loading from numpy arrays
        DEFAULT_COLOR Tuple[int,int,int]: color values used for drawing
    
    """
    COLUMN_NAMES: Tuple[str,...] = ('x','y','w','h')
    DEFAULT_COLOR: Tuple[int,int,int] = (0,0,0)

    def __init__(self, 
                 boxes_iter: Collection[Union[npt.NDArray,pd.DataFrame]]=[], 
                 keys: Optional[Collection[Hashable]]=None) -> None:
        if (keys is not None) and (len(boxes_iter)!=len(keys)):
            raise RuntimeError(f"Different frame counts between boxes: {len(boxes_iter)}, and names: {len(keys)}")
        if keys is None:
            keys = range(len(boxes_iter))
        boxes_list = []
        for boxes_data,frame_id in zip(boxes_iter,keys):
            if isinstance(boxes_data, pd.DataFrame):
                df = boxes_data
            elif isinstance(boxes_data, np.ndarray):
                df = pd.DataFrame(data=boxes_data,columns=self.COLUMN_NAMES)      
            boxes_list.append(df)
        super(BoundingBoxesDict,self).__init__(zip(keys,boxes_list))

    def draw(self, frame_key: Hashable, frame: npt.NDArray, 
             color: Optional[Tuple[int,int,int]]=None)-> npt.NDArray:
        """Returns image with all bounding boxes drawn on.
        
        This method uses self.get_bb_label, which is expected to be implemented by each class that
        extends BoundingBoxesDict. 
        
        Arguments:
            frame_key: key for finding frame in BoundingBoxesDict dictionary
            frame:     numpy array image to be updated
            color:     rgb color triplet to use for box and text. default can be overridden in child classes
                       
        Returns:
            numpy array image with bounding boxes drawn in"""
        if color is None:
            color = self.DEFAULT_COLOR
        pd = self[frame_key]
        for index in pd.index:
            x,y,w,h = pd['x'][index],pd['y'][index],pd['w'][index],pd['h'][index]
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x + w/2), int(y + h/2)), color, 2)
            cv2.putText(frame, 
                            self.get_bb_label(frame_key,index),# from {np.argmax(all_iou[:,label_ind].numpy(force=True))}", 
                            (int(x-w/2), int(y-h/2 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def get_bb_label(self, key:Hashable, index:int)->str:
        return ""
    
    @staticmethod
    def get_xyxytorch(bbox_df:pd.DataFrame)->torch.Tensor:
        x = np.expand_dims(bbox_df['x'].to_numpy(),axis=1)
        y = np.expand_dims(bbox_df['y'].to_numpy(),axis=1)
        w = np.expand_dims(bbox_df['w'].to_numpy(),axis=1)
        h = np.expand_dims(bbox_df['h'].to_numpy(),axis=1)
        xyxy = xywh2xyxy(torch.tensor(np.concatenate((
            x-w/2,y-h/2,w,h),axis=1),device='cpu'))
        return xyxy

class DetectionsDict(BoundingBoxesDict):
    """ The DetectionsDict class for multiple video detections

    This object stores a dictionary of pandas DataFrames that contain
    bounding boxes for a number of images. These are built on 
    BoundingBoxesDict, but additionally have additional confidence and 
    predicted class fields

    Use this as a dict, with all dict operations

    Attributes:
        COLUMN_NAMES Tuple[str,...]: Names expected for columns of DataFrames, 
                                     when loading from numpy arrays
        DEFAULT_COLOR Tuple[float,float,float]: color values used for drawing
    
    """
    COLUMN_NAMES: Tuple[str,...] = ('x','y','w','h','confidence','predicted_class')
    DEFAULT_COLOR: Tuple[int,int,int] = (0,0,255)
    def get_bb_label(self, key: Hashable, index: int)->str:
        pred_conf = self[key]['confidence'][index]
        return f"conf{pred_conf}"
    
    @staticmethod
    def load_directory(directory_path: Union[str,os.PathLike], 
                          frame_ids: Optional[Collection[int]]=None,
                          filter_class: Optional[npt.ArrayLike]=None):
        """Constructs a new DetectionsDict object from a directory of detection files.
        
        By default, loads all files found, assuming names go from 0000001.txt up to
        f'{frame_count:0>7}.txt' inclusive. While frame ids are 1 indexed in these
        filenames, the resulting object is zero indexed. If given a iteratable of
        frame_ids, it will only load that subset. Again, it will use 0 indexed
        frame_ids, even though detection files are 1 indexed. Can optionally filter
        detections, only including classes specified in filter_class.
        
        Arguments:
            directory_path: required path like directory containing detection files
            frame_ids:      optional 0-indexed list of frame_ids to load
            filter_class:   optional list of classes to include from detection files
                       
        Returns:
            numpy array image with bounding boxes drawn in"""

        # when not specified, load all frame_ids in directory
        if frame_ids is None:
            count_frames = len(os.listdir(directory_path))
            frame_ids = range(count_frames)
        data_list = []
        for frame_id in frame_ids:
            # directory is 1 indexed
            frame_plus_one = frame_id 
            det_file_path = os.path.join(directory_path, f'{frame_plus_one:0>7}.txt')
            with open(det_file_path, 'r') as fp:
                data = fp.read()
            predsxywh = np.fromstring(data.replace('[','').replace(']',''),sep=' ').reshape((-1,6))
            if filter_class is not None:
                filter_mask = np.isin(predsxywh[:,5], filter_class)
                predsxywh = predsxywh[filter_mask]
            # DEBUG
            # predsxyxy = predsxywh
            # predsxywh[2:4] = (predsxyxy[0:2]+predsxyxy[2:4])/2
            # predsxywh[0:2] = predsxywh[0:2]-(predsxywh[2:4])/2
            data_list.append(predsxywh)
        return DetectionsDict(boxes_iter=data_list,keys=frame_ids)


class LabelsDict(BoundingBoxesDict):
    """ The LabelsDict class for multiple video annotations

    This object stores a dictionary of pandas DataFrames that contain
    bounding boxes for a number of images. These are built on 
    BoundingBoxesDict, but additionally have personpath label fields,  
    defined in COLUMN_NAMES

    Use this as a dict, with all dict operations

    Attributes:
        COLUMN_NAMES Tuple[str,...]: Names expected for columns of DataFrames, 
                                     when loading from numpy arrays
        DEFAULT_COLOR Tuple[float,float,float]: color values used for drawing
    
    """
    COLUMN_NAMES_LOAD = ('frame', 'id', 'x', 'y', 'w', 'h', 'confidence', 'class', 'visibility', 'misc')
    COLUMN_NAMES:Tuple[str,...] = ('x', 'y', 'w', 'h','track_id', 'label_confidence', 'label_class', 'label_visibility', 'label_misc')
    DEFAULT_COLOR:Tuple[int,int,int] = (255,0,0)

    def get_bb_label(self, key: Hashable, index: int)->str:
        id = self[key]['track_id'][index]
        vis = self[key]['track_id'][index]
        return f"id:{id} vis:{vis}"
    
    @staticmethod
    def load_video_labels(annotation_file: Union[str, os.PathLike], 
                          frame_ids: Optional[Collection[int]]=None,
                          filter_class: Optional[npt.ArrayLike]=None):
        """Constructs a new LabelsDict object from an annotation file.
        
        Loads a single annotation file, downloaded from 
        https://github.com/JonathonLuiten/TrackEval. Each file contains all 
        labels for a single video, indexed by a 1-indexed frame id, but this
        object splits them up by frame. By default, loads all boxes found. 
        While frame ids are 1 indexed in the annotation file, the resulting object 
        is zero indexed. If given a iteratable of frame_ids, it will only load 
        that subset. Again, it will use 0 indexed frame_ids, even though detection 
        files are 1 indexed. Can optionally filter labels, only including classes 
        specified in filter_class.
        
        Arguments:
            annotation_file: path-like file from TrackEval to load
            frame_ids:       optional 0-indexed list of frame_ids to load
            filter_class:    optional list of classes to include from detection files
                       
        Returns:
            numpy array image with bounding boxes drawn in"""
        annotations = pd.read_csv(annotation_file, header=None)
        annotations.columns = LabelsDict.COLUMN_NAMES_LOAD
        max_frame_id = np.max(annotations['frame'])-1
        if frame_ids is None:
            frame_ids = range(max_frame_id)
        data_list = []
        frame_ids_labeled = []
        for frame_id in frame_ids:
            # directory is 1 indexed
            frame_plus_one = frame_id + 1
            frame_annotations = annotations[annotations['frame'] == (frame_plus_one)]

            frame_annotations = frame_annotations[frame_annotations['id']<10000]
            if filter_class is not None:
                filter_mask = np.isin(frame_annotations['class'], filter_class)
                frame_annotations = frame_annotations[filter_mask]
            x,y,w,h = frame_annotations['x'],frame_annotations['y'],frame_annotations['w'],frame_annotations['h']
            labelnp = np.array([x+w/2,y+h/2,w,h,frame_annotations['id'],frame_annotations['confidence'],
                                frame_annotations['class'],frame_annotations['visibility'], frame_annotations['misc']]).transpose()
            if labelnp.shape[0] > 0:
                data_list.append(labelnp)
                frame_ids_labeled.append(frame_id)
        return LabelsDict(boxes_iter=data_list,keys=frame_ids_labeled)

@dataclass
class detection_match_pair:
    detections : pd.DataFrame
    labels : pd.DataFrame

class MatchedDetectionsDict(dict):
    """ The MatchedDetectionsDict class for multiple video annotations

    This object stores a dictionary of tuples of pandas DataFrames that 
    contain detection and label bounding boxes for a number of images. 
    These are similar to BoundingBoxesDict, and are constructed from and 
    can be converted to BoundingBoxesDict derived classes.

    Use this as a dict, with all dict operations

    Attributes:
        DEFAULT_COLOR_DET Tuple[float,float,float]: color values used for drawing detection
        DEFAULT_COLOR_LABEL Tuple[float,float,float]: color values used for drawing label
    
    """
    DEFAULT_COLOR_DET:Tuple[int,int,int] = (0,255,255)
    DEFAULT_COLOR_LABEL:Tuple[int,int,int] = (255,255,0)
    # TODO, optional keys argument that allows specifying different names
    # may be useful if matching detections from subsequent frames, instead of labels
    def __init__(self, detections: DetectionsDict, labels: LabelsDict) -> None:
        if detections.keys() != labels.keys():
            raise RuntimeError(
                f"Different keys counts between detections: {len(detections)}, and labels: {len(labels)}")
        for key in detections.keys():
            if not len(detections[key].index) == len(labels[key].index):
                raise RuntimeError(
                    f"Different indices between detections: {len(detections[key].index)}, and labels: {len(labels[key].index)} at key {key}")
        
        boxes_list = [detection_match_pair(detections[key],labels[key]) for key in detections.keys()]
        super(MatchedDetectionsDict,self).__init__(zip(detections.keys(),boxes_list))

    def get_detections(self) -> DetectionsDict:
        """Converts to DetectionsDict object."""
        boxes_list = [self[key].detections for key in self.keys()]
        return DetectionsDict(boxes_iter=boxes_list,keys=self.keys())
    
    def get_labels(self) -> LabelsDict:
        """Converts to LabelsDict object."""
        boxes_list = [self[key].labels for key in self.keys()]
        return LabelsDict(boxes_iter=boxes_list,keys=self.keys())
    
    def draw(self, frame_key: Hashable, frame: npt.NDArray, 
             color: Optional[Tuple[int,int,int]]=None,
             draw_label: Union[bool, Tuple[int,int,int]]=True)-> npt.NDArray:
        """Returns image with all bounding boxes drawn on.
        
        This method uses self.get_bb_label, which is expected to be implemented by each class that
        extends BoundingBoxesDict. 
        
        Arguments:
            frame_key:  key for finding frame in BoundingBoxesDict dictionary
            frame:      numpy array image to be updated
            color:      rgb color triplet to use for box and text. default can be overridden in child classes
            draw_label: Union[bool, Tuple[int,int,int]] for specifying whether to draw the label as well
                        default, true, will use default label color. If color is specified, it will be used 
                        as the color for rectangle and text for matched labels
                   
        Returns:
            numpy array image with bounding boxes drawn in"""
        if color is None:
            color = self.DEFAULT_COLOR_DET
        label_color = self.DEFAULT_COLOR_LABEL
        to_draw_label = False
        if isinstance(draw_label, tuple):
            label_color = draw_label
            to_draw_label = True
        elif isinstance(draw_label,bool):
            to_draw_label = draw_label

        if to_draw_label:
            labels = self.get_labels()
            frame = labels.draw(frame=frame, frame_key=frame_key, color=label_color)
        detections = self.get_detections()
        frame = detections.draw(frame=frame, frame_key=frame_key, color=color)
        return frame
    
    @staticmethod
    def compute_similarity(box_a: BoundingBoxesDict, box_b: BoundingBoxesDict)-> npt.NDArray:
        labelxyxy = BoundingBoxesDict.get_xyxytorch(box_a)
        detctxyxy = BoundingBoxesDict.get_xyxytorch(box_b)                          
        all_iou = box_iou(detctxyxy,labelxyxy)

        return all_iou.numpy(force=True)
    
    @staticmethod
    def match_detections(detections: DetectionsDict, labels: LabelsDict, 
                         overlap_thresh: float = 0.5
                         )->'detection_match':
        """Performs bounding box matching between all detections and labels from a video.
        
        This consumes a DetectionsDict and LabelsDict and finds all overlapping and non-overlapping
        bounding boxes across them. Matching is Hungarian, maximizing the sum of confidence, 
        considering box pairs with iou greater than overlap_thresh for matching. Matched boxes are
        returned as a MatchedDetectionsDict object.
        It also generates two different DetectionsDict, one for frames that were unlabeled and one false
        positives in frames with labels. It finally generates a LabelsDict for all unmatched labels, 
        false negative labels.
        
        Arguments:
            detections (DetectionsDict): dictionary of detections
            labels (LabelsDict):   dictionary of labels, with keys that are a subset of detections keys
            overlap_thresh (float): float setting required overlap to consider match between detection and label
                   
        Returns:
            Tuple
            matches (MatchedDetectionsDict),
            false positive detections (DetectionsDict),
            unlabeled detections (DetectionsDict)
            false negative labels (LabelsDict)"""
        # TODO consider overlap as or in match measure
        unlabeled_list = []
        false_positives_list = []
        false_negatives_list = []
        matches_list = []


        for key in detections.keys():

            if key in labels:
                labeldf = labels[key]
                detectionsdf = detections[key]
                detections_conf = np.expand_dims(detectionsdf['confidence'].to_numpy(),axis=1)
                    
                all_iou = MatchedDetectionsDict.compute_similarity(labels[key], detections[key])
                # compare boxes for most confident
                # need to build a cost matrix between labels and detections
                # for all pairs with required overlap, set the cost to the -confidence
                # for all pairs without required overlap, set the cost ot positive cost
                all_conf = -((all_iou > overlap_thresh) * detections_conf) + (all_iou <= overlap_thresh) * 1
                 
                assignments_conf = linear_sum_assignment(all_conf)
                false_positives_list.append((detectionsdf.drop(assignments_conf[0]),key))
                false_negatives_list.append((labeldf.drop(assignments_conf[1]),key))
                matches_list.append((detectionsdf.filter(assignments_conf[0],axis=0),
                                     labeldf.filter(assignments_conf[1],axis=0),key))
            else:
                unlabeled_list.append((detections[key],key))
        matches_detections_list_transpose: Tuple[Tuple[pd.DataFrame],
                                                 Tuple[pd.DataFrame],
                                                 Tuple[Hashable]] = tuple(zip(*matches_list)) # type: ignore[assignment]
        matches_detection: DetectionsDict = DetectionsDict(matches_detections_list_transpose[0],
                                           matches_detections_list_transpose[2])
        matches_labels: LabelsDict = LabelsDict(matches_detections_list_transpose[1],
                                           matches_detections_list_transpose[2])
        matches = MatchedDetectionsDict(matches_detection,matches_labels)

        if len(false_positives_list) == 0:
            false_positives = None
        else:
            false_positives_list_inverse: Tuple[Tuple[pd.DataFrame],
                                                Tuple[Hashable]] = tuple(zip(*false_positives_list))# type: ignore[assignment]
            false_positives = DetectionsDict(false_positives_list_inverse[0],
                                            false_positives_list_inverse[1])
        if len(unlabeled_list) == 0:
            unlabeled=None
        else:
            unlabeled_list_inverse: Tuple[Tuple[pd.DataFrame],
                                                Tuple[Hashable]] = tuple(zip(*unlabeled_list)) # type: ignore[assignment]
            unlabeled = DetectionsDict(unlabeled_list_inverse[0],
                                             unlabeled_list_inverse[1])
        if len(false_negatives_list) == 0:
            false_negatives = None
        else:
            false_negatives_list_inverse: Tuple[Tuple[pd.DataFrame],
                                                Tuple[Hashable]] = tuple(zip(*false_negatives_list)) # type: ignore[assignment]
            false_negatives = LabelsDict(false_negatives_list_inverse[0],
                                            false_negatives_list_inverse[1])
        return detection_match(matches=matches,
                               unmatched_dets=false_positives,
                               unlabeled_dets=unlabeled,
                               unmatched_labs=false_negatives)
    

@dataclass
class detection_match:
    matches : MatchedDetectionsDict
    unmatched_labs : LabelsDict
    unmatched_dets : DetectionsDict
    unlabeled_dets : DetectionsDict