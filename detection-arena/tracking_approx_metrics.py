# CATE Detection Evaluation code
 
# Copyright 2025 Carnegie Mellon University.
 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
 
# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
 
# DM25-0275

import bounding_box_structs as bbs
import tradeoff_curve as tc
import numpy as np
import math
import numpy.typing as npt
import pandas as pd
from typing import Sequence, List, Dict, Union, Tuple
from dataclasses import dataclass
from collections import namedtuple
from tqdm import tqdm
import sys
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(dir_path,'..','tracker-modules/sort'))
from sort import Sort

sys.path.append(os.path.join(dir_path,'..','tracker-modules/ByteTrack'))
from yolox.tracker.byte_tracker import BYTETracker, STrack

sys.path.append(os.path.join(dir_path,'..','tracker-modules/OC_SORT/trackers'))
from ocsort_tracker.ocsort import OCSort

sys.path.append(os.path.join(dir_path,'..','tracker-modules/pkf/trackers'))
from pkf_tracker.pkf import PKFTracker

sys.path.append(os.path.join(dir_path,'..','TrackEval/'))
import trackeval.metrics.hota as hota
import trackeval.metrics.clear as clear
import trackeval.metrics.identity as identity

def __fp_selector(FP:npt.NDArray, divisor:int, max_rate:float=1, force_good:bool=False, scaling:float=1)->Tuple[npt.NDArray,npt.NDArray]:
    """Convenience function for computing either fp or (max_fp-fp)"""
    fp_raw = FP * scaling / (divisor)
    fp_rate_allowed_mask = fp_raw < max_rate
    if force_good:
        fp = max_rate - np.minimum(fp_raw, max_rate)
    else:
        fp = np.maximum(FP.astype(np.float64) / (divisor), 1e-9)
        # update observation for allowing max false positives
        fp[np.nonzero(np.logical_not(fp_rate_allowed_mask))] = max_rate

    return (fp, fp_rate_allowed_mask)



def __accumulate_matches(matches:Sequence[bbs.detection_match], tqdm_string:str) -> Tuple[pd.DataFrame,int,int]:
    """Accumulates over a sequence of detection match objects, getting FP and TP rates.
    
    Arguments:
        matches:     set of matched and unmatched detections to compute roc-a on
        tqdm_string: string used for tqdm printout
                    
    Returns:
        dataframe  with confidences and FP/TP counts
        int        count of labels processed
        int        count of frames processed
    """
    
    conf_list = []
    count_labels = 0
    count_frames = 0
    for ndm in tqdm(matches, desc=tqdm_string,leave=False): # loop over videos
        count_frames += len(ndm.matches.keys())

        for key in ndm.matches.keys(): # loop over frames
            # columns are: confidence, is FP?, is TP?
            TP_conf = ndm.matches[key].detections[['confidence',]].copy()
            TP_conf['isFP?'] = 0
            TP_conf['isTP?'] = 1

            FP_conf = ndm.unmatched_dets[key][['confidence',]].copy()
            FP_conf['isFP?'] = 1
            FP_conf['isTP?'] = 0
            conf_list += [TP_conf, FP_conf]

            count_labels += TP_conf.shape[0]
            count_labels += ndm.unmatched_labs[key].shape[0]


    conf_sort = pd.concat(conf_list,axis=0)
    conf_sort = conf_sort.sort_values('confidence',axis=0, ascending=False)

    # count occurences
    conf_sort['isFP?_sum'] = conf_sort['isFP?'].cumsum()
    conf_sort['isTP?_sum'] = conf_sort['isTP?'].cumsum()

    conf_sort = conf_sort.drop_duplicates(subset=['confidence'], keep='last')
    conf_sort = conf_sort.iloc[::-1]
    

    return (conf_sort, count_labels, count_frames)

def map(matches:Sequence[bbs.detection_match], name_prefix='', name_suffix='',
        iou_threshes = np.arange(0.5, 0.955, 0.05),conf_samples=np.linspace(0, 1, 1000))-> Dict[str,tc.ScoreCurves]:
    
    labels_dict_list = []
    det_dict_list = []
    video_match_detections = []
    video_idx_thresh_tuples = []
    det_match_over_thresh = []

    for idx, ndm in enumerate(tqdm(matches, desc=f'map video loop',leave=False)): # loop over videos
        key_list = []
        label_list = []
        det_list = []
        for key in tqdm(ndm.matches.keys(),leave=False, desc=f'map frame loop'): # loop over frames

            labels_df = pd.concat((ndm.matches[key].labels, ndm.unmatched_labs[key]), axis=0)
            detections_df = pd.concat((ndm.matches[key].detections, ndm.unmatched_dets[key]), axis=0)
            key_list.append(key)
            label_list.append(labels_df)
            det_list.append(detections_df)
        labels_dict = bbs.LabelsDict(label_list, keys=key_list)
        detections_dict = bbs.DetectionsDict(det_list, keys=key_list)
        labels_dict_list.append(labels_dict)
        det_dict_list.append(detections_dict)

        for i in iou_threshes:
            det_match = bbs.MatchedDetectionsDict.match_detections(detections_dict, labels_dict, i)
            video_idx_thresh_tuples.append((idx, i))
            video_match_detections.append(det_match)

    # At this line we have this list:
    # video_match_detections:
    #    [length len(videos), each entry is a list of per-video match-detections per threshold (0.5:0.95)]
    index = pd.MultiIndex.from_tuples(video_idx_thresh_tuples, names=['video_idx', 'thresh'])
    matches_pd = pd.Series(video_match_detections, index=index)
    regrouped_matches = matches_pd.groupby(level='thresh')

    score_curves_dict = {}
    for iou_thresh, matches_df in tqdm(regrouped_matches,desc='map matching'):
        name = f'map_{iou_thresh:.2f}'
        tqdm_string = f'Accumulating for {name}'
        matches_list = matches_df.tolist()
        conf_fp_tp_df, num_labels, num_frames = __accumulate_matches(matches_list, tqdm_string)
        conf_fp_tp_df["precision"] = conf_fp_tp_df["isTP?_sum"]/(conf_fp_tp_df['isTP?_sum']+conf_fp_tp_df['isFP?_sum'])
        conf_fp_tp_df["recall"] = conf_fp_tp_df["isTP?_sum"]/num_labels
        precision_conf_sample = np.interp(conf_samples, conf_fp_tp_df["confidence"], conf_fp_tp_df["precision"])
        recall_conf_sample = np.interp(conf_samples, conf_fp_tp_df["confidence"], conf_fp_tp_df["recall"])
        
        score_curves_dict[name]=tc.ScoreCurves(xs=[recall_conf_sample], ys=[precision_conf_sample],
                                                latents=[conf_samples], x_is_good=True, y_is_good=True, 
                                                x_is_reference=True, curve_names=[name_prefix+name+name_suffix],
                                                x_name=f"Recall IoU {iou_thresh}", y_name=f"Precision IoU {iou_thresh}")

    return score_curves_dict

def roc_a(matches:Sequence[bbs.detection_match], name_prefix='', name_suffix='', force_fp_good:bool=False)-> Dict[str,tc.ScoreCurves]:
    """Creates a roc-a tradeoff curve for a collection of matched detections.
    
    Following https://www.journalfieldrobotics.org/FR/Papers_files/10_Pezzementi.pdf this function
    creates a tradeoffcurve object representing a roc-a metric. This means the x axis is the per 
    object recall and the y axis is a modified negative false positive rate.
    
    Arguments:
        matches:   set of matched and unmatched detections to compute roc-a on
                    
    Returns:
        size 1 TradeoffCurves object, with name as key"""
    name = 'roc_a'
    tqdm_string = f'Accumulating ROCa for {name}'
    (conf_sort, count_labels, count_frames) = __accumulate_matches(matches, tqdm_string)

    TP = conf_sort['isTP?_sum'].to_numpy().reshape((-1,1))
    FP = conf_sort['isFP?_sum'].to_numpy().reshape((-1,1))

    confs = [conf_sort['confidence'].to_numpy().reshape((-1,1))]
    # convert counts into rates
    ys = [TP/count_labels]
    max_fp_per_image = 1
    xs_array, fp_include = __fp_selector(FP=FP, divisor=count_frames, max_rate=max_fp_per_image, force_good=force_fp_good)
    xs = [xs_array]

    # filter out confidences with a greater than allowed fp_rate
    confs[0] = confs[0][fp_include]
    ys[0] = ys[0][fp_include]
    xs[0] = xs[0][fp_include]

    if force_fp_good:
        x_name = f'{max_fp_per_image} - fpr-image'
    else:
        # dummy observation for no detections 
        np.append(xs[0],0)
        np.append(ys[0],0)
        np.append(confs[0],1)
        x_name = 'fpr-image'
    
    names = [name_prefix+name+name_suffix]

    return {name:tc.ScoreCurves(xs=xs,ys=ys,latents=confs, x_is_reference=True,
                            y_is_good=True,x_is_good=force_fp_good,
                            y_name='recall',x_name=x_name,curve_names=names)}

def roc_a_N(matches:Sequence[bbs.detection_match], name_prefix='', name_suffix='', min_views:int=3,force_fp_good:bool=False)-> tc.ScoreCurves:
    """Creates a roc-a Nth strongest tradeoff curve for a collection of matched detections.
    
    This function creates a tradeoffcurve object representing a modified version of the roc-a metric, 
    where only the Nth most confident detection of a track (with more than N labls) is considered in 
    the recall. This modified recall is stored as the x axis.The y axis is a modified negative false 
    positive rate, based on https://www.journalfieldrobotics.org/FR/Papers_files/10_Pezzementi.pdf
    
    Arguments:
        matches:   set of matched and unmatched detections to compute roc-a on
        name:      key used to describe this condition, so this curve can be combined with others
        min_views: N to use for filtering in the recall, default 3
                    
    Returns:
        size 1 TradeoffCurves object, with name as key"""
    name = 'roc_a_N'
    conf_list_TP = []
    conf_list_FP = []
    count_labels_array = np.zeros((0),dtype=np.int64)
    count_labels = 0
    count_frames = 0
    for ndm in tqdm(matches, desc=f'ROCa-N process videos for condition {name}',leave=False): # loop over videos
        count_frames += len(ndm.matches.keys())

        for key in ndm.matches.keys(): # loop over frames
            TP_conf = ndm.matches[key].detections['confidence'].to_numpy().reshape((-1,1))
            TP_id = ndm.matches[key].labels['track_id'].to_numpy().reshape((-1,1)).astype(np.int64)

            # if there are any TPs in the frame, update our running count of track appearances
            if TP_id.size > 0:
                new_track_count = np.max(TP_id) - count_labels_array.shape[0] + 1
                if new_track_count > 0:
                    count_labels_array = np.pad(count_labels_array, ((0, int(new_track_count))))
                count_labels_array[TP_id] += 1

            # columns are: confidence, is FP?, is TP?
            TP_conf = np.concatenate((TP_conf,np.zeros_like(TP_conf),np.ones_like(TP_conf),TP_id),axis=1)
            FP_conf = ndm.unmatched_dets[key]['confidence'].to_numpy().reshape((-1,1))
            FP_conf = np.concatenate((FP_conf,np.ones_like(FP_conf),np.zeros_like(FP_conf)),axis=1)
            conf_list_TP += [TP_conf]
            conf_list_FP += [FP_conf]

            # if there are any FNs in the frame, update our running count of track appearances
            undetected_ids = ndm.unmatched_labs[key]['track_id'].to_numpy().reshape((-1,1)).astype(np.int64)
            if undetected_ids.size > 0:
                new_track_count = np.max(undetected_ids) - count_labels_array.shape[0] + 1
                if new_track_count > 0:
                    count_labels_array = np.pad(count_labels_array, ((0, new_track_count)))
                count_labels_array[undetected_ids] += 1


        # Within one video, find each individual TP that is required to give N detections per track

        conf_TP_sort = np.concatenate(conf_list_TP,axis=0)
        conf_TP_sort = conf_TP_sort[conf_TP_sort[:,0].argsort()[::-1],:]

        idx_sort = np.argsort(conf_TP_sort[:,3], kind='mergesort')

        # sorts records array so all unique elements are together 
        sorted_by_track_id_then_conf = conf_TP_sort[idx_sort,3]

        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_by_track_id_then_conf, return_counts=True, return_index=True)

        idx_start = idx_start[count >= min_views] + min_views-1

        # set of all indices that represent the Nth strongest detection of a track 
        N_th_indices = idx_sort[idx_start]

        # filter down to only the confidences that are the Nth strongest of a track
        conf_TP_sort = conf_TP_sort[N_th_indices,:3]

        conf_list_FP += [conf_TP_sort]

        count_labels += np.sum(count_labels_array >= min_views)

    conf_sort = np.concatenate(conf_list_FP,axis=0)
    conf_sort = conf_sort[conf_sort[:,0].argsort()[::-1],:]

    # count occurences
    TP = np.cumsum(conf_sort[:,2])
    FP = np.cumsum(conf_sort[:,1])

    confs = [conf_sort[:,0]]
    # convert counts into rates
    ys = [TP/count_labels]
    max_fp_per_image = 1
    xs_array, fp_include = __fp_selector(FP=FP, divisor=count_frames, max_rate=max_fp_per_image, force_good=force_fp_good)
    _,unique_conf_indices = np.unique(confs,return_index=True)
    unique_conf_mask = np.zeros(confs[0].shape, dtype=bool)
    unique_conf_mask[unique_conf_indices] = True
    xs = [xs_array]
    # filter out confidences with a greater than allowed fp_rate
    confs[0] = confs[0][fp_include & unique_conf_mask]
    ys[0] = ys[0][fp_include & unique_conf_mask]
    xs[0] = xs[0][fp_include & unique_conf_mask]

    if force_fp_good:
        x_name = f'{max_fp_per_image} - fpr-image'
    else:
        # dummy observation for no detections 
        np.append(xs[0],0)
        np.append(ys[0],0)
        np.append(confs[0],1)
        x_name = 'fpr-image'
    
    names = [name_prefix+name+name_suffix]
    return {name:tc.ScoreCurves(xs=xs,ys=ys,latents=confs, x_is_reference=True,
                            y_is_good=True,x_is_good=force_fp_good,
                            y_name='track-recall',x_name=x_name,curve_names=names)}


#     return tc.ScoreCurves(xs=xs,ys=ys,latents=confs, x_is_reference=True,
                            # y_is_good=True,x_is_good=force_fp_good,
                            # y_name='recall',x_name=x_name,curve_names=names)
    #return tc.TradeoffCurves(xs=ys,ys=xs,latents=confs, y_name='track-recall',x_name='0.1-fptr-image',curve_names=names)

def tracker_metrics(matches:Sequence[bbs.detection_match], name_prefix='',name_suffix='', steps=20, fp_mode='hota-fp-mean', force_fp_good=False, tp_modes:Sequence[str]=['hota-recall-mean'],tracker='sort')-> tc.ScoreCurves:
    """Creates a SORT tracker under HOTA metric tradeoff curve for a collection of matched detections.
    
    This function creates a tradeoffcurve object representing a comparison between the average HOTA recall
    https://github.com/JonathonLuiten/TrackEval. Tracks are created using either the SORT tracker 
    https://github.com/abewley/sort. or BYTE Tracker https://github.com/ifzhang/ByteTrack/tree/main.
    Average HOTA recall is x axis. It is the Arithmetic mean over all overlap thresholds of the 
    geometric mean of the detection recall and the association recall. The y axis is configurable 
    
    All fp's are modifed to be "bigger" is "better", following 
    https://www.journalfieldrobotics.org/FR/Papers_files/10_Pezzementi.pdf
    
    false positive rate options
    fpr-image per image false positive rate, based on https://www.journalfieldrobotics.org/FR/Papers_files/10_Pezzementi.pdf
    fpr-tracked-image, currently only implemented for byte tracker, rate of 0.5 overlap false positives being assiged to live tracks
    hota-fp-0, hota FP rate from first overlap threshold, similar to HOTA-0
    hota-fp-mean, hota FP rate averaged across all overlap thresholds, more similar to how we handle average recall

    true positive rate options
    hota-mean
    hota-recall-mean
    
    Arguments:
        matches: set of matched and unmatched detections to compute on
        name:    key used to describe this condition, so this curve can be combined with others
        steps:   Number of confidence thresholds to evaluate the HOTA for, using distribution of all detections
        fp_mode: string from false positive rate options list above
        tp_mode: string from true positive rate options list above
        tracker: 'sort' or 'byte', choosing which tracker to evaluate

    Returns:
        size 1 ScoreCurves object, with name as key"""
    name = f'tracker_metrics-tracker{tracker}fp_mode{fp_mode}tp_mode{tp_modes}'
    keepcharacters = ('-','.','_')
    name = "".join(c for c in name if c.isalnum() or c in keepcharacters).rstrip()
 
    tqdm_string = f'hota-byte pre-process videos for condition {name}'
    (conf_sort, _, count_frames) = __accumulate_matches(matches, tqdm_string)

    all_conf = conf_sort['confidence'].to_numpy().reshape((-1,1))
    # all_conf_list = []
    # FP_conf_list = []
    FP_used_count_per_threshold = np.zeros(steps)
    # count_frames = 0

    # for ndm in tqdm(matches, desc=f'hota-byte pre-process videos for condition {name}',leave=False): # loop over videos
    #     count_frames += len(ndm.matches.keys())

    #     for key in tqdm(ndm.matches.keys(),leave=False, desc='Finding FP on sequence'): # loop over frames
    #         FP_conf = ndm.unmatched_dets[key]['confidence'].to_numpy().reshape((-1,1))
    #         FP_conf = np.concatenate((FP_conf,np.ones_like(FP_conf),np.zeros_like(FP_conf)),axis=1)
    #         FP_conf_list += [FP_conf]
    #         all_conf_list += [pd.concat((ndm.matches[key].detections,ndm.unmatched_dets[key]), axis=0)['confidence'].to_numpy()]

    # all_conf = np.concatenate(all_conf_list)

    # threshold is a confidence floor, so 1 should have no detections at all
    # convention here is to loop from lowest (many detections) to highest (no detections)
    thresholds = np.quantile(all_conf[:,0], np.linspace(0,1,num=steps,endpoint=False))
    thresholds[0] = 0 # confidence 0 will not appear, but we want to make limits make sense
    
    metric_object_keys_list = []
    if 'hota-mean' in tp_modes or 'hota-recall-mean' in tp_modes:
        metric_object_keys_list.append('hota')
    if 'idf1-mean' in tp_modes:
        metric_object_keys_list.append('identity')
    if 'mota-mean' in tp_modes:
        metric_object_keys_list.append('clear')

    # TODO reorder to video, threshold, frame OR video, frame, threshold to improve data locality
    per_threshold_metrics = []
    for thresh_index, threshold in enumerate(tqdm(thresholds, desc=f'hota {tracker} process for {name}',leave=False)): # loop over confidence thresholds
        per_video_metrics_dict = {k:{} for k in metric_object_keys_list}
        metrics_obj_dict_full = {'hota':hota.HOTA(),'identity':identity.Identity(), 'clear':clear.CLEAR()}
        metrics_obj_dict = {k:metrics_obj_dict_full[k] for k in metric_object_keys_list}


        for video_index, ndm in enumerate(tqdm(matches, desc=f'hota {tracker} process videos for threshold {threshold}',leave=False)): # loop over videos  
        
            # initialize for this video
            count_frames += len(ndm.matches.keys())
            if tracker == 'byte':
                Arg = namedtuple('Arg', ['track_thresh', 'track_buffer','match_thresh','mot20'])
                args = Arg(threshold, 30, 0.8, False)
                byte_tracker = BYTETracker(args, frame_rate=5)
            elif tracker == 'sort':
                sort_tracker = Sort()
            elif tracker == 'oc-sort':
                oc_tracker = OCSort(max(threshold,0.1))
            elif tracker == 'pkf':
                pkf_tracker = PKFTracker(max(threshold,0.1))
            labels_df_list = [] 
            tracks_df_list = []
            FPs_in_each_track : Dict[int,int] = dict() 
            active_tracks_ids = set()
            for key in tqdm(ndm.matches.keys(),leave=False, desc=f'Running {tracker} on sequence  '): # loop over frames
                labels_df = pd.concat((ndm.matches[key].labels,ndm.unmatched_labs[key]), axis=0)
                detections_df = pd.concat((ndm.matches[key].detections,ndm.unmatched_dets[key]), axis=0)
                # SORT conversion
                detections_sort = create_SORT_detections(detections_df)

                # discard detections below threshold
                detections_threshold_keep = np.flatnonzero(detections_df['confidence'].to_numpy() > threshold)
                detections_sort = detections_sort[detections_threshold_keep,:]
                

                if tracker == 'byte':
                    # pass to BYTE tracker assume detections are already scaled
                    track_bbs_ids = byte_tracker.update(detections_sort, [1,1], [1,1])

                    tracks_df = convert_from_BYTE_tracks(track_bbs_ids)
                elif tracker == 'sort':
                    # pass to SORT tracker
                    track_bbs_ids = sort_tracker.update(detections_sort)

                    # return to our format
                    tracks_df = convert_from_SORT_tracks(track_bbs_ids) 
                elif tracker == 'oc-sort':
                    track_bbs_ids = oc_tracker.update(detections_sort, [1,1], [1,1])
                    tracks_df = convert_from_SORT_tracks(track_bbs_ids)
                elif tracker == 'pkf':
                    track_bbs_ids = pkf_tracker.update(detections_sort, [1,1], [1,1])
                    tracks_df = convert_from_SORT_tracks(track_bbs_ids)                
                elif tracker in {'gt-id','gt-full'}:
                    # use detections with matched labels as tracker, 
                    detections_threshold_keep = np.flatnonzero(ndm.matches[key].detections['confidence'].to_numpy() > threshold)
                    tracks_df = ndm.matches[key].labels.copy().iloc[detections_threshold_keep]
                    # simplest to start from labels and overwrite xywh
                    if tracker == 'gt-id':
                        tracks_df[['x','y','w','h']] = ndm.matches[key].detections[['x','y','w','h']].iloc[detections_threshold_keep].to_numpy()
                    if not np.all(np.isfinite(tracks_df[['x','y','w','h']].to_numpy())):
                        print('not finite')

                labels_df_list.append(labels_df)
                tracks_df_list.append(tracks_df)

            if tracker == 'byte':
                FP_used_count_per_threshold[thresh_index] += sum([FPs_in_each_track[track_id] for track_id in active_tracks_ids.intersection(FPs_in_each_track.keys())])

            # compute hota for this video (at this threshold)
            data_dict = create_HOTA_dataset(tracks_df_list, labels_df_list)
            
            for key in metric_object_keys_list:
                per_video_metrics_dict[key][video_index] = metrics_obj_dict[key].eval_sequence(data_dict)

        # combine hota across videos (for this threshold)
        overall_metrics_dict = {k:v.combine_sequences(per_video_metrics_dict[k]) for k,v in metrics_obj_dict.items()}
        per_threshold_metrics.append(overall_metrics_dict)
    tp_modes_reverse = {}
    score_vecs = {}
    if 'hota-recall-mean' in tp_modes:
        hota_recall_vec = np.array([np.mean(np.sqrt( overall_hota['hota']['DetRe'] * overall_hota['hota']['AssRe']))for overall_hota in per_threshold_metrics])
        curve_name = f'tracker_metrics-tracker{tracker}fp_mode{fp_mode}tp_modehota-recall-mean'
        tp_modes_reverse[curve_name] = 'hota-recall-mean'
        score_vecs[curve_name]= hota_recall_vec
    if 'hota-mean' in tp_modes:
        hota_mean_vec = np.array([np.mean( overall_hota['hota']['HOTA']) for overall_hota in per_threshold_metrics])
        curve_name = f'tracker_metrics-tracker{tracker}fp_mode{fp_mode}tp_modehota-mean'
        tp_modes_reverse[curve_name] = 'hota-mean' 
        score_vecs[curve_name]= hota_mean_vec
    if 'mota-mean' in tp_modes:
        curve_name = f'tracker_metrics-tracker{tracker}fp_mode{fp_mode}tp_modemota-mean'
        tp_modes_reverse[curve_name] = 'mota-mean'
        score_vecs[curve_name]= np.array([np.mean( overall_hota['clear']['MOTA']) for overall_hota in per_threshold_metrics])
    if 'idf1-mean' in tp_modes:
        curve_name = f'tracker_metrics-tracker{tracker}fp_mode{fp_mode}tp_modeidf1-mean'
        tp_modes_reverse[curve_name] = 'idf1-mean'
        score_vecs[curve_name]= np.array([np.mean( overall_hota['identity']['IDF1']) for overall_hota in per_threshold_metrics])

    # fp_mode = 'fpr-image' # fpr-tracked-image, # hota-fp-0, hota-fp-mean
    if fp_mode == 'fpr-tracked-image': 
        fp_scale = 100.
        FP = FP_used_count_per_threshold * fp_scale
    elif fp_mode == 'fpr-image':
        fp_scale = 1.
        # conf_sort = np.concatenate(FP_conf_list,axis=0)
        
        # count occurences
        FP = conf_sort['isFP?_sum'].to_numpy().reshape((-1,1))
        if FP.size > 0:
            index_left = np.searchsorted(all_conf[:,0], thresholds ,side='left')
            index_right = index_left + 1

            index_left[index_left >= len(FP)] = len(FP) - 1
            index_right[index_right >= len(FP)] = len(FP) - 1

            FP = np.maximum(FP[index_left],FP[index_right])
        else:
            FP = np.zeros_like(thresholds)
    elif fp_mode == 'hota-fp-0':
        fp_scale = 1.
        FP = np.expand_dims(np.array([overall_hota['HOTA_FP'][0] for overall_hota in per_threshold_metrics]),1) * fp_scale
    elif fp_mode == 'hota-fp-mean':
        fp_scale = 1.
        FP = np.expand_dims(np.array([np.mean(overall_hota['HOTA_FP']) for overall_hota in per_threshold_metrics]),1) * fp_scale
    else:
        print('this should not happen')
    confs = [thresholds]
    # convert counts into rates

    max_fp_per_image = 1
    xs_array, fp_include = __fp_selector(FP=FP[:,0], divisor=count_frames, max_rate=max_fp_per_image, force_good=force_fp_good, scaling=fp_scale)
    xs = [xs_array]
    # filter out confidences with a greater than allowed fp_rate
    score_curve_dict = {}
    confs[0] = confs[0][fp_include]
    xs[0] = xs[0][fp_include]

    for full_parameter_name in score_vecs.keys():
        ys = [score_vecs[full_parameter_name]]
        ys[0] = ys[0][fp_include]

        if not force_fp_good:
            # dummy observation for no detections 
            np.append(xs[0],0)
            np.append(ys[0],0)
            np.append(confs[0],1)

        curve_names = [name_prefix + full_parameter_name + name_suffix]
        tqdm.write(f'storing {curve_names[0]}')
        score_curve_dict[full_parameter_name] = ( tc.ScoreCurves(xs=xs,ys=ys,latents=confs, x_is_reference=True,
                            y_is_good=True,x_is_good=force_fp_good,
                            y_name=tp_modes_reverse[full_parameter_name],x_name=f'{fp_mode}-{fp_scale}',curve_names=curve_names))
    return score_curve_dict

def convert_roca_to_map(input:tc.ScoreCurves,count_labels = 264224, count_frames = 9280)->tc.ScoreCurves:


    xs = []
    ys = []
    confs = []
    for curve_df in input.curves:
        recall_rate = curve_df['ys'].to_numpy()
        if input.x_is_good:
            fp_rate = curve_df['xs'].to_numpy()
        else:
            fp_rate = 1-curve_df['xs'].to_numpy()

        c = curve_df['latents'].to_numpy()
        tp_count = recall_rate * count_labels
        fp_count = fp_rate * count_frames
        y = tp_count/(fp_count+tp_count)
        x = recall_rate

        xs.append(x)
        ys.append(y)
        confs.append(c)
    y_name = 'precision'
    x_name = 'confidence'
    y_is_good = input.y_is_good
    curve_names = input.curve_names

    flipped_curve = tc.ScoreCurves(xs=xs,ys=ys,latents=confs, x_is_reference=True,
                          y_is_good=y_is_good,x_is_good=True,
                          y_name=y_name,x_name=x_name,curve_names=curve_names)
    return flipped_curve

def replace_x_w_recall(input:tc.ScoreCurves,pr:tc.ScoreCurves,count_labels = 264224, count_frames = 9280)->tc.ScoreCurves:


    xs = []
    ys = []
    confs = []
    for index, curve_df in enumerate(input.curves):
        conf = curve_df['latents'].to_numpy()
        pr_recall = pr[index]['xs'].to_numpy()
        pr_conf = pr[index]['latents'].to_numpy()
        x = np.interp(conf, pr_conf, pr_recall)

        xs.append(x)
        ys.append(curve_df['ys'].to_numpy())
        confs.append(conf)
    y_name = input.y_name
    x_name = pr.x_name
    y_is_good = input.y_is_good
    curve_names = input.curve_names

    flipped_curve = tc.ScoreCurves(xs=xs,ys=ys,latents=confs, x_is_reference=True,
                          y_is_good=y_is_good,x_is_good=True,
                          y_name=y_name,x_name=x_name,curve_names=curve_names)
    return flipped_curve

def convert_tradeoff_fp_goodness(input:tc.ScoreCurves, add_max_fp_obs:bool=False, use_latent=False)->tc.ScoreCurves:
    """returns new tradeoff curve with fp swapped between good and bad"""

    x_is_good = not input.x_is_good
    max_fp_rate = 1
    xs = []
    ys = []
    confs = []
    for curve_df in input.curves:
        x = max_fp_rate - curve_df['xs'].to_numpy()
        y = curve_df['ys'].to_numpy()
        c = curve_df['latents'].to_numpy()
        if use_latent:
            x = c
            x_name = f'confidence'
            x_is_good = None
        else:

            if add_max_fp_obs and not input.x_is_good:
                if x[0] > x[-1]:
                    x = x.append(0)
                    y = y.append(y[-1])
                    c = c.append(c[-1] + 1e-9 * np.sign(c[-1]-c[0]))
                else:
                    x = np.insert(x,0,0)
                    y = np.insert(y,0,y[0])
                    c = np.insert(c,0,c[0] - 1e-9 * np.sign(c[-1]-c[0]))
            x_name = f'{input.x_name}_negate'
        xs.append(x)
        ys.append(y)
        confs.append(c)
    y_is_good = input.y_is_good

    y_name = input.y_name
    curve_names = input.curve_names

    flipped_curve = tc.ScoreCurves(xs=xs,ys=ys,latents=confs, x_is_reference=True,
                          y_is_good=y_is_good,x_is_good=x_is_good,
                          y_name=y_name,x_name=x_name,curve_names=curve_names)
    return flipped_curve

def convert_from_BYTE_tracks(track_list : List[STrack]) -> pd.DataFrame:
    """Convert BYTE track object format into bounding_box_structs dataframe format"""

    if len(track_list) == 0:
        return pd.DataFrame(data=np.zeros((0,len(bbs.LabelsDict.COLUMN_NAMES))),columns=bbs.LabelsDict.COLUMN_NAMES)
    else:
        track_np_list = []
        for t in track_list:
            tlwh = np.expand_dims(t.tlwh,axis=0)
            tid = np.array([[t.track_id]])
            track_np_list.append(np.concatenate((tlwh,tid),axis=1))
        track_bbs_ids = np.concatenate(track_np_list,axis=0)
        zeros = np.zeros_like(track_bbs_ids[:,(2,)])
        ones = np.ones_like(track_bbs_ids[:,(2,)])
        track_bbs_ids = np.concatenate((track_bbs_ids, ones, zeros, ones, zeros), axis=1)

        return pd.DataFrame(index=track_bbs_ids[:,4],data=track_bbs_ids,columns=bbs.LabelsDict.COLUMN_NAMES)

def convert_from_SORT_tracks(track_bbs_ids:npt.NDArray)->pd.DataFrame:
    """Convert SORT numpy tracks format into bounding_box_structs dataframe format"""

    # convert to xywh
    track_bbs_ids[:,2] = track_bbs_ids[:,2] - track_bbs_ids[:,0]
    track_bbs_ids[:,3] = track_bbs_ids[:,3] - track_bbs_ids[:,1]
    
    zeros = np.zeros_like(track_bbs_ids[:,(2,)])
    ones = np.ones_like(track_bbs_ids[:,(2,)])
    track_bbs_ids = np.concatenate((track_bbs_ids, ones, zeros, ones, zeros), axis=1)

    # collect in dataframe for ease of access later
    return pd.DataFrame(index=track_bbs_ids[:,4],data=track_bbs_ids,columns=bbs.LabelsDict.COLUMN_NAMES)

def create_SORT_detections(detections_df : pd.DataFrame)->npt.NDArray:
    """Convert from bounding_box_structs dataframe to SORT numpy array format"""
    x = np.expand_dims(detections_df['x'].to_numpy(),axis=1)
    y = np.expand_dims(detections_df['y'].to_numpy(),axis=1)
    w = np.expand_dims(detections_df['w'].to_numpy(),axis=1)
    h = np.expand_dims(detections_df['h'].to_numpy(),axis=1)
    c = np.expand_dims(detections_df['confidence'].to_numpy(),axis=1)
    return np.concatenate((x,y,x+w,y+h,c),axis=1)

def create_HOTA_dataset(tracks_list : List[pd.DataFrame], gt_list: List[pd.DataFrame])->Dict[str,Union[int, List[float], List[npt.NDArray]]]:
    """Creates a dict matching the expected format for the TrackEval library.
    
    This function creates a dict mapping to lists of data from lists of track and gt data. Follows:
    https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/datasets/_base_dataset.py#L67
    
    Arguments:
        tracks_list: list of track dataframes, matching the LabelsDict in bounding_box_structs
        gt_list:     list of labeled dataframes, matching the LabelsDict in bounding_box_structs

    Returns:
        dict trackeval dataset, compartible with HOTA and other metrics"""


    data_dict : Dict[str,Union[int, List[float],List[npt.NDArray]]] = {}

    # find all tracker ids
    raw_tracker_ids = [(df['track_id'].astype('int').to_numpy() - 1) for df in tracks_list]
    raw_gt_ids = [(df['track_id'].astype('int').to_numpy() - 1) for df in gt_list]

    # filter to unique ids
    unique_tracker_ids = np.array(list(set([tr_det for tr_dets in raw_tracker_ids for tr_det in tr_dets])))
    unique_gt_ids = np.array(list(set([tr_gt for tr_gts in raw_gt_ids for tr_gt in tr_gts])))

    # remap unique ids to contiguous ids starting at 0
    data_dict['tracker_ids'] = [np.nonzero(np.isin(unique_tracker_ids,raw_tracker_ids_t,assume_unique=True))[0] for raw_tracker_ids_t in raw_tracker_ids]
    data_dict['gt_ids'] = [np.nonzero(np.isin(unique_gt_ids,raw_gt_ids_t,assume_unique=True))[0] for raw_gt_ids_t in raw_gt_ids]
    
    # split fields from dataframes into individual lists
    data_dict['gt_classes'] = [df['label_class'].astype('int').to_list() for df in gt_list]
    data_dict['tracker_classes'] = [df['label_class'].astype('int').to_list() for df in tracks_list]
    data_dict['tracker_confidences'] = [df['label_confidence'].to_list() for df in tracks_list]

    # count things
    data_dict['num_timesteps'] = len(tracks_list)
    data_dict['num_tracker_dets'] = sum([len(track_dets) for track_dets in raw_tracker_ids])
    data_dict['num_gt_dets'] = sum([len(gt_dets) for gt_dets in raw_gt_ids])
    data_dict['num_tracker_ids'] = len(unique_tracker_ids)
    data_dict['num_gt_ids'] = len(unique_gt_ids)

    # compute bounding box IOU for each frame
    data_dict['similarity_scores'] = [bbs.MatchedDetectionsDict.compute_similarity(track,gt) for (track, gt) in tqdm(zip(tracks_list, gt_list), leave=False, desc="matching for HOTA")]
   
    return data_dict