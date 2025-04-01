# CATE Detection Evaluation code
 
# Copyright 2025 Carnegie Mellon University.
 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
 
# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
 
# DM25-0275

from deeplite_torch_zoo.src.object_detection.eval.evaluate import scale_boxes 
from deeplite_torch_zoo.src.object_detection.datasets.utils import xyxy2xywh, xywh2xyxy
import bounding_box_structs as bbs
import tradeoff_curve as tc
import tracking_approx_metrics as tam

from deeplite_torch_zoo.src.object_detection.eval.utils import (
    box_iou,
)
import pickle

import os
import math
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import fnmatch

import matplotlib.pylab as plt

from pathos.multiprocessing import ProcessingPool as Pool

dir_path = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(dir_path,'..','model-characterizer','ttabor.json')
with open(json_path, "rb") as f:
    config = json.load(f)
    

## Configuration

load_matches = True # try to load an existing cached set of matched gt/detections
load_tradeoffs = False # try to load set of tradeoff curve objects
num_process = 30 # number of processes to use for computing tradeoff curves
swap_fp = True

db_root = config['metadata']['db_root']
annotation_root = '/mnt/drive/tracking-dataset/person_path_22_data/person_path_22/person_path_22-test/'
data_root = '/mnt/drive/tracking-dataset/dataset/personpath22/raw_data/'
output_dir = '/mnt/drive/itar-ttabor/trade_offs'


model_filters = ['yolo4*', 'yolo5*', 'fasterrcnn_resnet50_fpn_v2',
'fcos_resnet50_fpn', 'retinanet_resnet50_fpn','ssd300_vgg16',
'ssdlite320_mobilenet_v3_large']
# models_complete = ['yolo11m', 'rtdetr-l', 'yolo11n', 'rtdetr-x']


'yolo5n_hswish_640','yolo4s_640','yolo5n_relu_640','yolo4m_640','yolo5m_640','yolo5s_640', 'yolov8n',

# models_white_list = ['yolov8l','yolov10m','yolov10l','yolo11m', 'rtdetr-l', 'yolo11n', 'rtdetr-x',
#                      'yolov5m_640', 'yolov5s_640', 'yolov5n_640','yolov5n_fullres','yolov5s_fullres','yolov5m_fullres',
#                      'yolov10l_fullres','yolov8l_fullres','yolov10m_fullres',
#                      'yolov8n_fullres','rtdetr-x_fullres','rtdetr-l_fullres',
#                      'yolo11n_fullres','yolo11m_fullres']

models_white_list = ['yolo5n_hswish_640','yolo4s_640','yolo5n_relu_640','yolo4m_640','yolo5m_640','yolo5s_640',
                     'yolov8n','yolov8l','yolov10m','yolov10l','yolo11m', 'rtdetr-l', 'yolo11n', 'rtdetr-x',
                     'yolov5m_640', 'yolov5s_640', 'yolov5n_640',]

index_list = []
matches_list = []
experiment_list = []

# load all detection files, ground truth files, and perform matching between them
if not load_matches:

    sys.path.append(os.path.join(dir_path,'..','model-characterizer'))
    from query_evals import evaluations_dictionary

    evaluations_dir = os.path.join(db_root,'evaluations')
    eval_dict = evaluations_dictionary([json_path])
    for exp_id in tqdm(eval_dict.keys(),desc='indexing experiments: ',total=len(eval_dict.keys()),leave=False):
        model_name = eval_dict[exp_id]['model']['architecture']
        if not any([fnmatch.fnmatch( model_name, model_filter) for model_filter in models_white_list]): continue
        video_name = eval_dict[exp_id]['dataset']['name']
        perturbation_key = str((eval_dict[exp_id]['perturbation']['name'],str(eval_dict[exp_id]['perturbation']['parameters'])))
        experiment_list.append(exp_id)
        index_list.append((model_name,perturbation_key,video_name))

        # video_file = os.path.join(data_root, video_name)

    experiment_multi_index = pd.MultiIndex.from_tuples(index_list, names=['model','perturbation','video'])
    experiments_pd = pd.Series(experiment_list, index=experiment_multi_index)

    def load_match_save( group_by ):
        model_tuple, df = group_by
        model_name = model_tuple[0]
        inner_index_list = []
        matches_list = []
        # i = 0
        for (_, perturbation_key, video_name), exp_id in df.items():
            # i += 1
            # if i > 5:
            #     break
            det_dir = os.path.join(evaluations_dir, exp_id,'detections')
            if os.path.isdir(det_dir):
                detections = bbs.DetectionsDict.load_directory(det_dir,filter_class=[0])
                annotation_file = os.path.join(annotation_root, video_name, 'gt/gt.txt')
                labels = bbs.LabelsDict.load_video_labels(annotation_file)
                detection_match = bbs.MatchedDetectionsDict.match_detections(detections=detections,labels=labels)
                matches_list.append(detection_match)
                inner_index_list.append((model_name,perturbation_key,video_name))
            else:
                open(f'{exp_id}_is_missing','wb').close()
        index = pd.MultiIndex.from_tuples(inner_index_list, names=['model','perturbation','video'])
        matches_pd = pd.Series(matches_list, index=index)
        cache_path = f'{model_name}_matches.pickle'
        file = open(cache_path, 'wb')
        pickle.dump(matches_pd, file)
        file.close()
        return (model_name, cache_path)

    # most of the work is here, so it has a parallel mode. Single process mode is special case, for debug

    
    exper_iter = experiments_pd.groupby(['model'])

    if num_process == 1:
        matches_list = list(tqdm(map(load_match_save, exper_iter),desc='loading matches', leave=False, total=len(list(exper_iter))))
    else:
        with Pool(num_process) as p:
            matches_list = list(tqdm(p.imap(load_match_save, exper_iter),desc='loading matches', leave=False, total=len(list(exper_iter))))

    models_matched = pd.DataFrame(matches_list, columns=['model', 'cache_path'])

    print('saving to matches-ultra.pickle')
    # with open() as file:
    file = open('matches-ultra.pickle', 'wb')
    pickle.dump(models_matched, file)
    file.close()
else:
    print('loading from matches-ultra.pickle')
    matches_list = []
    for model_name in models_white_list:
        cache_path = f'{model_name}_matches.pickle'
        matches_list.append((model_name, cache_path))
    models_matched = pd.DataFrame(matches_list, columns=['model', 'cache_path'])
    # file = open('matches-ultra.pickle', 'rb')
    # models_matched = pickle.load(file)
    # file.close()
    # print('loaded from matches-ultra.pickle')
    


# using matched detections and ground truth, run list of specified metrics
# if loading existing tradeoff file, will only have metrics available in that file
tradeoff_file_name = 'tradeoffs.pickle'
if not load_tradeoffs:
    iou_threshes = np.arange(0.05, 0.955, 0.05).tolist()
    
    tradeoff_funcs = [
                      partial(tam.tracker_metrics,tracker='gt-id',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='pkf',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='oc-sort',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='sort',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='byte',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='gt-full',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.map,iou_threshes=iou_threshes),
                      tam.roc_a,tam.roc_a_N,]
    tradeoff_funcs = [partial(tam.map,iou_threshes=iou_threshes),]
    # Example of using it witha parameter

    for (model_name, path) in tqdm(models_matched.itertuples(index=False), leave=False, desc ='computing tradeoffs'):
        file = open(f'{model_name}_matches.pickle', 'rb')
        matches_pd = pickle.load(file)
        
        file.close()
        cache_path = f'{model_name}_trade_offs.pickle'
        if os.path.isfile(cache_path):
            file = open(f'{model_name}_trade_offs.pickle', 'rb')
            tradeoffs_pd_existing = pickle.load(file)
            file.close()
        else:
            tradeoffs_pd_existing = None
        tradeoffs_pd = {}
        index_list = []
        func_list = []
        df_list = []
        for tradeoff_func in tqdm(tradeoff_funcs,leave=False,desc=f'tradeoffs for {model_name}'):


            grouped_matches = matches_pd.groupby(level=[1])
            def split_apply( triple ): return triple[1](triple[0].to_list(), name_prefix="('", name_suffix="'"+str(triple[2])[1:])
            # def split_apply( name-func-, func=tradeoff_func ): return func(pair[0].to_list(), name=str(pair[1]))
            # trick for letting tradeoff_funcs list contain partial objects with bound arguments
            if isinstance(tradeoff_func, partial):
                func_names = []
                keywords = tradeoff_func.keywords
                if 'tp_mode' in keywords:
                    for tp_mode in keywords['tp_modes']:
                        func_name = f'tracker_metrics-tracker{keywords["tracker"]}fp_mode{keywords["fp_mode"]}tp_mode{tp_mode}'
                        # func_name = f'{tradeoff_func.func.__name__}-{tradeoff_func.keywords}'
                        keepcharacters = ('-','.','_')
                        func_name = "".join(c for c in func_name if c.isalnum() or c in keepcharacters).rstrip()
                        func_names.append(func_name)
                if 'iou_threshes' in keywords:
                    for iou_thresh in keywords['iou_threshes']:
                        func_name = f'map_{iou_thresh:.2f}'
                        # func_name = f'{tradeoff_func.func.__name__}-{tradeoff_func.keywords}'
                        keepcharacters = ('-','.','_')
                        func_name = "".join(c for c in func_name if c.isalnum() or c in keepcharacters).rstrip()
                        func_names.append(func_name)
                tqdm.write(f'{func_names}')
            else:
                func_names = tradeoff_func.__name__
            # check if we've already run this tradeoff curve
            for model_perturbation, new_df in grouped_matches:
                count = 0
                for func_name in func_names:
                    index_tuple = (func_name, model_name, model_perturbation)
                    if tradeoffs_pd_existing is not None and index_tuple in tradeoffs_pd_existing.index:
                        count = count + 1
                if count == len(func_names):
                    continue
                index_list.append((model_name, model_perturbation))
                func_list.append(tradeoff_func)
                df_list.append(new_df)

            
        # most of the work is here, so it has a parallel mode. Single process mode is special case, for debug
        if num_process == 1:
            tradeoff_list = list(tqdm(map(split_apply, zip(df_list,func_list,index_list)),desc=f'computing {func_name}', leave=False, total=len(df_list)))
        else:
            with Pool(num_process) as p:
                tradeoff_list = list(tqdm(p.imap(split_apply, zip(df_list,func_list,index_list)),desc=f'computing {func_name}', leave=False, total=len(df_list)))

        # convert results into pandas datastructure for indexing ease
        index_list_expanded = []
        tradeoff_list_expanded = []
        for ind, tradeoff_dict in enumerate(tradeoff_list):
            for k, v in tradeoff_dict.items():
                tradeoff_list_expanded.append(v)
                index_list_expanded.append((k, index_list[ind][0],  index_list[ind][1]))
        experiment_multi_index = pd.MultiIndex.from_tuples(index_list_expanded, names=['func_name','model','perturbation'])        
        tradeoffs_pd = pd.Series(tradeoff_list_expanded, index=experiment_multi_index)
        if tradeoffs_pd_existing is not None :
            tradeoffs_pd = pd.concat((tradeoffs_pd,tradeoffs_pd_existing))
            tradeoffs_pd = tradeoffs_pd[~tradeoffs_pd.index.duplicated(keep='first')]
        
        file = open(cache_path, 'wb')
        pickle.dump(tradeoffs_pd, file)
        file.close()

#     # save, for skipping next time
    file_roca = open(tradeoff_file_name, 'wb')
    pickle.dump(tradeoff_funcs, file_roca)
    file_roca.close()
    print(f'saved to {tradeoff_file_name}')
else:
    print(f'loading from {tradeoff_file_name}')
    file_roca = open(tradeoff_file_name, 'rb')
    tradeoff_funcs = pickle.load(file_roca)
    file_roca.close()
    print(f'loaded from {tradeoff_file_name}')

# Plot all tradeoffs AND generate heat map tables for them

# for (model_name, path) in tqdm(models_matched.itertuples(index=False)):

# def compute_areas(model_name):
#     file = open(f'{model_name}_trade_offs.pickle', 'rb')
#     tradeoffs_pd = pickle.load(file)
#     file.close()

#     nominal_pd_area_list = []
#     perturb_pd_area_list = []
#     perturb_naive_pd_area_list = []
#     any_pd_area_list = []
#     for (tradeoff_name,),tradeoff_df in tqdm(tradeoffs_pd.groupby(['func_name']), leave=False, desc='plotting tradeoffs'):
#         if 'roc_a' in tradeoff_name:
#             tqdm.write(tradeoff_name)
#         perturbations_all = tradeoff_df.index.get_level_values('perturbation')
#         nominal_key="('no_op', '{}')"
#         perturbations = set(perturbations_all)
#         perturbations.remove( nominal_key )

#         index_w_perturb_list = []
#         index_list = []
#         tradeoff_worst_list = []
#         tradeoff_nominal_list = []
#         tradeoff_any_list = []
#     # for model, new_df in tqdm(tradeoffs_dict[tradeoff_name].groupby(level=[0]),desc='computing worst', leave=False):
#         index_list.append((tradeoff_name,model_name))
        
#         if swap_fp:
#             tradeoff_nominal_list.append(tam.convert_tradeoff_fp_goodness(tradeoffs_pd[tradeoff_name][model_name][nominal_key]))
#             tradeoff_any_list.append(tradeoff_nominal_list[-1])
#         else:
#             tradeoff_nominal_list.append(tradeoffs_pd[tradeoff_name][model_name][nominal_key])
#             tradeoff_any_list.append(tradeoff_nominal_list[-1])
#         index_counter = 0
#         # plot tradeoffs, per perturbation
#         for key in tqdm(perturbations,desc=f"{model_name} perturbation",leave=False):
#             index_w_perturb_list.append((tradeoff_name,model_name,key))
#             if swap_fp:
#                 curve_perturb = tam.convert_tradeoff_fp_goodness(tradeoffs_pd[tradeoff_name][model_name][key])
#             else:
#                 curve_perturb = tradeoffs_pd[tradeoff_name][model_name][key]
#             merged_curves = curve_perturb + tradeoff_nominal_list[-1]
#             tradeoff_any_list[-1] = tradeoff_any_list[-1]  + curve_perturb
#             tradeoff_worst_list.append(merged_curves)
#             index_counter += 1 
#             tradeoff_worst_list[-1].plot(f'plot_worst_{tradeoff_name}_{model_name}_{index_counter}.png',show_latent=True,show_worst=True)
#         # plot tradeoffs with all perturbations
#         tradeoff_any_list[-1].plot(f'plot_worst_{tradeoff_name}_{model_name}_all.png',show_latent=True,show_worst=True)

#         # convert to pandas for ease of indexing
#         index_w_perturb = pd.MultiIndex.from_tuples(index_w_perturb_list, names=['tradeoff','model','perturbation'])
#         index= pd.MultiIndex.from_tuples(index_list, names=['tradeoff','model',])


#         nominal_pd = pd.Series(tradeoff_nominal_list, index=index)
#         any_pd = pd.Series(tradeoff_any_list, index=index)
#         perturb_pd = pd.Series(tradeoff_worst_list, index=index_w_perturb)

#         # use pandas sugar to get areas for heat maps
#         nominal_pd_area_list.append(nominal_pd.apply(lambda tcs: tcs.get_area(0)))
#         perturb_pd_area_list.append(perturb_pd.apply(lambda tcs: tcs.worst_case().get_area(0)))
#         perturb_naive_pd_area_list.append(perturb_pd.apply(lambda tcs: tcs.get_area(0)))
#         any_pd_area_list.append(any_pd.apply(lambda tcs: tcs.worst_case().get_area(0)))

#     nominal_pd_area = pd.concat(nominal_pd_area_list)
#     perturb_pd_area = pd.concat(perturb_pd_area_list)
#     perturb_naive_pd_area = pd.concat(perturb_naive_pd_area_list)
#     any_pd_area = pd.concat(any_pd_area_list)

#     return (nominal_pd_area,perturb_pd_area,perturb_naive_pd_area,any_pd_area)

# # most of the work is here, so it has a parallel mode. Single process mode is special case, for debug
# if num_process == 1:
#     tradeoff_list = list(tqdm(map(compute_areas, models_matched['model'].to_list()),desc='computing areas', leave=False, total=len(models_matched)))
# else:
#     with Pool(num_process) as p:
#         tradeoff_list = list(tqdm(p.imap(compute_areas, models_matched['model'].to_list()),desc='computing areas', leave=False, total=len(models_matched)))

# nominal_pd_area_list = []
# perturb_pd_area_list = []
# perturb_naive_pd_area_list = []
# any_pd_area_list = []
# for (nominal_pd_area,perturb_pd_area,perturb_naive_pd_area,any_pd_area) in tradeoff_list:
#     nominal_pd_area_list.append(nominal_pd_area)
#     perturb_pd_area_list.append(perturb_pd_area)
#     perturb_naive_pd_area_list.append(perturb_naive_pd_area)
#     any_pd_area_list.append(any_pd_area)

# nominal_pd_area = pd.concat(nominal_pd_area_list)
# perturb_pd_area = pd.concat(perturb_pd_area_list)
# perturb_naive_pd_area = pd.concat(perturb_naive_pd_area_list)
# any_pd_area = pd.concat(any_pd_area_list)

# area_file_name = 'areas.pickle'
# file_roca = open(area_file_name, 'wb')
# pickle.dump({'nominal_pd_area':nominal_pd_area, 'perturb_pd_area':perturb_pd_area, 'any_pd_area':any_pd_area}, file_roca)
# file_roca.close()
# print(f'saved to {area_file_name}')

# for tradeoff_name in tqdm(nominal_pd_area.index.unique('tradeoff'), leave=False, desc='creating heatmaps'):

#     perturb_naive_pd_area_df = perturb_naive_pd_area[tradeoff_name].unstack(level='perturbation')
#     perturb_pd_area_df=perturb_pd_area[tradeoff_name].unstack(level='perturbation')

#     nominal_pd_area_np = nominal_pd_area[tradeoff_name].to_numpy().reshape((1,-1))
#     any_pd_area_np = any_pd_area[tradeoff_name].to_numpy().reshape((1,-1))

#     perturb_pd_area_np = np.concatenate((any_pd_area_np,perturb_pd_area_df.to_numpy().transpose()),axis=0)
#     perturb_naive_pd_area_np = np.concatenate((np.full_like(any_pd_area_np,np.nan),perturb_naive_pd_area_df.to_numpy().transpose()),axis=0)

#     perturbations_strings = ['any']+perturb_pd_area_df.columns.tolist()
#     model_strings = list(nominal_pd_area[tradeoff_name].index)
#     perturbations_strings_with_nominal = ['any', *perturb_pd_area_df.columns.tolist(), 'nominal']
#     # heatmap table, title, colormap maximum (-1 computes max on data)
#     heatmap_settings = ((perturb_pd_area_np,'raw-worst-case',-1),
#         (perturb_naive_pd_area_np,'raw-naive',-1),
#         (perturb_pd_area_np/nominal_pd_area_np,'ratio-worst-case',1.2),
#         (perturb_naive_pd_area_np/nominal_pd_area_np,'ratio-naive',1.2))
#         # generate heatmaps
#     for perturb_np,name,max_val in heatmap_settings:
#         fig, ax_dict = plt.subplot_mosaic([['A panel'],['B panel']], sharex=True,
#                                              height_ratios=[19,1],
#                                              width_ratios=[1])
#         ax_pert = ax_dict['A panel']
#         ax_nom = ax_dict['B panel']
#         # fig = plt.figure()
#         # gs = fig.add_gridspec(2, 1, height_ratios=(19, 1))
#         # # Create the Axes.
#         # ax_pert = fig.add_subplot(gs[0, 0])
#         # ax_nom = fig.add_subplot(gs[1, 0], sharex=ax_pert)
        
#         save_df = pd.DataFrame(np.concatenate((perturb_np, nominal_pd_area_np),axis=0), columns = model_strings, index=perturbations_strings_with_nominal)
#         save_df.to_csv(f'table_{tradeoff_name}_{name}.csv')

#         if max_val < 0:
#             max_val = max(np.max(np.nan_to_num(perturb_np)),np.max(np.nan_to_num(nominal_pd_area_np)))

        
#         im_pert = ax_nom.imshow(nominal_pd_area_np,vmin=0, vmax=max_val,aspect='auto')
#         im_nom = ax_pert.imshow(perturb_np,vmin=0, vmax=max_val,aspect='auto')
        

#         ax_nom.set_xticks(np.arange(len(model_strings)),labels=model_strings)
#         ax_pert.set_yticks(np.arange(len(perturbations_strings)),labels=perturbations_strings)
#         ax_nom.set_yticks((0,),labels=['nominal'])

#         plt.setp(ax_nom.get_xticklabels(), rotation=45, ha="right",
#                 rotation_mode="anchor")

#         for model_ind in range(len(model_strings)):
#             # if nominal_pd_area_np[0, model_ind]/max_val < 0.5:
#             #     c = 'w'
#             # else:
#             #     c = 'k'
#             c = 'k'
#             text = ax_nom.text(model_ind, 0, round(nominal_pd_area_np[0, model_ind],2),
#                         ha="center", va="center", color=c, fontsize='small')
#             for pert_ind in range(len(perturbations_strings)):
#                 # if perturb_np[pert_ind, model_ind]/max_val < 0.5:
#                 #     c = 'w'
#                 # else:
#                 #     c = 'k'
#                 c = 'k'
#                 text = ax_pert.text(model_ind, pert_ind, round(perturb_np[pert_ind, model_ind],2),
#                             ha="center", va="center", color=c , fontsize='small')
#         fig.suptitle(name)
#         #fig.subplots_adjust(right=0.8)
#         #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#         #fig.colorbar(im_nom, cax=cbar_ax)
#         fig.colorbar(im_nom,  ax=(ax_pert,ax_nom), shrink=0.8)
#         # ,bbox_inches='tight'
#         plt.savefig(f'table_{tradeoff_name}_{name}.png', dpi=300,bbox_inches='tight')
#         plt.clf()
#         plt.close()