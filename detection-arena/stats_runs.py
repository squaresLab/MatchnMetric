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
from statsmodels.formula.api import ols
import statsmodels.api as sm
import itertools

import matplotlib.pylab as plt

from pathos.multiprocessing import ProcessingPool as Pool

dir_path = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(dir_path,'..','model-characterizer','ttabor.json')
with open(json_path, "rb") as f:
    config = json.load(f)
    
create_plots_heatmaps = True
use_max = True
use_recall = False
if use_max and use_recall:
    print('bad do not do')
    sys.exit()
## Configuration

load_tradeoffs = False # try to load set of tradeoff curve objects
num_process = 32 # number of processes to use for computing tradeoff curves
swap_fp = True

db_root = config['metadata']['db_root']
annotation_root = '/mnt/drive/tracking-dataset/person_path_22_data/person_path_22/person_path_22-test/'
data_root = '/mnt/drive/tracking-dataset/dataset/personpath22/raw_data/'
output_dir = '/mnt/drive/itar-ttabor/trade_offs'


model_filters = ['yolo4*', 'yolo5*', 'fasterrcnn_resnet50_fpn_v2',
'fcos_resnet50_fpn', 'retinanet_resnet50_fpn','ssd300_vgg16',
'ssdlite320_mobilenet_v3_large']
# models_complete = ['yolo11m', 'rtdetr-l', 'yolo11n', 'rtdetr-x']




models_white_list = ['yolo5n_hswish_640','yolo4s_640','yolo5n_relu_640','yolo4m_640','yolo5m_640','yolo5s_640',
                     'yolov8n','yolov8l','yolov10m','yolov10l','yolo11m', 'rtdetr-l', 'yolo11n', 'rtdetr-x',
                     'yolov5m_640', 'yolov5s_640', 'yolov5n_640','yolov5n_fullres','yolov5s_fullres','yolov5m_fullres',
                     'yolov10l_fullres','yolov8l_fullres','yolov10m_fullres',
                     'yolov8n_fullres','rtdetr-x_fullres','rtdetr-l_fullres',
                     'yolo11n_fullres','yolo11m_fullres',]

models_white_list = ['yolo5n_hswish_640','yolo4s_640', 'yolo5n_relu_640','yolo4m_640','yolo5m_640','yolo5s_640',
                     'yolov8n','yolov8l','yolov10m','yolov10l','yolo11m','rtdetr-l','yolo11n','rtdetr-x',
                     'yolov5m_640', 'yolov5s_640', 'yolov5n_640']

index_list = []
matches_list = []
experiment_list = []


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

    
    tradeoff_funcs = [partial(tam.tracker_metrics,tracker='pkf',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='gt-id',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='oc-sort',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='sort',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='byte',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      partial(tam.tracker_metrics,tracker='gt-full',fp_mode='fpr-image',tp_modes=['hota-mean','idf1-mean','mota-mean']),
                      tam.roc_a,tam.roc_a_N]
    # Example of using it witha parameter

#     for (model_name, path) in tqdm(models_matched.itertuples(index=False), leave=False, desc ='computing tradeoffs'):
#         file = open(f'{model_name}_matches.pickle', 'rb')
#         matches_pd = pickle.load(file)
        
#         file.close()
#         cache_path = f'{model_name}_trade_offs.pickle'
#         if os.path.isfile(cache_path):
#             file = open(f'{model_name}_trade_offs.pickle', 'rb')
#             tradeoffs_pd_existing = pickle.load(file)
#             file.close()
#         else:
#             tradeoffs_pd_existing = None
#         tradeoffs_pd = {}
#         index_list = []
#         func_list = []
#         df_list = []
#         for tradeoff_func in tqdm(tradeoff_funcs,leave=False,desc=f'tradeoffs for {model_name}'):


#             grouped_matches = matches_pd.groupby(level=[1])
#             def split_apply( triple ): return triple[1](triple[0].to_list())
#             # def split_apply( name-func-, func=tradeoff_func ): return func(pair[0].to_list(), name=str(pair[1]))
#             # trick for letting tradeoff_funcs list contain partial objects with bound arguments
#             if isinstance(tradeoff_func, partial):
#                 func_names = []
#                 keywords = tradeoff_func.keywords
#                 for tp_mode in keywords['tp_modes']:
#                     func_name = f'tracker_metrics-tracker{keywords["tracker"]}fp_mode{keywords["fp_mode"]}tp_mode{tp_mode}'
#                     # func_name = f'{tradeoff_func.func.__name__}-{tradeoff_func.keywords}'
#                     keepcharacters = ('-','.','_')
#                     func_name = "".join(c for c in func_name if c.isalnum() or c in keepcharacters).rstrip()
#                     func_names.append(func_name)
#                 print(func_names)
#             else:
#                 func_names = tradeoff_func.__name__
#             # check if we've already run this tradeoff curve
#             for model_perturbation, new_df in grouped_matches:
#                 count = 0
#                 for func_name in func_names:
#                     index_tuple = (func_name, model_name, model_perturbation)
#                     if tradeoffs_pd_existing is not None and index_tuple in tradeoffs_pd_existing.index:
#                         count = count + 1
#                 if count == len(func_names):
#                     continue
#                 index_list.append((model_name, model_perturbation))
#                 func_list.append(tradeoff_func)
#                 df_list.append(new_df)


            
#         # # most of the work is here, so it has a parallel mode. Single process mode is special case, for debug
#         # if num_process == 1:
#         #     tradeoff_list = list(tqdm(map(split_apply, zip(df_list,func_list)),desc=f'computing {func_name}', leave=False, total=len(df_list)))
#         # else:
#         #     with Pool(num_process) as p:
#         #         tradeoff_list = list(tqdm(p.imap(split_apply, zip(df_list,func_list)),desc=f'computing {func_name}', leave=False, total=len(df_list)))

#         # # convert results into pandas datastructure for indexing ease
#         # index_list_expanded = []
#         # tradeoff_list_expanded = []
#         # for ind, tradeoff_dict in enumerate(tradeoff_list):
#         #     for k, v in tradeoff_dict.items():
#         #         tradeoff_list_expanded.append(v)
#         #         index_list_expanded.append((k, index_list[ind][0],  index_list[ind][1]))
#         # experiment_multi_index = pd.MultiIndex.from_tuples(index_list_expanded, names=['func_name','model','perturbation'])        
#         # tradeoffs_pd = pd.Series(tradeoff_list_expanded, index=experiment_multi_index)
#         # if tradeoffs_pd_existing is not None :
#         #     tradeoffs_pd = pd.concat((tradeoffs_pd,tradeoffs_pd_existing))
        
#         # file = open(cache_path, 'wb')
#         # pickle.dump(tradeoffs_pd, file)
#         # file.close()

#     tradeoffs_pd = tradeoffs_pd_existing
#     print(tradeoffs_pd.index)
#     print('raw')
#     print(tradeoffs_pd)
# #     # save, for skipping next time

# else:
#     print(f'loading from {tradeoff_file_name}')
#     file_roca = open(tradeoff_file_name, 'rb')
#     tradeoff_funcs = pickle.load(file_roca)
#     file_roca.close()
#     print(f'loaded from {tradeoff_file_name}')

# Plot all tradeoffs AND generate heat map tables for them

# for (model_name, path) in tqdm(models_matched.itertuples(index=False)):

def compute_areas(model_name):
    file = open(f'{model_name}_trade_offs.pickle', 'rb')
    tradeoffs_pd = pickle.load(file)
    file.close()
    tradeoffs_pd = tradeoffs_pd[~tradeoffs_pd.index.duplicated()]

    nominal_pd_area_list = []
    perturb_pd_area_list = []
    perturb_naive_pd_area_list = []
    any_pd_area_list = []
    # tradeoffs_pd.groupby(['func_name'])
    # creat new tradeoff in tradeoffs_pd based on roc_a
    tradeoffs_roc_a = tradeoffs_pd['roc_a'][model_name]
    index_list = []
    object_list = []
    for (perturbation_name,),tradeoff_obj in tqdm(tradeoffs_roc_a.groupby(['perturbation']), leave=False, desc='adding map tradeoff'):
        # tradeoffs_pd.loc[('precision',model_name,perturbation_name)]
        object_list.append(tam.convert_roca_to_map(tradeoff_obj[perturbation_name]))
        index_list.append(('precision',model_name,perturbation_name))
    experiment_multi_index = pd.MultiIndex.from_tuples(index_list, names=['func_name','model','perturbation'])
    precision_tradeoff_pd = pd.Series(object_list, index=experiment_multi_index)
    tradeoffs_pd = pd.concat((tradeoffs_pd,precision_tradeoff_pd))
    for (tradeoff_name,),tradeoff_df in tqdm(tradeoffs_pd.groupby(['func_name']), leave=False, desc='plotting tradeoffs'):

        perturbations_all = tradeoff_df.index.get_level_values('perturbation')
        nominal_key="('no_op', '{}')"
        perturbations = set(perturbations_all)
        perturbations.remove( nominal_key )

        index_w_perturb_list = []
        index_list = []
        tradeoff_worst_list = []
        tradeoff_nominal_list = []
        tradeoff_any_list = []
    # for model, new_df in tqdm(tradeoffs_dict[tradeoff_name].groupby(level=[0]),desc='computing worst', leave=False):
        index_list.append((tradeoff_name,model_name))
        nominal_score_curve = tradeoffs_pd[tradeoff_name][model_name][nominal_key]
        # seems that we can have duplicates for nominal, must not have skipped them correctly when loading old
        # if isinstance(tradeoffs_pd[tradeoff_name][model_name][nominal_key], pd.Series):
        #     nominal_score_curve = nominal_score_curve[0]
        if swap_fp:
            if 'roc_a' in tradeoff_name or 'precision' in tradeoff_name:
                tradeoff_nominal_list.append(tam.convert_tradeoff_fp_goodness(nominal_score_curve,add_max_fp_obs=True))
            elif 'map' in tradeoff_name:
                tradeoff_nominal_list.append(nominal_score_curve)
            else:
                tradeoff_nominal_list.append(tam.convert_tradeoff_fp_goodness(nominal_score_curve,add_max_fp_obs=True,use_latent=True))
            tradeoff_any_list.append(tradeoff_nominal_list[-1])
        else:
            tradeoff_nominal_list.append(nominal_score_curve)
            tradeoff_any_list.append(tradeoff_nominal_list[-1])
        index_counter = 0
        # plot tradeoffs, per perturbation
        for key in tqdm(perturbations,desc=f"{model_name} perturbation",leave=False):
            index_w_perturb_list.append((tradeoff_name,model_name,key))
            if swap_fp:
                if 'roc_a' in tradeoff_name or 'precision' in tradeoff_name:
                    curve_perturb = tam.convert_tradeoff_fp_goodness(tradeoffs_pd[tradeoff_name][model_name][key],add_max_fp_obs=True)
                elif 'map' in tradeoff_name:
                    curve_perturb = tradeoffs_pd[tradeoff_name][model_name][key]
                else:
                    curve_perturb = tam.convert_tradeoff_fp_goodness(tradeoffs_pd[tradeoff_name][model_name][key],add_max_fp_obs=True,use_latent=True)
            else:
                curve_perturb = tradeoffs_pd[tradeoff_name][model_name][key]
            merged_curves = curve_perturb + tradeoff_nominal_list[-1]
            tradeoff_any_list[-1] = tradeoff_any_list[-1]  + curve_perturb
            tradeoff_worst_list.append(merged_curves)
            index_counter += 1 
            tradeoff_worst_list[-1].plot(f'plot_worst_{tradeoff_name}_{model_name}_{index_counter}.png',show_latent=True,show_worst=True)
        # plot tradeoffs with all perturbations
        tradeoff_any_list[-1].plot(f'plot_worst_{tradeoff_name}_{model_name}_all.png',show_latent=True,show_worst=True)

        # convert to pandas for ease of indexing
        index_w_perturb = pd.MultiIndex.from_tuples(index_w_perturb_list, names=['tradeoff','model','perturbation'])
        index= pd.MultiIndex.from_tuples(index_list, names=['tradeoff','model',])


        nominal_pd = pd.Series(tradeoff_nominal_list, index=index)
        any_pd = pd.Series(tradeoff_any_list, index=index)
        perturb_pd = pd.Series(tradeoff_worst_list, index=index_w_perturb)

        # use pandas sugar to get areas for heat maps
        if use_max and 'mota' in tradeoff_name or (not use_recall and ('hota' in tradeoff_name or 'idf1' in tradeoff_name )):
            nominal_pd_area_list.append(nominal_pd.apply(lambda tcs: tcs.get_max(0,min_thresh=0.1)[1]))
            perturb_pd_area_list.append(perturb_pd.apply(lambda tcs: tcs.worst_case().get_max(0,min_thresh=0.1)[1]))
            perturb_naive_pd_area_list.append(perturb_pd.apply(lambda tcs: tcs.get_max(0,min_thresh=0.1)[1]))
            any_pd_area_list.append(any_pd.apply(lambda tcs: tcs.worst_case().get_max(0,min_thresh=0.1)[1]))
        else:
            nominal_pd_area_list.append(nominal_pd.apply(lambda tcs: tcs.get_area(0)))
            perturb_pd_area_list.append(perturb_pd.apply(lambda tcs: tcs.worst_case().get_area(0)))
            perturb_naive_pd_area_list.append(perturb_pd.apply(lambda tcs: tcs.get_area(0)))
            any_pd_area_list.append(any_pd.apply(lambda tcs: tcs.worst_case().get_area(0)))

    nominal_pd_area = pd.concat(nominal_pd_area_list)
    perturb_pd_area = pd.concat(perturb_pd_area_list)
    perturb_naive_pd_area = pd.concat(perturb_naive_pd_area_list)
    any_pd_area = pd.concat(any_pd_area_list)

    return (nominal_pd_area,perturb_pd_area,perturb_naive_pd_area,any_pd_area)

def add_coco_map(in_pd:pd.DataFrame) -> pd.DataFrame:
    iou_threshes = np.arange(0.5, 0.955, 0.05)
    map_names = [f'map_{iou_thresh:.2f}' for iou_thresh in iou_threshes]
    areas_coco_map = []
    index_coco_map = []
    for (model_name,),subframe in tqdm(in_pd.groupby(['model']), leave=False, desc='adding map tradeoff to each model'):
        areas_coco_map.append(subframe[subframe.index.get_level_values('tradeoff').str.contains('|'.join(map_names), na=False)].mean())
        index_coco_map.append(('map_coco',model_name))
    index= pd.MultiIndex.from_tuples(index_coco_map, names=['tradeoff','model',])
    coco_map_pd = pd.Series(areas_coco_map, index=index)
    return pd.concat((coco_map_pd, in_pd))


if create_plots_heatmaps:
# most of the work is here, so it has a parallel mode. Single process mode is special case, for debug
    if num_process == 1:
        tradeoff_list = list(tqdm(map(compute_areas, models_matched['model'].to_list()),desc='computing areas', leave=False, total=len(models_matched)))
    else:
        with Pool(num_process) as p:
            tradeoff_list = list(tqdm(p.imap(compute_areas, models_matched['model'].to_list()),desc='computing areas', leave=False, total=len(models_matched)))

    nominal_pd_area_list = []
    perturb_pd_area_list = []
    perturb_naive_pd_area_list = []
    any_pd_area_list = []
    for (nominal_pd_area,perturb_pd_area,perturb_naive_pd_area,any_pd_area) in tradeoff_list:
        nominal_pd_area_list.append(nominal_pd_area)
        perturb_pd_area_list.append(perturb_pd_area)
        perturb_naive_pd_area_list.append(perturb_naive_pd_area)
        any_pd_area_list.append(any_pd_area)

    nominal_pd_area = pd.concat(nominal_pd_area_list)
    perturb_pd_area = pd.concat(perturb_pd_area_list)
    perturb_naive_pd_area = pd.concat(perturb_naive_pd_area_list)
    any_pd_area = pd.concat(any_pd_area_list)

    # add coco map
    any_pd_area = add_coco_map(any_pd_area)
    nominal_pd_area = add_coco_map(nominal_pd_area)


    area_file_name = 'areas.pickle'
    file_roca = open(area_file_name, 'wb')
    pickle.dump({'nominal_pd_area':nominal_pd_area, 'perturb_pd_area':perturb_pd_area, 'any_pd_area':any_pd_area}, file_roca)
    file_roca.close()
    print(f'saved to {area_file_name}')

    for tradeoff_name in tqdm(nominal_pd_area.index.unique('tradeoff'), leave=False, desc='creating heatmaps'):

        perturb_naive_pd_area_df = perturb_naive_pd_area[tradeoff_name].unstack(level='perturbation')
        perturb_pd_area_df=perturb_pd_area[tradeoff_name].unstack(level='perturbation')

        nominal_pd_area_np = nominal_pd_area[tradeoff_name].to_numpy().reshape((1,-1))
        any_pd_area_np = any_pd_area[tradeoff_name].to_numpy().reshape((1,-1))

        perturb_pd_area_np = np.concatenate((any_pd_area_np,perturb_pd_area_df.to_numpy().transpose()),axis=0)
        perturb_naive_pd_area_np = np.concatenate((np.full_like(any_pd_area_np,np.nan),perturb_naive_pd_area_df.to_numpy().transpose()),axis=0)

        perturbations_strings = ['any']+perturb_pd_area_df.columns.tolist()
        model_strings = list(nominal_pd_area[tradeoff_name].index)
        perturbations_strings_with_nominal = ['any', *perturb_pd_area_df.columns.tolist(), 'nominal']
        # heatmap table, title, colormap maximum (-1 computes max on data)
        heatmap_settings = ((perturb_pd_area_np,'raw-worst-case',-1),
            (perturb_naive_pd_area_np,'raw-naive',-1),
            (perturb_pd_area_np/nominal_pd_area_np,'ratio-worst-case',1.2),
            (perturb_naive_pd_area_np/nominal_pd_area_np,'ratio-naive',1.2))
            # generate heatmaps
        for perturb_np,name,max_val in heatmap_settings:
            fig, ax_dict = plt.subplot_mosaic([['A panel'],['B panel']], sharex=True,
                                                height_ratios=[19,1],
                                                width_ratios=[1])
            ax_pert = ax_dict['A panel']
            ax_nom = ax_dict['B panel']
            # fig = plt.figure()
            # gs = fig.add_gridspec(2, 1, height_ratios=(19, 1))
            # # Create the Axes.
            # ax_pert = fig.add_subplot(gs[0, 0])
            # ax_nom = fig.add_subplot(gs[1, 0], sharex=ax_pert)
            
            save_df = pd.DataFrame(np.concatenate((perturb_np, nominal_pd_area_np),axis=0), columns = model_strings, index=perturbations_strings_with_nominal)
            save_df.to_csv(f'table_{tradeoff_name}_{name}.csv')

            if max_val < 0:
                max_val = max(np.max(np.nan_to_num(perturb_np)),np.max(np.nan_to_num(nominal_pd_area_np)))

            min_val = min(min(np.min(np.nan_to_num(perturb_np)),np.min(np.nan_to_num(nominal_pd_area_np))),0)
            
            im_pert = ax_nom.imshow(nominal_pd_area_np,vmin=min_val, vmax=max_val,aspect='auto')
            im_nom = ax_pert.imshow(perturb_np,vmin=min_val, vmax=max_val,aspect='auto')
            

            ax_nom.set_xticks(np.arange(len(model_strings)),labels=model_strings)
            ax_pert.set_yticks(np.arange(len(perturbations_strings)),labels=perturbations_strings)
            ax_nom.set_yticks((0,),labels=['nominal'])

            plt.setp(ax_nom.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            for model_ind in range(len(model_strings)):
                # if nominal_pd_area_np[0, model_ind]/max_val < 0.5:
                #     c = 'w'
                # else:
                #     c = 'k'
                c = 'k'
                text = ax_nom.text(model_ind, 0, round(nominal_pd_area_np[0, model_ind],2),
                            ha="center", va="center", color=c, fontsize='small')
                for pert_ind in range(len(perturbations_strings)):
                    # if perturb_np[pert_ind, model_ind]/max_val < 0.5:
                    #     c = 'w'
                    # else:
                    #     c = 'k'
                    c = 'k'
                    text = ax_pert.text(model_ind, pert_ind, round(perturb_np[pert_ind, model_ind],2),
                                ha="center", va="center", color=c , fontsize='small')
            fig.suptitle(name)
            #fig.subplots_adjust(right=0.8)
            #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            #fig.colorbar(im_nom, cax=cbar_ax)
            fig.colorbar(im_nom,  ax=(ax_pert,ax_nom), shrink=0.8)
            # ,bbox_inches='tight'
            plt.savefig(f'table_{tradeoff_name}_{name}.png', dpi=300,bbox_inches='tight')
            plt.clf()
            plt.close()

else:
    area_file_name = 'areas.pickle'
    file_roca = open(area_file_name, 'rb')
    areas_dict = pickle.load(file_roca)
    file_roca.close()
    nominal_pd_area, perturb_pd_area, any_pd_area = areas_dict['nominal_pd_area'], areas_dict['perturb_pd_area'], areas_dict['any_pd_area']



metric_types = ['modehota', 'modemota','modeidf1']
to_consider = ['gt_id_metric', 'roc_a_metric','precision_metric','map_50','map_coco']
to_predict = ['trackersortfp_modefpr', 'trackerbytefp_modefpr', 'trackerpkffp_modefpr', 'trackeroc-sortfp_modefpr']
marker = itertools.cycle(('P', 'D', '*', 'v')) 

# for tradeoff_name in tqdm(nominal_pd_area.index.unique('tradeoff'), leave=False):
# model_strings = list(nominal_pd_area[tradeoff_name].index)
# outer loop over metric_types
for ind, pd_relevant in enumerate([nominal_pd_area,any_pd_area]):
    pd_roca = pd_relevant[[tradeoff for tradeoff in nominal_pd_area.index.unique('tradeoff') if ('roc_a' in tradeoff) and not ('roc_a_N' in tradeoff)]].reset_index(name='roc_a_metric').drop(axis=1,labels='tradeoff')
    pd_prec = pd_relevant[[tradeoff for tradeoff in nominal_pd_area.index.unique('tradeoff') if ('precision' in tradeoff)]].reset_index(name='precision_metric').drop(axis=1,labels='tradeoff')
    pd_map_5 = pd_relevant[[tradeoff for tradeoff in nominal_pd_area.index.unique('tradeoff') if ('map_0.50' in tradeoff)]].reset_index(name='map_50').drop(axis=1,labels='tradeoff')
    pd_map_coco = pd_relevant[[tradeoff for tradeoff in nominal_pd_area.index.unique('tradeoff') if ('map_coco' in tradeoff)]].reset_index(name='map_coco').drop(axis=1,labels='tradeoff')

    for metric_type in metric_types:

        pd_reduced = pd_relevant[[tradeoff for tradeoff in pd_relevant.index.unique('tradeoff') if metric_type in tradeoff]]
        pd_gt_id = pd_reduced[[tradeoff for tradeoff in pd_reduced.index.unique('tradeoff') if 'trackergt-idfp_modefpr' in tradeoff]].reset_index(name='gt_id_metric').drop(axis=1,labels='tradeoff')
        
        pd_tracker_metrics = pd_reduced[[tradeoff for tradeoff in pd_reduced.index.unique('tradeoff') if any(tracker_type in tradeoff for tracker_type in to_predict)]].reset_index(name='tracker_metric')
        pd_tracker_metrics['tracker'] = pd_tracker_metrics['tradeoff'].str.extract('-tracker(\S+)fp_mode')
        for metric_pd in [pd_gt_id,pd_roca,pd_prec,pd_map_5,pd_map_coco]:
            pd_tracker_metrics = pd.merge(pd_tracker_metrics,metric_pd,on='model')
        pd_tracker_metrics.info()
        trackers = pd.unique(pd_tracker_metrics.tracker)
        for independent_var in to_consider:
            model = ols(f"tracker_metric ~ tracker + {independent_var} + tracker:{independent_var}",data=pd_tracker_metrics).fit()
            aov_table = sm.stats.anova_lm(model, typ=3)
            print(f'{independent_var} ignore model {metric_type} {ind}')
            print(aov_table)
            print(f'R^2 {aov_table.sum_sq[independent_var]/aov_table.sum(0)["sum_sq"]}')
            print()

            model = ols(f"tracker_metric ~ tracker + {independent_var}",data=pd_tracker_metrics).fit()
            aov_table = sm.stats.anova_lm(model, typ=3)
            print(f'{independent_var} ignore model and tracker interaction {metric_type} {ind}')
            print(aov_table)
            print(f'R^2 {aov_table.sum_sq[independent_var]/aov_table.sum(0)["sum_sq"]}')
            print()

            model = ols(f"tracker_metric ~ {independent_var}",data=pd_tracker_metrics).fit()
            aov_table = sm.stats.anova_lm(model, typ=3)
            print(f'{independent_var} ignore model and tracker {metric_type} {ind}')
            print(aov_table)
            print(f'R^2 {aov_table.sum_sq[independent_var]/aov_table.sum(0)["sum_sq"]}')
            print()
            

            # model = ols(f"tracker_metric ~ tracker + {independent_var} + model + tracker:{independent_var}  + model:{independent_var} + tracker:model + tracker:model:{independent_var}",data=pd_tracker_metrics).fit()
            # aov_table = sm.stats.anova_lm(model, typ=3)
            # print(f'{independent_var} model and tracker {metric_type} {ind}')
            # print(aov_table)

            fig, ax = plt.subplots()
            for tracker in trackers:
                pd_filt = pd_tracker_metrics[pd_tracker_metrics.tracker == tracker]
                ax.plot(pd_filt[independent_var], pd_filt.tracker_metric, marker = next(marker), linestyle='')
                ax.set_xlabel(f'{independent_var} ind')
                ax.set_ylabel(f'Area under average {metric_type} {ind}')
            lgd = fig.legend(trackers)
            path = f'metric_plot_{metric_type}_{independent_var}_{ind}'
            fig.savefig(path, dpi=300,bbox_extra_artists=(lgd, ),bbox_inches='tight')
            fig.clf()

