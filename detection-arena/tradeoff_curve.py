# CATE Detection Evaluation code
 
# Copyright 2025 Carnegie Mellon University.
 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
 
# Licensed under an Creative Commons Attribution-Non-Commercial 4.0 International (CC BY-NC 4.0)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
 
# DM25-0275

import numpy as np
import matplotlib.pylab as plt # type: ignore
import pandas as pd # type: ignore
from typing import Optional, Union, Sequence, List
import numpy.typing as npt
import scipy
import os

import scipy.ndimage



class ScoreCurves:
    """ Collection of score curves for eval or display
    
    This object allows the user to manage a collection of general
    2D score curves. Score curves are represented as a list
    of pandas dataframes with fields of 'xs', 'ys', and 'latents'
    For plotting, or for ease of access, each curve can be given a 
    name. 

    Attributes:
        curves (List[pd.DataFrame]): have fields of 'xs','ys','latents'
        x_name (str): string to use when plotting x, defult: ''
        y_name (str): string to use when plotting y, defult: ''
        x_is_reference (bool): monotonic axis for area-under-curve, x if True, y if False
        x_is_good (bool): whether up or down is better for this score
        y_is_good (bool): whether up or down is better for this score
        latent_name (str): string to use when plotting latent, defult: 'confidence'
        curve_names: Optional[Sequence[str]] strings to label each of the curves
              used in plotting and can also be used to index particular curves
        min_x: float offset to apply to x when taking area under the curve, default: 0 
        min_y: float offset to apply to y when taking area under the curve, default: 0
    """
    def __init__(self, xs: Sequence[npt.NDArray], 
                 ys: Sequence[npt.NDArray], 
                 latents: Sequence[npt.NDArray], 
                 x_is_reference: bool,
                 x_is_good: bool=True, y_is_good:bool=True,
                 x_name: str='',y_name: str= '',
                 latent_name: str='confidence',
                 curve_names: Optional[Sequence[str]]=None,
                 min_x: float = 0, min_y: float = 0)->None:
        if len(xs) != len(ys) or len(xs) != len(latents):
            raise RuntimeError(f"Different curve counts between xs: {len(xs)}, ys: {len(ys)} and latents: {len(latents)}")

        if (curve_names is not None) and (len(xs)!=len(curve_names)):
            raise RuntimeError(f"Different curve counts between xs: {len(xs)}, and names: {len(curve_names)}")

        self.x_is_reference = x_is_reference
        self.x_is_good = x_is_good
        self.y_is_good = y_is_good
        self.curves: List[pd.DataFrame]=[]
        self.curve_names = curve_names
        if curve_names is not None:
            self.curve_names = list(curve_names)

        self.x_name = x_name
        self.y_name = y_name
        self.latent_name = latent_name

        self.min_x=min_x
        self.min_y=min_y

        for i in range(len(xs)):

            if xs[i].shape != ys[i].shape or xs[i].shape != latents[i].shape:
                raise RuntimeError(f"Different element counts in curve {i} xs: {xs[i].shape}, ys: {ys[i].shape} and latents: {latents[i].shape}")
            sort_indices = np.argsort(latents[i])
            latents_sorted = pd.Series(latents[i][sort_indices],name='latents')
            xs_sorted = pd.Series(xs[i][sort_indices],name='xs')
            ys_sorted = pd.Series(ys[i][sort_indices],name='ys')
            df = pd.concat((xs_sorted,ys_sorted,latents_sorted),axis=1)
            self.curves.append(df)
    def __add__(self, o:'ScoreCurves'):
        if bool(self.curve_names) is not bool(o.curve_names):
            raise RuntimeError(
                    "One of the ScoreCurves objects has optional curve_names, but the other does not")
        elif self.curve_names and o.curve_names:
            overlapping_curves = set(self.curve_names).intersection(set(o.curve_names))
            if overlapping_curves:
                raise RuntimeError(
                        f"These keys are in common between Score curves: {overlapping_curves}")
            
        if (self.min_x != o.min_x) | (self.min_y != o.min_y):
            raise RuntimeError(
                    "These ScoreCurves use different minimum values and cannot be appended")
        if (self.x_is_good != o.x_is_good) | (self.y_is_good != o.y_is_good):
            raise RuntimeError(
                    "These ScoreCurves use different score definitions and cannot be appended")
        xs = []
        ys = []
        latents = []
        curve_names = []
        for curve in (self, o):
            for key_element in range(len(curve)):
                xs += [curve[key_element].curves[0]['xs']]
                ys += [curve[key_element].curves[0]['ys']]
                latents += [curve[key_element].curves[0]['latents']]
                if curve.curve_names is not None:
                    curve_names += curve[key_element].curve_names

        return ScoreCurves(xs=xs,ys=ys,latents=latents, x_is_reference=self.x_is_reference,
                        latent_name=self.latent_name, curve_names=curve_names,
                        x_is_good=self.x_is_good, y_is_good=self.y_is_good,
                        min_x=self.min_x,min_y=self.min_y,x_name=self.x_name,y_name=self.y_name)
    
    def get_area(self, key: Union[slice,int,str,Sequence]):
        tc_minimal = self[key]
        if len(tc_minimal) != 1:
            raise RuntimeError("Can only use to get one area at a time")
        ys=tc_minimal.curves[0]['ys'].to_numpy()
        xs=tc_minimal.curves[0]['xs'].to_numpy()
        return self.area_under_curve(xs,ys,self.min_x,self.min_y,self.x_is_reference)

    def get_max(self, key:  Union[slice,int,str,Sequence], axis='y', min_thresh=0):
        tc_minimal = self[key]
        if len(tc_minimal) != 1:
            raise RuntimeError("Can only use to get one area at a time")
        latents = tc_minimal.curves[0]['latents'].to_numpy()

        filter = latents > min_thresh
        latents = latents[filter]
        ys=tc_minimal.curves[0]['ys'].to_numpy()[filter]
        xs=tc_minimal.curves[0]['xs'].to_numpy()[filter]
        
        
        if axis == 'x':
            ind = np.argmax(xs)
        if axis == 'y':
            ind = np.argmax(ys)
        if axis == 'latent':
            ind = np.argmax(latents)
        return (xs[ind], ys[ind], latents[ind])
            

    @staticmethod
    def area_under_curve(x: npt.NDArray,y: npt.NDArray,min_x: float,min_y: float,x_is_reference: bool) -> float:
        """Returns float area under the curve.
        
        Computes area for single Score curve. Integrates between max in data and specified min value.
        
        Arguments:
            x:     numpy array for one score
            y:     numpy array for other score
            min_x: min allowed value for first Score curve
            min_y: min allowed value for second Score curve
                       
        Returns:
            float area under curve"""
        x_offset = x - min_x
        y_offset = y - min_y
        if np.all(x_offset==0) or np.all(y_offset==0):
            return 0.

        if not x_is_reference:
            reference, score, min_reference, min_score = y, x, min_y*np.ones([1]), min_x*np.ones([1])
        else:
            reference, score, min_reference, min_score = x, y, min_x*np.ones([1]), min_y*np.ones([1])


        # # trim leading/trailing zeros
        # increasing_is_not_min = reference != min_reference
        # decreasing_is_not_min = score != min_score
        # inds = [decreasing_is_not_min.argmax()-1, increasing_is_not_min.size - increasing_is_not_min[::-1].argmax()]
        # if inds[0] == -1:
        #     inds[0] =0 
        # reference = reference[inds[0]:inds[1]]
        # score = score[inds[0]:inds[1]]

        # reference = np.concatenate((min_reference, reference))
        # score = np.concatenate((score[0:1], score))

        #return np.trapz(score-min_score,reference)
        #return np.trapz(np.minimum.accumulate(decreasing)-min_decreasing,np.minimum.accumulate(increasing[::-1])[::-1])
        if np.trapz(score-min_score,reference) == 0:
            print('WARNING: zero area')

        if np.all(np.diff(reference) >= -np.finfo(float).eps):
            return np.trapz(score-min_score,reference)
        elif np.all(np.diff(reference) <= np.finfo(float).eps):
            return -np.trapz(score-min_score,reference)
        else:
            print('WARNING: non-monotonic reference')
            return 0

    def __len__(self):
        """Returns int of count of stored Score curves."""
        return len(self.curves)
    
    @staticmethod
    def _expand_and_filter_vec(vector, is_good,start_value,end_value):

        if is_good is None:
            return np.repeat(vector, 3)

        if is_good:
            pad_val = float('inf')
        else:
            pad_val = float('-inf')
        vector_worst = np.full((vector.size*3,),pad_val)
        vector_worst[1:-1:3] = vector

        if is_good:
            vector_worst = scipy.ndimage.minimum_filter1d(vector_worst,5)
        else:
            vector_worst = scipy.ndimage.maximum_filter1d(vector_worst,5)
        vector_worst[0] = start_value
        vector_worst[-1] = end_value
        return vector_worst

    def _interpolate_xylatent(self, xs, ys, latents):
        xs_interp = self._expand_and_filter_vec(xs, self.x_is_good, self.min_x, self.min_x)[1:-1]
        ys_interp = self._expand_and_filter_vec(ys, self.y_is_good, self.min_y, self.min_y)[1:-1]
        latents_interp = np.repeat(latents,3)[1:-1]
        return xs_interp, ys_interp, latents_interp

    def plot(self, path: Union[str,os.PathLike], show_points: bool =False, 
             show_worst: bool =False, show_latent: bool =False, title: str = ''):
        """Saves plots of Score curves.
        
        Saves matplotlib plot of Score curve to location in "path"
        
        Arguments:
            path:        file path passed to savefig, controls file type
            title:       string to use as title for figure, default: ''
            show_points: boolean whether points along latent range are drawn, default: False
            show_worst:  boolean whether worst case curve should also be shown, default: False
            show_latent: boolean whether latent vs Score values should also be shown, default: False
                       
        Returns:
            float area under curve"""
        
        marker_order = ['v','1', '*','^','2','+']

        xs, ys, latents = [],[],[]
        for index in range(len(self.curves)):
            x, y, latent = self._interpolate_xylatent(self.curves[index]['xs'].to_numpy(),self.curves[index]['ys'].to_numpy(),self.curves[index]['latents'].to_numpy())
            xs.append(x)
            ys.append(y)
            latents.append(latent)

        if self.curve_names is not None:
            curve_names = list(self.curve_names)
        if show_worst or show_points:
            worst = self.worst_case()
        if show_points:
            latent_full = worst.curves[0]['latents'].to_numpy()
            # equally spaced from smallest to largest latent
            latent_points = np.linspace(latent_full[0],latent_full[-1],6,endpoint=True)
            # alternative, equal number of detections: np.quantile(latent_full,[0,0.2,0.4,0.6,0.8,1])
        if show_worst:
            x, y, latent = self._interpolate_xylatent(worst.curves[0]['xs'].to_numpy(),worst.curves[0]['ys'].to_numpy(),worst.curves[0]['latents'].to_numpy())
            xs.append(x)
            ys.append(y)
            latents.append(latent)
            if curve_names is not None:
                curve_names += ['worst case']

        if show_latent:
            fig = plt.figure(constrained_layout=True)
            axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                                    gridspec_kw={'width_ratios':[2, 1]})
            axs['Left'].set_title(title)
            axs['Left'].set_xlabel(self.x_name)
            axs['Left'].set_ylabel(self.y_name)
            axs['TopRight'].set_title(self.y_name)
            axs['BottomRight'].set_title(self.x_name)
            axs['BottomRight'].set_xlabel(self.latent_name)
            if not self.x_is_good and self.x_is_good is not None:
                axs['Left'].set_xscale('log')
                axs['BottomRight'].set_yscale('log')
            
            if not self.y_is_good:
                axs['Left'].set_yscale('log')
                axs['TopRight'].set_yscale('log')

            for curve_ind in range(len(xs)):
                if show_worst and (curve_ind == (len(xs) - 1)):
                    linestyle = '--'
                else:
                    linestyle = '-'
                color = next(axs['Left']._get_lines.prop_cycler)['color']
                axs['Left'].plot(xs[curve_ind],ys[curve_ind],linestyle=linestyle, color = color)
                axs['TopRight'].plot(latents[curve_ind],ys[curve_ind],linestyle=linestyle, color = color)
                axs['BottomRight'].plot(latents[curve_ind],xs[curve_ind],linestyle=linestyle, color = color)
                area = self.area_under_curve(xs[curve_ind],ys[curve_ind],self.min_x,self.min_y,self.x_is_reference)
                if show_points:
                    for marker_ind in range(len(marker_order)):
                        x = np.interp(latent_points[marker_ind], latents[curve_ind], xs[curve_ind])
                        y = np.interp(latent_points[marker_ind], latents[curve_ind], ys[curve_ind])

                        axs['Left'].plot(x,y,linestyle='-', color = color,marker=marker_order[marker_ind],label='_nolegend_')
                        axs['TopRight'].plot(latent_points[marker_ind],y,linestyle='-',marker=marker_order[marker_ind], color = color,label='_nolegend_')
                        axs['BottomRight'].plot(latent_points[marker_ind],x,linestyle='-',marker=marker_order[marker_ind], color = color,label='_nolegend_')
                curve_names[curve_ind] = f'{curve_names[curve_ind]} {area}'

        
        else:
            for curve_ind in range(len(self)):
                if show_worst and (curve_ind == (len(xs) - 1)):
                    linestyle = '--'
                else:
                    linestyle = '-'
                # color = next(plt._get_lines.prop_cycler)['color']
                line, = plt.plot(xs[curve_ind],ys[curve_ind],linestyle=linestyle)
                area = self.get_area(curve_ind)
                curve_names[curve_ind] = f'{curve_names[curve_ind]} {area}'
                if show_points:
                    for marker_ind in range(len(marker_order)):
                        x = np.interp(latent_points[marker_ind], latents[curve_ind], xs[curve_ind])
                        y = np.interp(latent_points[marker_ind], latents[curve_ind], ys[curve_ind])
              
                        plt.plot(x,y, color = line.get_color(),marker=marker_order[marker_ind],label='_nolegend_')
            plt.xlabel(self.x_name)
            plt.ylabel(self.y_name)
            plt.title(title)
   
        if self.curve_names is not None:
            if show_latent:
                lgd = fig.legend(curve_names,prop={'size': 6},loc='upper center',bbox_to_anchor=(0.5,0.0))
            else:
                lgd = plt.legend(curve_names,prop={'size': 6},loc='upper center',bbox_to_anchor=(0.5,0.0))

        if path is not None:
            plt.savefig(path, dpi=300,bbox_extra_artists=(lgd, ),bbox_inches='tight')
        plt.clf()
        plt.close()

    def __getitem__(self, key: Union[slice,int,str,Sequence]):
        if isinstance(key, slice):
            curves = self.curves[key]
            xs = [curve['xs'].to_numpy() for curve in curves]
            ys = [curve['ys'].to_numpy() for curve in curves]
            latents = [curve['latents'].to_numpy() for curve in curves]
            if self.curve_names is not None:
                curve_names = self.curve_names[key]
            else:
                curve_names = None
        elif isinstance(key, int):
            curve = self.curves[key]
            xs = [curve['xs'].to_numpy()]
            ys = [curve['ys'].to_numpy()]
            latents = [curve['latents'].to_numpy()]
            if self.curve_names is not None:
                curve_names = [self.curve_names[key]]
            else:
                curve_names = None
        elif isinstance(key, str):
            if self.curve_names is not None:
                return self[self.curve_names.index(key)]
            else:
                raise TypeError("Indexing by string is invalid when curve_names is not set.")
        elif hasattr(key, '__iter__'):
            xs = []
            ys = []
            latents = []
            if self.curve_names is not None:
                curve_names = []
            else:
                curve_names = None
            for key_element in key:
                xs += [self[key_element].curves[0]['xs']]
                ys += [self[key_element].curves[0]['ys']]
                latents += [self[key_element].curves[0]['latents']]
                if self.curve_names is not None:
                    curve_names += self[key_element].curve_names
        else:
            raise TypeError("Invalid argument type.")
        return ScoreCurves(xs=xs,ys=ys,latents=latents,x_is_reference=self.x_is_reference,
                        latent_name=self.latent_name, curve_names=curve_names,
                        min_x=self.min_x,min_y=self.min_y)

    def worst_case(self) -> 'ScoreCurves':
        """Returns new ScoreCurves containing the worst-case combination of self curves 
        
        For all latents available in all Score curves, finds the worst value for each of xs
        and ys, then assembles a ScoreCurves object with these values. Carries over all other
        parameters from self.
        
        Arguments:
            ScoreCurves object: collection of curves to use for worst case
                       
        Returns:
            ScoreCurves object: single Score curve representing worst case, for computing robustness
        """

        x_worst = self.curves[0]['xs'].to_numpy()
        y_worst = self.curves[0]['ys'].to_numpy()
        latent_worst = self.curves[0]['latents'].to_numpy()

        min_latent = np.min(latent_worst)
        max_latent = np.max(latent_worst)

        if self.x_is_reference: 
            reference = x_worst
            min_x = None
            min_y = self.min_y
        else: 
            reference = y_worst
            min_x = self.min_x
            min_y = None

        for curve_ind in range(1, len(self)):
            latent_other = self.curves[curve_ind]['latents'].to_numpy()
            min_latent = max(np.min(latent_other),min_latent)
            max_latent = min(np.max(latent_other),max_latent)
            x_other = self.curves[curve_ind]['xs'].to_numpy()
            y_other = self.curves[curve_ind]['ys'].to_numpy()
            indices_next_in_worst = np.searchsorted(latent_worst, latent_other,side='right') + 1
            indices_last_in_worst = np.searchsorted(latent_worst, latent_other,side='left')
            indices_next_in_other = np.searchsorted(latent_other, latent_worst,side='right') + 1
            indices_last_in_other = np.searchsorted(latent_other, latent_worst,side='left') 

            latent_cat = np.concatenate((latent_worst, latent_other))
            _, sort_indices = np.unique(latent_cat, return_index=True) # not pessimistic TODO
            def worst_cat(vec0_0, vec0_1, vec1_0, vec1_1, is_good, indices):
                cat_1 = np.concatenate((vec0_0, vec0_1))[indices]
                cat_2 = np.concatenate((vec1_0, vec1_1))[indices]
                return worst(cat_1,cat_2,is_good)
            
            def worst(vec_1, vec_2, is_good):
                if is_good is None:
                    return vec_1
                if is_good:
                    return np.minimum(vec_1, vec_2)
                else:
                    return np.maximum(vec_1, vec_2)
            
            def index_search_sorted(vec, ind, is_good, min_val=None,max_val=None):
                # TODO make special case for known monotonic
                if min_val is None:
                    if is_good:
                        min_val = -float('inf')
                    else:
                        min_val = -float('inf')
                if max_val is None:
                    if is_good:
                        max_val = -float('inf')
                    else:
                        max_val = float('inf')

                if is_good:
                    bad_val = min_val
                else:
                    bad_val = max_val
                # inf_array = min_val * np.ones([1])
                # used to be inf_array on both sides
                # if monotonic, could look like this
                # vec_pad = np.concatenate(( np.ones([1]) * vec[0], vec, np.ones([1]) * vec[-1]))
                vec_pad = np.concatenate(( np.ones([1]) * bad_val, vec, np.ones([1]) * bad_val))
                return vec_pad[ind]
            
            max_x = float('nan')
            max_y = float('nan')

            x_cat_next = worst_cat(x_worst, index_search_sorted(x_worst,indices_next_in_worst,self.x_is_good,min_x,max_x),
                            index_search_sorted(x_other,indices_next_in_other, self.x_is_good, min_x,max_x), x_other, self.x_is_good,sort_indices,)
            x_cat_last = worst_cat(x_worst, index_search_sorted(x_worst,indices_last_in_worst,self.x_is_good, min_x,max_x),
                            index_search_sorted(x_other,indices_last_in_other, self.x_is_good, min_x,max_x), x_other, self.x_is_good,sort_indices,)
            
            y_cat_next = worst_cat(y_worst, index_search_sorted(y_worst,indices_next_in_worst,self.y_is_good, min_y,max_y),
                            index_search_sorted(y_other,indices_next_in_other,self.y_is_good, min_y,max_y), y_other, self.y_is_good,sort_indices)
            y_cat_last = worst_cat(y_worst, index_search_sorted(y_worst,indices_last_in_worst,self.y_is_good, min_y,max_y),
                            index_search_sorted(y_other,indices_last_in_other,self.y_is_good, min_y,max_y), y_other, self.y_is_good,sort_indices)
            x_worst = worst(x_cat_next,x_cat_last,self.x_is_good)
            y_worst = worst(y_cat_next,y_cat_last,self.y_is_good)
            latent_worst = latent_cat[sort_indices]

            keep = np.logical_and(np.logical_and(latent_worst >= min_latent,latent_worst <= max_latent),np.logical_and(np.isfinite(x_worst),np.isfinite(y_worst)))
            x_worst = x_worst[keep]
            y_worst = y_worst[keep]
            latent_worst = latent_worst[keep]

        if self.x_is_reference:
            reference = x_worst
        else:
            reference = y_worst

        if np.all(np.diff(reference) >= -np.finfo(float).eps):
            pass
        elif np.all(np.diff(reference) <= np.finfo(float).eps):
            pass
        else:
            print('WARNING: non-monotonic reference')
        if self.x_is_good is None:
            x_worst = latent_worst
        return ScoreCurves(xs=[x_worst], ys=[y_worst], 
                              latents=[latent_worst], x_is_reference=self.x_is_reference,
                              x_is_good=self.x_is_good,y_is_good=self.y_is_good,
                              x_name=self.x_name, y_name=self.y_name,
                              min_x=self.min_x, min_y=self.min_y, latent_name=self.latent_name)


class TradeoffCurves(ScoreCurves):
    """ Collection of tradeoff curves for eval or display
    
    Convenience class for score curves that represent "tradeoffs", special case
    ScoreCurves where two measures are 1. independent and 2. move in opposite directions

    Attributes:
        curves (List[pd.DataFrame]): have fields of 'xs','ys','latents'
        x_name (str): string to use when plotting x, defult: ''
        y_name (str): string to use when plotting y, defult: ''
        latent_name (str): string to use when plotting latent, defult: 'confidence'
        curve_names: Optional[Sequence[str]] strings to label each of the curves
              used in plotting and can also be used to index particular curves
        min_x: float offset to apply to x when taking area under the curve, default: 0 
        min_y: float offset to apply to y when taking area under the curve, default: 0
    """
    def __init__(self, xs: Sequence[npt.NDArray], 
                 ys: Sequence[npt.NDArray], 
                 latents: Sequence[npt.NDArray], 
                 x_name: str='',y_name: str= '',
                 latent_name: str='confidence',
                 curve_names: Optional[Sequence[str]]=None,
                 min_x: float = 0, min_y: float = 0)->None:
        ScoreCurves.__init__(self, xs, ys, latents, x_is_reference=not self.determine_if_y_increasing(xs[0],ys[0]),
                             x_is_good=True,y_is_good=True,
                             x_name=x_name, y_name=y_name,
                             latent_name=latent_name,
                             curve_names=curve_names,
                             min_x=min_x, min_y=min_y)
        
    @staticmethod
    def determine_if_y_increasing(xs,ys) -> bool:
        if np.argmax(ys[::-1]) < ys.size/2 :
            return True
        else:
            return False


        