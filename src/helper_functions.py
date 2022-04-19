import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import talib

import os, sys

sys.path.append(os.path.abspath(os.path.join('')))
from generative_models import *
from fast_dm_simulator import *


#  Returns tuple of handles, labels for axis ax, after reordering them to conform to the label order `order`, and if unique is True, after removing entries with duplicate labels.
def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels,
              fontsize=16, loc='upper right')
    return(handles, labels)


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]

def pr_check(emp_data, post_d_samples, n_sim, sma_period=5):
    # get experimental context
    context = emp_data.stim_type.values - 1
    # get empirical response times
    emp_rt = np.abs(emp_data.rt.values)
    sma_emp_rt = talib.SMA(emp_rt, timeperiod=sma_period)

    n_obs = emp_rt.shape[0]
    pred_rt = np.zeros((n_sim, n_obs))
    sma_pred_rt = np.zeros((n_sim, n_obs))
    # iterate over number of simulations
    for sim in range(n_sim):
        # Iterate over number of trials
        rt = np.zeros(n_obs)
        for t in range(n_obs):
            # Run diffusion process
            rt[t] = diffusion_trial(post_d_samples[sim, t, context[t]], post_d_samples[sim, t, 4], post_d_samples[sim, t, 5])
        pred_rt[sim] = np.abs(rt)
        sma_pred_rt[sim] = talib.SMA(np.abs(rt), timeperiod=sma_period)

    return pred_rt, sma_pred_rt

def plot_pred_rt(person_data, pred_data_dynamic, pred_data_sma, pred_data_fast_dm):

    # summarize predicted response times
    quantiles = np.quantile(pred_data_sma, [0.05, 0.95], axis=0)
    median = np.median(pred_data_sma, axis=0)

    # prepare emp rt data
    emp_rt = np.abs(person_data.rt.values)
    n_obs = emp_rt.shape[0]
    emp_data_sma = talib.SMA(emp_rt, timeperiod=5)

    # initialize figure
    f, ax = plt.subplots(1, 2, figsize=(18, 8),
                        gridspec_kw={'width_ratios': [6, 1]})

    axrr = ax.flat

    # plot empiric and predicted response times series
    time = np.arange(n_obs) 
    axrr[0].plot(time, emp_data_sma, color='black', lw=1.5, label='SMA5: Empiric')
    axrr[0].plot(time, median, color='#598f70', lw=1.5, label='SMA5: Predicted median', alpha=0.8)
    axrr[0].fill_between(time, quantiles[0, :], quantiles[1, :], color='#598f70', linewidth=0, alpha=0.5, label='Predictive uncertainty')
    for idx in np.argwhere(person_data.session.diff().values == 1):
        if idx == 800:
            axrr[0].axvline(idx, color='black', linestyle='solid', lw=1.5, alpha=0.7)
        else:
            axrr[0].axvline(idx, color='black', linestyle='solid', lw=1.5, alpha=0.7)
    for idx in np.argwhere(person_data.block.diff().values == 1):
        if idx == 100:
            axrr[0].axvline(idx, color='black', linestyle='dotted', lw=1.5, alpha=0.4)
        else:
            axrr[0].axvline(idx, color='black', linestyle='dotted', lw=1.5, alpha=0.4)
    sns.despine(ax=axrr[0])
    axrr[0].grid(alpha=0.3)
    axrr[0].set_ylabel('RT(s)', fontsize=18)
    axrr[0].set_xlabel('Time(t)', fontsize=18)
    axrr[0].tick_params(axis='both', which='major', labelsize=16)
    reorderLegend(axrr[0],['SMA5: Empiric', 'SMA5: Predicted median', 'Predictive uncertainty'])
    axrr[0].grid(b=None)
    axrr[0].set_xticks(np.arange(0, 3201, 800))

    # plot empiric and predicted response time dist
    # axrr[1].get_shared_y_axes().join(axrr[1], axrr[0])
    # axrr[0].sharey(axrr[1])
    plt.setp(ax, ylim=(0, 1.5))
    sns.kdeplot(y=np.abs(emp_rt), fill="black", color="black", linewidth=1.5, alpha=0.3, label="Empiric", ax=axrr[1])
    sns.kdeplot(y=np.abs(pred_data_fast_dm), fill= '#852626', color="#852626", alpha=0.3, linewidth=1.5, label="Fast-dm", ax=axrr[1])
    sns.kdeplot(y=pred_data_dynamic.flatten(),fill="#598f70", color="#598f70", linewidth=1.5, alpha=0.5, label="Dynamic dm", ax=axrr[1])

    axrr[1].legend(fontsize=16)
    axrr[1].set_xlabel('Density', fontsize=18)
    axrr[1].tick_params(axis='both', which='major', labelsize=16)
    axrr[1].set_yticklabels('')
    sns.despine(ax=axrr[1])
    plt.subplots_adjust(wspace = 0.05)
    f.tight_layout()
    f.savefig("plots/rt_time_series_sub_{}.png".format(sub), dpi=300)
