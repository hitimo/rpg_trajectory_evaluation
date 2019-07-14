#!/usr/bin/env python2
"""
@author: Christian Forster
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import transformations as tf
import numpy as np
rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

FORMAT = '.pdf'


def color_box(bp, color):
    elements = ['medians', 'boxes', 'caps', 'whiskers']
    # Iterate over each of the elements changing the color
    for elem in elements:
        [plt.setp(bp[elem][idx], color=color, linestyle='-', lw=1.0)
         for idx in range(len(bp[elem]))]
    return


def boxplot_compare(ax, xlabels,
                    data, data_labels, data_colors):
    n_data = len(data)
    n_xlabel = len(xlabels)
    leg_handles = []
    leg_labels = []
    idx = 0
    for idx, d in enumerate(data):
        w = 1 / (1.5 * n_data + 1.5)
        widths = [w for pos in np.arange(n_xlabel)]
        positions = [pos - 0.5 + 1.5 * w + idx * w
                     for pos in np.arange(n_xlabel)]
        bp = ax.boxplot(d, 0, '', positions=positions, widths=widths)
        color_box(bp, data_colors[idx])
        tmp, = plt.plot([1, 1], data_colors[idx])
        leg_handles.append(tmp)
        leg_labels.append(data_labels[idx])
        idx += 1
    ax.set_xticks(np.arange(n_xlabel))
    ax.set_xticklabels(xlabels)
    xlims = ax.get_xlim()
    ax.set_xlim([xlims[0]-0.1, xlims[1]-0.1])
    if n_data != 1:
        ax.legend(leg_handles, leg_labels, bbox_to_anchor=(
            1.05, 1), loc=2, borderaxespad=0.)
    map(lambda x: x.set_visible(False), leg_handles)


def plot_trajectory_top(ax, pos, color, name):
    ax.grid(ls='--', color='0.7')
    # pos_0 = pos - pos[0, :]
    ax.plot(pos[:, 0], pos[:, 1], color+'-', label=name)


def plot_trajectory_side(ax, pos, color, name):
    ax.grid(ls='--', color='0.7')
    # pos_0 = pos - pos[0, :]
    ax.plot(pos[:, 0], pos[:, 2], color+'-', label=name)


def plot_aligned_top(ax, p_gt, p_es, n_align_frames):
    if n_align_frames <= 0:
        n_align_frames = p_es.shape[0]
    # p_es_0 = p_es - p_gt[0, :]
    # p_gt_0 = p_gt - p_gt[0, :]
    ax.plot(p_es[0:n_align_frames, 0], p_es[0:n_align_frames, 1],
            'g-', linewidth=2, label='aligned')
    for (x1, y1, z1), (x2, y2, z2) in zip(
            p_es[:n_align_frames:10, :], p_gt[:n_align_frames:10, :]):
        ax.plot([x1, x2], [y1, y2], '-', color="gray")

def plot_xyz(ax, distances, pos, i, color, name):
    ax.plot(distances, pos[:,i], color+'-', label=name)

def plot_rpy(ax, distances, quat, angle, color, name):
    # Convert quaternion to Euler angles
    rpy = np.zeros(np.shape(quat[:,0:3]))
    for i in range(np.shape(quat[:,0:3])[0]):
        R = tf.matrix_from_quaternion(quat[i, :])
        rpy[i, :] = tf.euler_from_matrix(R, 'rxyz')

    ax.plot(distances, rpy[:,angle]*180.0/np.pi, color+'-', label=name)

def plot_error_n_dim(ax, distances, errors, results_dir,
                     colors=['r', 'g', 'b'],
                     labels=['x', 'y', 'z']):
    assert len(colors) == len(labels)
    assert len(colors) == errors.shape[1]
    for i in range(len(colors)):
        ax.plot(distances, errors[:, i],
                colors[i]+'-', label=labels[i])

def plot_error_n_dim_abs(ax, distances, errors, results_dir,
                     colors=['r', 'g', 'b'],
                     labels=['x', 'y', 'z']):
    assert len(colors) == len(labels)
    assert len(colors) == errors.shape[1]
    for i in range(len(colors)):
        ax.plot(distances, abs(errors[:, i]),
                colors[i]+'-', label=labels[i])
