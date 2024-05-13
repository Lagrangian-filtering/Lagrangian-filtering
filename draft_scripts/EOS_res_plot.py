import sys
# import os
sys.path.append('../../master_files/')
import pickle
import configparser
import json
import time
import math
from scipy import stats

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ##################################################################
    # # RUN THE REGRESSION ROUTINE GIVEN DEPENDENT DATA AND EXPLANATORY 
    # # VARS. DATA IS PRE-PROCESSED 
    # ################################################################## 
    
    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # LOADING MESO MODEL
    pickle_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Filenames']['meso_pickled_filename']
    MesoModelLoadFile = pickle_directory + meso_pickled_filename

    print('================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = plot_ranges['x_range']
    y_range = plot_ranges['y_range']
    num_T_slices = int(config['Models_settings']['num_T_slices'])
    t_meso = meso_model.domain_vars['T'][int((num_T_slices-1)/2)]
    visualizer = Plotter_2D()


    EOS_res, extent = visualizer.get_var_data(meso_model, 'eos_res', t_meso, x_range, y_range)
    m, M = np.amin(EOS_res), np.amax(EOS_res)
    print(f'm, M (EOS)= {m} , {M}\n')

    Pi_res, extent = visualizer.get_var_data(meso_model, 'Pi_res', t_meso, x_range, y_range)
    m, M = np.amin(Pi_res), np.amax(Pi_res)
    print(f'm, M (Pi_res)= {m} , {M}\n')

    sum = EOS_res + Pi_res
    m, M = np.amin(sum), np.amax(sum)
    print(f'm, M (sum)= {m} , {M}\n')

    print(f'extent: {extent}\n')
    
    # plt.rc("font",family="serif")
    # plt.rc("mathtext",fontset="cm")
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12,4], sharey=True)
    # axes = axes.flatten()

    # images = []

    # im = axes[0].imshow(-EOS_res, extent=extent, origin='lower', cmap='plasma', norm = 'log')
    # divider = make_axes_locatable(axes[0])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cax.set_axis_off()
    # images.append(im)
    # # fig.colorbar(im, cax=cax, orientation='vertical')

    # title = r'$-M$'
    # axes[0].set_xlabel(r'$x$')
    # axes[0].set_ylabel(r'$y$')
    # axes[0].set_title(title, fontsize=10)

    # im = axes[1].imshow(Pi_res, extent=extent, origin='lower', cmap='plasma', norm = 'log')
    # divider = make_axes_locatable(axes[1])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cax.set_axis_off()
    # images.append(im)
    # # fig.colorbar(im, cax=cax, orientation='vertical')

    # title = r'$\tilde{\Pi}$'
    # axes[1].set_xlabel(r'$x$')
    # axes[1].set_ylabel(r'$y$')
    # axes[1].set_title(title, fontsize=10)

    # im = axes[2].imshow(sum, extent=extent, origin='lower', cmap='plasma', norm = 'log')
    # divider = make_axes_locatable(axes[2])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # images.append(im)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    # title = r'$M + \tilde{\Pi}$'
    # axes[2].set_xlabel(r'$x$')
    # axes[2].set_ylabel(r'$y$')
    # axes[2].set_title(title, fontsize=10)


    # vmin = min(image.get_array().min() for image in images)
    # vmax = max(image.get_array().max() for image in images)
    # norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    # for im in images:
    #     im.set_norm(norm)

    # def update(changed_image):
    #     for im in images:
    #         if (changed_image.get_cmap() != im.get_cmap()
    #                 or changed_image.get_clim() != im.get_clim()):
    #             im.set_cmap(changed_image.get_cmap())
    #             im.set_clim(changed_image.get_clim())

    # for im in images:
    #     im.callbacks.connect('changed', update)

    # fig.tight_layout()

    # fig_directory = config['Directories']['figures_dir'] 
    # filename = 'EOS_res_plots'
    # format = 'png'
    # dpi = 400
    # filename += "." + format
    # plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    # plt.close()


    # # ################################################################## 
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5,4])


    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
    #     warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')

    #     X = np.log10(np.abs(EOS_res))
    #     X = X.flatten()
    #     sns.histplot(X, ax=ax, stat='density', kde=True, label=r'$\log(|M|)$')

    #     X = np.log10(Pi_res)
    #     X = X.flatten()
    #     sns.histplot(X, ax=ax, stat='density', kde=True, label=r'$\log(\tilde{\Pi})$')

    #     X = np.log10(Pi_res + EOS_res)
    #     X = X.flatten()
    #     sns.histplot(X, ax=ax, stat='density', kde=True, label=r'$\log(\tilde{\Pi} + M)$')


    #     ax.legend(loc = 'best', prop={'size': 10})
   
    # fig.tight_layout()

    # fig_directory = config['Directories']['figures_dir'] 
    # filename = 'EOS_res_distr'
    # format = 'png'
    # dpi = 400
    # filename += "." + format
    # plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    # plt.close()

    # ######## 
    # MERGING THE TWO PLOTS 
    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[13,4], sharey=True)
    # axes = axes.flatten()

    # now plotting    
    fig = plt.figure(figsize=[13,4])
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3, sharey=ax2)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax4 = fig.add_subplot(1, 4, 4, sharey=ax2)
    plt.setp(ax4.get_yticklabels(), visible=False)
    axes = [ax1, ax2, ax3, ax4]
    axesRight = [ax2, ax3, ax4]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
        warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')

        X = np.log10(np.abs(EOS_res))
        X = X.flatten()
        sns.histplot(X, ax=ax1, stat='density', kde=True, label=r'$\log(|M|)$')

        X = np.log10(Pi_res)
        X = X.flatten()
        sns.histplot(X, ax=ax1, stat='density', kde=True, label=r'$\log(\tilde{\Pi})$')

        X = np.log10(Pi_res + EOS_res)
        X = X.flatten()
        sns.histplot(X, ax=ax1, stat='density', kde=True, label=r'$\log(\tilde{\Pi} + M)$')

        ax1.legend(loc = 'best', prop={'size': 10})

    images = []

    im = ax2.imshow(-EOS_res, extent=extent, origin='lower', cmap='plasma', norm = 'log')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_axis_off()
    images.append(im)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    title = r'$-M$'
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.set_title(title, fontsize=10)

    im = ax3.imshow(Pi_res, extent=extent, origin='lower', cmap='plasma', norm = 'log')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_axis_off()
    images.append(im)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    title = r'$\tilde{\Pi}$'
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$y$')
    ax3.set_title(title, fontsize=10)

    im = ax4.imshow(sum, extent=extent, origin='lower', cmap='plasma', norm = 'log')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    images.append(im)
    fig.colorbar(im, cax=cax, orientation='vertical')

    title = r'$M + \tilde{\Pi}$'
    ax4.set_xlabel(r'$x$')
    ax4.set_ylabel(r'$y$')
    ax4.set_title(title, fontsize=10)


    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacks.connect('changed', update)

    fig.tight_layout()

    fig_directory = config['Directories']['figures_dir'] 
    filename = 'EOS_res_plots+distr'
    format = 'png'
    dpi = 400
    filename += "." + format
    plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    plt.close()