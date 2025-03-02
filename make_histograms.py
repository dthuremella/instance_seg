import matplotlib.pyplot as plt
import pickle
import numpy as np
import pickle
import skimage.measure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_name", help='northloop southloop or centerloop', type=str)
    parser.add_argument("--heatmap_type", help='tracked_occupancy, cyclelane_overlap, dist or ttc', type=str)
    args = parser.parse_args()

    map_name = 'northloop' if args.map_name is None else args.map_name

    # 'dist', 'tracked_occupancy', 'ttc'
    heatmap_type = 'cyclelane_overlap' if args.heatmap_type is None else args.heatmap_type

    with open('{}_im_map.pkl'.format(map_name), 'rb') as f:
        im_map = pickle.load(f)

    ###### for all heatmaps ###################################
    max_val = {'dist': 10, 'ttc': 60} # max distance is 10m, max ttc is 1 minute
    bin_size = 150

    if args.heatmap_type is not None:
        with open('{}_{}_heatmap.pkl'.format(map_name, heatmap_type), 'rb') as f:
            heatmap = pickle.load(f)

        if map_name == 'southloop':
            if heatmap_type == 'cyclelane_overlap':
                factor = 20
                exponent = 3

            if heatmap_type == 'tracked_occupancy':
                factor = 1
                exponent = 1.8

            if heatmap_type == 'dist':
                factor = 1
                exponent = 2

            if heatmap_type == 'ttc':
                factor = 0.1
                exponent = 2

        if map_name == 'centerloop':
            if heatmap_type == 'cyclelane_overlap':
                factor = 1
                exponent = 2

            if heatmap_type == 'tracked_occupancy':
                factor = 1
                exponent = 1.8

            if heatmap_type == 'dist':
                factor = 10
                exponent = 1.5

            if heatmap_type == 'ttc':
                factor = 0.02
                exponent = 2

        if map_name == 'northloop':
            if heatmap_type == 'cyclelane_overlap':
                factor = 50
                exponent = 1

            if heatmap_type == 'tracked_occupancy':
                factor = 1
                exponent = 1.8

            if heatmap_type == 'dist':
                factor = 5
                exponent = 1.5

            if heatmap_type == 'ttc':
                factor = 0.1
                exponent = 1.5

    else:
        heatmap_type = 'all'
        exponent = 1.3
        factor = 100
        reduced_list = []
        for htype in ['tracked_occupancy', 'cyclelane_overlap', 'dist', 'ttc']:
            with open('{}_{}_reduced.pkl'.format(map_name, htype), 'rb') as f:
                reduced = pickle.load(f)
            if htype in ['dist', 'ttc']:
                reduced = reduced[:-1,:-1]
                reduced[np.where(reduced > max_val[htype])] = 0
                reduced[np.where(reduced != 0)] = max_val[htype] - reduced[np.where(reduced != 0)]
            if len(reduced_list) != 0 and reduced_list[0].shape != reduced.shape:
                reduced = np.pad(reduced, ((0, reduced_list[0].shape[0] - reduced.shape[0]), 
                                           (0, reduced_list[0].shape[1] - reduced.shape[1])), 'constant', constant_values=(0))
            reduced_list.append(reduced / np.mean(reduced))

        reduced_all = np.stack(reduced_list, axis=0)
        total_risk_heatmap = np.average(reduced_all, axis=0)
        total_risk_heatmap = total_risk_heatmap / np.max(total_risk_heatmap)

        plt.axis('off')  # command for hiding the axis.
        plt.imshow(total_risk_heatmap)
        plt.savefig('{}_{}heatmap.png'.format(map_name, heatmap_type), bbox_inches='tight', pad_inches=0)

        # make histogram map figures
        reduced = total_risk_heatmap
        midpoint_agg_vals = []
        for i in range(0, reduced.shape[0]):
            for j in range(0, reduced.shape[1]):
                if reduced[i,j] > 0: 
                    xmid = (i*bin_size + (i+1)*bin_size) / 2
                    ymid = (j*bin_size + (j+1)*bin_size) / 2
                    val = reduced[i,j]
                    midpoint_agg_vals.append([xmid, ymid, val])
        midpoint_agg_vals = np.array(midpoint_agg_vals)
        # sort by counts
        midpoint_agg_vals = midpoint_agg_vals[midpoint_agg_vals[:, 2].argsort()]
        plt.close()

        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')
        ax.imshow(im_map)
        sc = ax.scatter(midpoint_agg_vals[:,1], midpoint_agg_vals[:,0], 
                        s= midpoint_agg_vals[:,2]**exponent * factor,
                        c=midpoint_agg_vals[:,2],
                        cmap='magma')


    if heatmap_type in ['dist', 'ttc']:
        reduced = skimage.measure.block_reduce(heatmap, (bin_size,bin_size), np.min)
        with open('{}_{}_reduced.pkl'.format(map_name, heatmap_type), 'wb') as f: pickle.dump(reduced, f)
        plt.axis('off')  # command for hiding the axis. 
        plt.imshow(reduced)
        plt.savefig('{}_{}heatmap.png'.format(map_name, heatmap_type), bbox_inches='tight', pad_inches=0)

        # make histogram map figures
        midpoint_agg_vals = []
        for i in range(0, reduced.shape[0]-1):
            for j in range(0, reduced.shape[1]-1):
                if ~np.isinf(reduced[i,j]) and reduced[i,j] < max_val[heatmap_type] and reduced[i,j] > 0: 
                    # xmid = (x_edges[i] + x_edges[i+1]) / 2
                    # ymid = (y_edges[j] + y_edges[j+1]) / 2
                    xmid = (i*bin_size + (i+1)*bin_size) / 2
                    ymid = (j*bin_size + (j+1)*bin_size) / 2
                    val = reduced[i,j]
                    midpoint_agg_vals.append([xmid, ymid, val])
        midpoint_agg_vals = np.array(midpoint_agg_vals)
        # sort by counts
        midpoint_agg_vals = midpoint_agg_vals[midpoint_agg_vals[:, 2].argsort()][::-1]
        plt.close()

        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')
        ax.imshow(im_map)
        sc = ax.scatter(midpoint_agg_vals[:,1], midpoint_agg_vals[:,0], 
                        s= (np.max(midpoint_agg_vals[:,2]) - midpoint_agg_vals[:,2])**exponent * factor,
                        c=midpoint_agg_vals[:,2],
                        cmap='magma_r')

    if heatmap_type in ['tracked_occupancy', 'cyclelane_overlap']:
        new_hmap_dict = {}
        x_ticks = list(range(int(bin_size / 2), im_map.shape[0], bin_size))
        y_ticks = list(range(int(bin_size / 2), im_map.shape[1], bin_size))
        for point in heatmap:
            x = int(np.round(point[0] / im_map.shape[0] * len(x_ticks)))
            y = int(np.round(point[1] / im_map.shape[1] * len(y_ticks)))
            key = (x, y)
            if key not in new_hmap_dict:
                new_hmap_dict[key] = set()
            for tid in heatmap[point]:
                new_hmap_dict[key].add(tid)

        new_hmap = np.zeros((len(x_ticks), len(y_ticks)))
        midpoint_agg_vals = []
        for point in new_hmap_dict:
            if ((heatmap_type == 'tracked_occupancy' and len(new_hmap_dict[point]) > 50) # set up issues in beginning of run
                or (heatmap_type == 'cyclelane_overlap' and len(new_hmap_dict[point]) >= 15)): # weird outlier
                continue
            x_r = point[0] / len(x_ticks) * im_map.shape[0]
            y_r = point[1] / len(y_ticks) * im_map.shape[1]
            midpoint_agg_vals.append([x_r, y_r, len(new_hmap_dict[point])])
            new_hmap[point[0], point[1]] = len(new_hmap_dict[point])
        midpoint_agg_vals = np.array(midpoint_agg_vals)
        # sort by counts
        midpoint_agg_vals = midpoint_agg_vals[midpoint_agg_vals[:, 2].argsort()]
        
        # reduced is the new_hmap
        reduced = new_hmap
        with open('{}_{}_reduced.pkl'.format(map_name, heatmap_type), 'wb') as f: pickle.dump(reduced, f)
        plt.axis('off')  # command for hiding the axis. 
        plt.imshow(reduced)
        plt.savefig('{}_{}heatmap.png'.format(map_name, heatmap_type), bbox_inches='tight', pad_inches=0)

        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')
        ax.imshow(im_map)
        sc = ax.scatter(midpoint_agg_vals[:,1], midpoint_agg_vals[:,0], 
                        s= midpoint_agg_vals[:,2]**exponent * factor,
                        c=midpoint_agg_vals[:,2],
                        cmap='magma')
        
    cbaxes = inset_axes(ax, width="3%", height="30%", loc='center right') 
    color_bar = plt.colorbar(sc, cax=cbaxes, fraction=0.025, pad=0)
    plt.rcParams['figure.facecolor'] = 'black'
    color_bar_ticklabels = plt.getp(color_bar.ax.axes, 'yticklabels')
    plt.setp(color_bar_ticklabels, color='k')
    plt.rcParams['figure.facecolor'] = 'white'  # Reset for future plots.
    # plt.show()
    # import pdb; pdb.set_trace()

    plt.savefig("hotspot_{}heatmap_".format(heatmap_type)+map_name+".png", bbox_inches='tight', pad_inches=0)

        





if __name__ == '__main__':
    main()