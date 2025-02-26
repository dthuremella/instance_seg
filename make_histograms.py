import matplotlib.pyplot as plt
import pickle
import numpy as np
import pickle
import skimage.measure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

map_name = 'north' # 'south', 'north', 'center'
heatmap_type = 'tracked_occupancy' # 'dist', 'tracked_occupancy'

with open('{}loop_heatmap.pkl'.format(map_name), 'rb') as f:
    heatmap = pickle.load(f)
with open('{}loop_im_map.pkl'.format(map_name), 'rb') as f:
    im_map = pickle.load(f)

###### for dist heatmap ###################################
with open('{}loop_{}_heatmap.pkl'.format(map_name, heatmap_type), 'rb') as f:
    heatmap = pickle.load(f)

if heatmap_type == 'dist':
    heatmap += 0.01
    heatmap = 1 / heatmap


reduced = skimage.measure.block_reduce(heatmap, (150,150), np.sum)
plt.axis('off')  # command for hiding the axis. 
plt.imshow(reduced)
plt.savefig('{}_loop_{}heatmap.png'.format(map_name, heatmap_type), bbox_inches='tight', pad_inches=0)


# make histogram map figures
heatmap_points = np.where(heatmap > 0)
num_bins = 100
hist, x_edges, y_edges , _ = plt.hist2d(heatmap_points[0],heatmap_points[1], bins=num_bins)
midpoint_counts = []
for i in range(0, num_bins):
    for j in range(0, num_bins):
        if hist[i,j] > 0:
            xmid = (x_edges[i] + x_edges[i+1]) / 2
            ymid = (y_edges[j] + y_edges[j+1]) / 2
            count = hist[i,j]
            midpoint_counts.append([xmid, ymid, count])
midpoint_counts = np.array(midpoint_counts)
# sort by counts
midpoint_counts = midpoint_counts[midpoint_counts[:, 2].argsort()]


# plt.imshow(hist)
# plt.show()
# plt.savefig()
_, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.axis('off')
ax.imshow(im_map)
sc = ax.scatter(midpoint_counts[:,1], midpoint_counts[:,0], 
                s=midpoint_counts[:,2] / 10,
                c=midpoint_counts[:,2],
                cmap='magma')
# cbaxes = inset_axes(ax, width="3%", height="30%", loc='center right') 
# color_bar = plt.colorbar(sc, cax=cbaxes, fraction=0.025, pad=0)
# plt.rcParams['figure.facecolor'] = 'black'
# color_bar_ticklabels = plt.getp(color_bar.ax.axes, 'yticklabels')
# plt.setp(color_bar_ticklabels, color='k')
# plt.rcParams['figure.facecolor'] = 'white'  # Reset for future plots.
# plt.show()
# import pdb; pdb.set_trace()

plt.savefig("hotspot_{}heatmap_".format(heatmap_type)+map_name+".png", bbox_inches='tight', pad_inches=0)