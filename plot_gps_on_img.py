import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cyclelane_heatmap_generation import latlong_to_pixels

############ params to change ##########################
map_name = 'centerloop' # or northloop or southloop

csv_dir = 'central_loop_csvs'

im_map = plt.imread('paper_central_loop.png') # do for northloop and southloop too

########################################################

# read gps and imu csvs
gps_filename = csv_dir + '/gps.csv'
gps_df = pd.read_csv(gps_filename, sep=';')

xs = []
ys = []
for index, row in gps_df.iterrows():
    lat = [row['latitude']]
    lon = [row['longitude']]
    pixel = latlong_to_pixels(lat, lon, map_name)
    xs.append(int(pixel[0][0]))
    ys.append(int(pixel[0][1]))

plt.axis('off')  # command for hiding the axis. 
plt.imshow(im_map)
plt.plot(ys, xs, linewidth=5, color='black', alpha=0.3)
plt.savefig('{}_with_gps.png'.format(map_name),  dpi=1000, bbox_inches='tight', pad_inches=0)

