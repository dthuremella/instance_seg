import numpy as np
import pcl

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pickle
import pandas as pd
import open3d as o3d

from cyclelane_heatmap_generation import P, m_pixels_to_xy, m_xy_to_pixels
from cyclelane_heatmap_generation import LonLat_To_XY, XY_To_LonLat, pixels_to_latlong, latlong_to_pixels, point_cloud_center, point_cloud_volume
from segmentation_to_instance import get_labels_instances, gen_labels

############ params to change ##########################
map_prefix = 'north'
csv_dir = '/Volumes/scratchdata/robotcycle_exports/2024-10-18-15-10-24/motion'

bin_file_folder = '/Volumes/scratchdata/efimia/robotcycle_{}_loop/concat_pc_bin_files'.format(map_prefix)
label_file_folder = '{}_loop_concat_cylinder8x'.format(map_prefix)
translation_pkl_file = '{}_loop_concat_to_right_translation.pkl'.format(map_prefix)
map_name = '{}loop'.format(map_prefix)

output_file_folder = '{}_loop_clustering_instance_seg_labeled'.format(map_prefix)
save_npy = False

with open(translation_pkl_file, 'rb') as handle:
    # filename_to_timestamp = pickle.load(handle)
    bad_dict = pickle.load(handle)
bad_keys = [k  for  k in  bad_dict.keys()]
bad_keys.sort()
bad_vals = [v  for  v in  bad_dict.values()]
bad_vals.sort()
filename_to_timestamp = {}
for i in range(800, len(bad_keys)):
    key = bad_keys[i]
    val = bad_vals[i]
    filename_to_timestamp[key] = val

plot_cluster = True

offset = 20 # calculated by eye for map, for x and y

print_info = False

sample_every_100th = False

########################################################


# read image map
im_map = plt.imread('oxford_' + map_name + '.png') # do for northloop and southloop too
im_map[np.where(im_map[:,:,2] != 1)] = 0  # remove  except cycle_lanes and outlines
is_cyclelane = im_map.copy()
is_cyclelane[np.where(im_map[:,:,2]+im_map[:,:,1]+im_map[:,:,0] == 3)] = 0 #remove outlines
is_cyclelane = is_cyclelane[:,:,2]
road_lines = im_map[:,:,1]
heatmap = np.zeros(im_map[:,:,0].shape)

# read gps and imu csvs
gps_filename = csv_dir + '/gps.csv'
gps_df = pd.read_csv(gps_filename, sep=';')
gps_timestamps = np.array(gps_df['timestamp'])
imu_filename = csv_dir + '/imu.csv'
imu_df = pd.read_csv(imu_filename, sep=';')
imu_timestamps = np.array(imu_df['timestamp'])


def interpolate(array, query, dtype):
    i = np.searchsorted(array, query)
    if dtype == 'gps':
        value_i = [gps_df.iloc[i]['latitude'], gps_df.iloc[i]['longitude']]
        value_im1 = [gps_df.iloc[i-1]['latitude'], gps_df.iloc[i-1]['longitude']]
    else:
        value_i = [imu_df.iloc[i]['orientation_x'], imu_df.iloc[i]['orientation_y'], imu_df.iloc[i]['orientation_z']]
        value_im1 = [imu_df.iloc[i-1]['orientation_x'], imu_df.iloc[i-1]['orientation_y'], imu_df.iloc[i-1]['orientation_z']]

    total_diff = array[i] - array[i-1]
    diff_to_i = (array[i] - query) / total_diff
    diff_to_im1 = (query - array[i-1]) / total_diff

    ret = []
    for j in range(len(value_i)):
        interpolated = diff_to_i * value_i[j] + diff_to_im1 * value_im1[j]
        ret.append(interpolated)
    return ret

def main():

    bin_files = [f for f in listdir(bin_file_folder) if isfile(join(bin_file_folder, f))]
    label_files = [f for f in listdir(label_file_folder) if isfile(join(label_file_folder, f))]

    bin_files.sort()
    label_files.sort()

    assert len(bin_files) == len(label_files), "bin files and label files are not same length"
    for i in range(len(bin_files)):
        if sample_every_100th and i % 100 != 0: 
            continue
        print ('i = {} out of {}'.format(i, len(bin_files)))
        #calculate timestamp
        filenumber = bin_file.split('/')[-1].split('.')[0]
        if filenumber not in filename_to_timestamp:
            print(filenumber + ' not in dict')
            continue
        timestamp = int(filename_to_timestamp[filenumber])

        bin_file = bin_files[i]
        label_file = label_files[i]
        print('doing ' + bin_file)
        cluster = get_labels_instances(bin_file_folder + '/' + bin_file, 
                            label_file_folder + '/' + label_file, 
                            output_file_folder,
                            save_npy=save_npy)
        
        print('doing file ' + filenumber + ' timestamp {}'.format(timestamp))
        latlong = interpolate(gps_timestamps, timestamp, 'gps')
        if print_info: print(latlong)
        orientation = interpolate(imu_timestamps, timestamp, 'imu') # [x, y, z]

        # lidars are originally rotated 90deg around z axis and 180deg around x axis
        R_toIMU = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, np.pi / 2))
        lidar_to_imu_cluster = np.matmul(cluster[:,:3], R_toIMU)

        # transform cluster's xyz into 0,0,0 orientation
        R = o3d.geometry.get_rotation_matrix_from_xyz(tuple(orientation))
        rotated_points = np.matmul(lidar_to_imu_cluster, R)

        # it's supposed to be NED, but it's rotated to the left 90 degrees
        # which means it's WND
        R_fromWND_toENU = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0.1*np.pi))
        enu_points = np.matmul(rotated_points, R_fromWND_toENU)


        # remove cars with > 2m high center, small volume
        is_veh = np.where((cluster[:,4] == (1 or 4 or 5)))
        for inst_id in np.unique(cluster[:,5][is_veh]):
            is_inst = np.where(cluster[:,5] == inst_id)
            inst_points = enu_points[is_inst][:,:3]
            if ((point_cloud_volume(inst_points) < 1) or # less than 1m cubed
                (point_cloud_center(inst_points)[-1] > 2)): # higher than 2m
                for i in is_inst: cluster[i,4] = 0 # put it in unknown class
                if print_info: print('inst_id {} is too small {} or high {}'.format(inst_id, 
                            point_cloud_volume(inst_points),
                            point_cloud_center(inst_points)[-1]))

        # get vehicle_ids
        vehicle_ids = np.where((cluster[:,4] == (1 or 4 or 5)))
        enu_vehs_top_down = enu_points[:,:2][vehicle_ids]

        # translate to map_coords
        translation = np.array(LonLat_To_XY(latlong[1], latlong[0]))
        if print_info: print('translation ', translation)
        vehs_top_down_xy = enu_vehs_top_down + translation
        veh_xy_extended = np.concatenate((vehs_top_down_xy, 
                                          np.ones((vehs_top_down_xy.shape[0],1))), 
                                          axis=1)
        vehs_top_down_pixels = np.matmul(veh_xy_extended, m_xy_to_pixels[map_name])
        center_xy_extended = np.array([translation[0], translation[1], 1])
        new_center = np.matmul(center_xy_extended, m_xy_to_pixels[map_name])
        if print_info: print(new_center)

        ppoints = vehs_top_down_pixels.astype(np.int64)
        # remove duplicates
        upoints = np.vstack([np.array(u) for u in set([tuple(p) for p in ppoints])])
        upoints[:,0] += -offset
        upoints[:,1] += offset
        for p in upoints:
            if is_cyclelane[p[0], p[1]]:
                heatmap[p[0], p[1]] += 1

        if plot_cluster: 
            heatmap_mask = (heatmap > 0).astype(np.int64)
            im_map[:,:,0] = heatmap_mask
            new_center = new_center.astype(np.int64)
            im_map[new_center[0]+20, new_center[1]-20] = [1,0,0,1]
            fig, ax = plt.subplots()
            ax.imshow(im_map)
            ax.scatter(vehs_top_down_pixels[:,1]+20, vehs_top_down_pixels[:,0]-20, s=1)
            plt.show()

            import pdb; pdb.set_trace()
            
            fig, ax = plt.subplots()
            ax.scatter(enu_vehs_top_down[:,0], enu_vehs_top_down[:,1], s=1)
            ax.set_xlim((-25,25))
            ax.set_ylim((-25,25))
            plt.show()

            cluster_to_plot = enu_points
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xs = cluster_to_plot[:,0][np.where(cluster[:,4] == 1)]
            ys = cluster_to_plot[:,1][np.where(cluster[:,4] == 1)]
            zs = cluster_to_plot[:,2][np.where(cluster[:,4] == 1)]
            colors = cluster[:,5][np.where(cluster[:,4] == 1)]
            ax.scatter(xs, ys, zs, c=colors, s=1)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()

    # heatmap_mask = (heatmap > 0).astype(np.int64)
    # im_map[:,:,0] = heatmap_mask

    with open('heatmap_{}.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(heatmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('im_map_{}.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(im_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()