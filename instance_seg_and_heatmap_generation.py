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

from cyclelane_heatmap_generation import P, map_names, pixels, latlongs, m_pixels_to_xy, m_xy_to_pixels
from cyclelane_heatmap_generation import LonLat_To_XY, XY_To_LonLat, pixels_to_latlong, latlong_to_pixels, point_cloud_center, point_cloud_volume


learning_map={0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped       x
    10: 1,     # "car"
    11: 2,     # "bicycle"                                                              x
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"                                                           x
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"                                                               x
    31: 7,     # "bicyclist"                                                            x
    32: 8,     # "motorcyclist"                                                         x
    40: 9,     # "road"
    44: 10,     # "parking"  
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5}

max_key = max(learning_map.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(learning_map.keys())] = list(learning_map.values())

node_map={
  1: 0,      # "car"
  4: 1,      # "truck"
  5: 2,      # "other-vehicle"
  11: 3,     # "sidewalk"
  12: 4,     # "other-ground"
  13: 5,     # "building"
  14: 6,     # "fence"
  15: 7,     # "vegetation"
  16: 8,     # "trunk"
  17: 9,     # "terrain"
  18: 10,    # "pole"
  19: 11     # "traffic-sign"
  }

def gen_labels(scan_name, label_name, label_output_dir):
        # start = time.time()
        # open scan
        scan = np.fromfile(scan_name, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        # put in attribute
        points = scan[:, 0:4]  # get xyzr
        remissions = scan[:, 3]  # get remission

        label = np.fromfile(label_name, dtype=np.uint32)
        label = label.reshape((-1))

        # # demolition or not
        # if FLAGS.demolition == True:
        #     start_angle = np.random.random()
        #     start_angle *= 360
        #     end_angle = (start_angle + drop_angle)%360

        #     angle = np.arctan2(points[:, 1], points[:, 0])
        #     angle = angle*180/np.pi
        #     angle += 180
        #     # print("angle:", angle)
        #     if end_angle > start_angle:
        #         remain_id = np.argwhere(angle < start_angle).reshape(-1)
        #         remain_id = np.append(remain_id, np.argwhere(angle > end_angle).reshape(-1))
        #     else:
        #         remain_id = np.argwhere((angle > end_angle) & (angle < start_angle)).reshape(-1)

        #     points = points[remain_id, : ]
        #     label = label[remain_id]

        if label.shape[0] == points.shape[0]:
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16  # instance id in upper half
            assert ((sem_label + (inst_label << 16) == label).all())
        else:
            print("Points shape: ", points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        sem_label = remap_lut[sem_label]
        sem_label_set = list(set(sem_label))
        sem_label_set.sort()

        # Start clustering
        cluster = []
        inst_id = 0
        for id_i, label_i in enumerate(sem_label_set):
            # print('sem_label:', label_i)
            index = np.argwhere(sem_label == label_i)
            index = index.reshape(index.shape[0])
            sem_cluster = points[index, :]
            # print("sem_cluster_shape:",sem_cluster.shape[0])

            tmp_inst_label = inst_label[index]
            tmp_inst_set = list(set(tmp_inst_label))
            tmp_inst_set.sort()
            # print(tmp_inst_set)

            if label_i in [9, 10]:    # road/parking, dont need to cluster
                inst_cluster = sem_cluster
                inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), label_i, dtype=np.uint32)), axis=1)
                # inst_cluster = np.insert(inst_cluster, 4, label_i, axis=1)
                inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), inst_id, dtype=np.uint32)), axis=1)
                inst_id = inst_id + 1
                cluster.extend(inst_cluster)  # Nx6                
                continue
                
            elif label_i in [0,2,3,6,7,8]:    # discard
                continue
            
            elif len(tmp_inst_set) > 1 or (len(tmp_inst_set) == 1 and tmp_inst_set[0] != 0):     # have instance labels
                for id_j, label_j in enumerate(tmp_inst_set):
                    points_index = np.argwhere(tmp_inst_label == label_j)
                    points_index = points_index.reshape(points_index.shape[0])
                    # print(id_j, 'inst_size:', len(points_index))
                    if len(points_index) <= 20:
                        continue
                    inst_cluster = sem_cluster[points_index, :]
                    inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), label_i, dtype=np.uint32)), axis=1)
                    # inst_cluster = np.insert(inst_cluster, 4, label_i, axis=1)
                    inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), inst_id, dtype=np.uint32)), axis=1)
                    inst_id = inst_id + 1
                    cluster.extend(inst_cluster)
            else:    # Euclidean cluster
                # time_start = time.time()
                if label_i in [1, 4, 5, 14]:     # car truck other-vehicle fence
                    cluster_tolerance = 0.5
                elif label_i in [11, 12, 13, 15, 17]:    # sidewalk other-ground building vegetation terrain
                    cluster_tolerance = 2
                else:
                    cluster_tolerance = 0.2

                if label_i in [16, 19]:    # trunk traffic-sign
                    min_size = 50
                elif label_i == 15:     # vegetation
                    min_size = 200
                elif label_i in [11, 12, 13, 17]:    # sidewalk other-ground building terrain
                    min_size = 300
                else:
                    min_size = 100

                # print(cluster_tolerance, min_size)
                cloud = pcl.PointCloud(sem_cluster[:, 0:3])
                tree = cloud.make_kdtree()
                ec = cloud.make_EuclideanClusterExtraction()
                ec.set_ClusterTolerance(cluster_tolerance)
                ec.set_MinClusterSize(min_size)
                ec.set_MaxClusterSize(50000)
                ec.set_SearchMethod(tree)
                cluster_indices = ec.Extract()
                # time_end = time.time()
                # print(time_end - time_start)
                for j, indices in enumerate(cluster_indices):
                    # print('j = ', j, ', indices = ' + str(len(indices)))
                    inst_cluster = np.zeros((len(indices), 4), dtype=np.float32)
                    inst_cluster = sem_cluster[np.array(indices), 0:4]
                    inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), label_i, dtype=np.uint32)), axis=1)
                    # inst_cluster = np.insert(inst_cluster, 4, label_i, axis=1)
                    inst_cluster = np.concatenate((inst_cluster, np.full((inst_cluster.shape[0],1), inst_id, dtype=np.uint32)), axis=1)
                    inst_id = inst_id + 1
                    cluster.extend(inst_cluster) # Nx6

        # print(time.time()-start)
        # print('*'*80)
        cluster = np.array(cluster)
        # np.save(label_output_dir+'/'+label_name.split('/')[-1].split('.')[0]+".npy", cluster)

        # if 'path' in FLAGS.pub_or_path:
        #     np.save(label_output_dir+'/'+label_name.split('/')[-1].split('.')[0]+".npy", cluster)
        # if 'pub' in FLAGS.pub_or_path:
        #     # print(cluster[11100:11110])
        #     msg_points = pc2.create_cloud(header=self.header1, fields=_make_point_field(cluster.shape[1]), points=cluster)
        #     self._labels_pub.publish(msg_points)

        return cluster 


############ params to change ##########################
map_prefix = 'north'
csv_dir = '/Volumes/scratchdata/robotcycle_exports/2024-10-18-15-10-24/motion'

bin_file_folder = '/Volumes/scratchdata/efimia/robotcycle_{}_loop/concat_pc_bin_files'.format(map_prefix)
label_file_folder = '{}_loop_concat_cylinder_8x'.format(map_prefix)
output_file_folder = '{}_loop_clustering_instance_seg_labeled'.format(map_prefix)
translation_pkl_file = '{}_loop_concat_to_right_translation.pkl'.format(map_prefix)
map_name = '{}loop'.format(map_prefix)

with open(translation_pkl_file, 'rb') as handle:
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


plot_cluster = False

offset = 20 # calculated by eye for map, for x and y

print_info = False

########################################################


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
        bin_file = bin_files[i]
        label_file = label_files[i]
        print('doing ' + bin_file)
        cluster = gen_labels(bin_file_folder + '/' + bin_file, 
                            label_file_folder + '/' + label_file, 
                            output_file_folder)
        
        #calculate timestamp
        filenumber = bin_file.split('/')[-1].split('.')[0]
        if filenumber not in filename_to_timestamp:
            print(filenumber + ' not in dict')
            continue

        timestamp = int(filename_to_timestamp[filenumber])
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

    heatmap_mask = (heatmap > 0).astype(np.int64)
    im_map[:,:,0] = heatmap_mask

    with open('heatmap_{}.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(heatmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('im_map_{}.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(im_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()