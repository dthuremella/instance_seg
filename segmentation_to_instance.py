import numpy as np
import pcl
import open3d as o3d
from sklearn.cluster import DBSCAN

from os import listdir
from os.path import isfile, join

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

def gen_labels(scan_name, label_name, label_output_dir, save_npy=False):
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
    if save_npy:
        np.save(label_output_dir+'/'+label_name.split('/')[-1].split('.')[0]+".npy", cluster)

    # if 'path' in FLAGS.pub_or_path:
    #     np.save(label_output_dir+'/'+label_name.split('/')[-1].split('.')[0]+".npy", cluster)
    # if 'pub' in FLAGS.pub_or_path:
    #     # print(cluster[11100:11110])
    #     msg_points = pc2.create_cloud(header=self.header1, fields=_make_point_field(cluster.shape[1]), points=cluster)
    #     self._labels_pub.publish(msg_points)

    return cluster 

def get_labels_instances(scan_name, label_name, label_output_dir, save_npy=False):
    scan = np.fromfile(scan_name, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:4]
    label = np.fromfile(label_name, dtype=np.uint32).reshape((-1))
    assert label.shape[0] == points.shape[0], "Scan and Label don't contain the same number of points"

    sem_label = label & 0xFFFF
    inst_label = label >> 16
    assert (sem_label + (inst_label << 16) == label).all(), "Invalid label format"

    max_key = max(learning_map.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(learning_map.keys())] = list(learning_map.values())

    sem_label = remap_lut[sem_label]
    sem_label_set = list(set(sem_label))
    sem_label_set.sort()

    # Start clustering
    cluster = []
    inst_id = 0

    for label_i in sem_label_set:
        index = np.argwhere(sem_label == label_i).reshape(-1)
        sem_cluster = points[index]

        tmp_inst_label = inst_label[index]
        tmp_inst_set = np.unique(tmp_inst_label)

        if label_i in [20, 31, 32, 33, 34, 35, 36]:  # discard unlabeled, outliers, moving objects
            continue
        elif label_i in [9, 10]:  # road/parking, dont need to cluster
            inst_cluster = np.hstack((sem_cluster, np.full((sem_cluster.shape[0], 1), label_i, dtype=np.uint32)))
            inst_cluster = np.hstack((inst_cluster, np.full((inst_cluster.shape[0], 1), inst_id, dtype=np.uint32)))
            inst_id += 1
            cluster.extend(inst_cluster)
            continue
        elif len(tmp_inst_set) > 1 or (len(tmp_inst_set) == 1 and tmp_inst_set[0] != 0):  # have instance labels
            for label_j in tmp_inst_set:
                points_index = np.argwhere(tmp_inst_label == label_j).reshape(-1)
                if len(points_index) <= 20:
                    continue
                inst_cluster = sem_cluster[points_index]
                inst_cluster = np.hstack((inst_cluster, np.full((inst_cluster.shape[0], 1), label_i, dtype=np.uint32)))
                inst_cluster = np.hstack((inst_cluster, np.full((inst_cluster.shape[0], 1), inst_id, dtype=np.uint32)))
                inst_id += 1
                cluster.extend(inst_cluster)
        else:  # Euclidean cluster
            if label_i in [1, 4, 5, 14]:  # car truck other-vehicle fence
                cluster_tolerance = 0.5
            elif label_i in [11, 12, 13, 15, 17]:  # sidewalk other-ground building vegetation terrain
                cluster_tolerance = 2
            else:
                cluster_tolerance = 0.2

            if label_i in [16, 19]:  # trunk traffic-sign
                min_size = 50
            elif label_i == 15:  # vegetation
                min_size = 200
            elif label_i in [11, 12, 13, 17]:  # sidewalk other-ground building terrain
                min_size = 300
            else:
                min_size = 100

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(sem_cluster[:, 0:3])
            dbscan = DBSCAN(eps=cluster_tolerance, min_samples=min_size)
            labels = dbscan.fit_predict(sem_cluster[:, :3])
            max_label = labels.max()
            clusters = []
            for label_j in range(max_label + 1):
                indices = np.where(labels == label_j)[0]
                inst_cluster = sem_cluster[indices, :4]
                inst_cluster = np.hstack((inst_cluster, np.full((inst_cluster.shape[0], 1), label_i, dtype=np.uint32)))
                inst_cluster = np.hstack((inst_cluster, np.full((inst_cluster.shape[0], 1), inst_id, dtype=np.uint32)))
                inst_id += 1
                cluster.extend(inst_cluster)

    cluster = np.asarray(cluster)
    if save_npy:
        np.save(label_output_dir+'/'+label_name.split('/')[-1].split('.')[0]+".npy", cluster)

    return cluster

def main():
    bin_file_folder = '/Volumes/scratchdata/efimia/robotcycle_central_loop/concat_pc_bin_files'
    label_file_folder = 'central_loop_concat_cylinder8x'
    output_file_folder = 'central_loop_clustering_instance_seg_labeled'


    bin_files = [f for f in listdir(bin_file_folder) if isfile(join(bin_file_folder, f))]
    label_files = [f for f in listdir(label_file_folder) if isfile(join(label_file_folder, f))]

    bin_files.sort()
    label_files.sort()

    assert len(bin_files) == len(label_files), "bin files and label files are not same length"
    for i in range(len(bin_files)):
        print ('i = {} out of {}'.format(i, len(bin_files)))
        if i % 50 != 0: 
            continue
        bin_file = bin_files[i]
        label_file = label_files[i]
        if int(bin_file.split('/')[-1].split('.')[0]) < 17324184471833792: # for central loop
            continue
        # if int(bin_file.split('/')[-1].split('.')[0]) < 17314107088439474: # for north loop
        #     continue
        
        print('doing ' + bin_file)
        cluster = get_labels_instances(bin_file_folder + '/' + bin_file, 
                            label_file_folder + '/' + label_file, 
                            output_file_folder,
                            save_npy=True)
    
if __name__ == '__main__':
    main()
