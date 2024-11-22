import copy
import numpy as np
import open3d as o3d
import pcl

# make label dicts

label_to_string = {
  0 : "unlabeled",
  1 : "outlier",
  10: "car",
  11: "bicycle",
  13: "bus",
  15: "motorcycle",
  16: "on-rails",
  18: "truck",
  20: "other-vehicle",
  30: "person",
  31: "bicyclist",
  32: "motorcyclist",
  40: "road",
  44: "parking",
  48: "sidewalk",
  49: "other-ground",
  50: "building",
  51: "fence",
  52: "other-structure",
  60: "lane-marking",
  70: "vegetation",
  71: "trunk",
  72: "terrain",
  80: "pole",
  81: "traffic-sign",
  99: "other-object",
  252: "moving-car",
  253: "moving-bicyclist",
  254: "moving-person",
  255: "moving-motorcyclist",
  256: "moving-on-rails",
  257: "moving-bus",
  258: "moving-truck",
  259: "moving-other-vehicle",
}

label_to_color = { # bgr
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
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

color_to_label = {}
for key, value in label_to_color.items():
    color_to_label[(value[2], value[1], value[0])] = key

# load ply file
ply_load = o3d.io.read_point_cloud("./concat_cylinder3d_8x/1730384558737696220.ply")
ply_load = o3d.io.read_point_cloud("./left_cylinder3d_8x/1728984944167984000.ply")

points = np.asarray(ply_load.points).astype(np.float32)
colors = np.asarray(ply_load.colors).astype(np.float32)

# make labels, check shape
print(points.shape)
print(colors.shape)
ply_labels = []
for color in colors:
    ply_labels.append(color_to_label[tuple((color*255).astype(np.int32))])
label = np.array(ply_labels)
print(label.shape[0])

# start of efimias script
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

import pdb; pdb.set_trace()


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


import pdb; pdb.set_trace()

# print(time.time()-start)
# print('*'*80)
cluster = np.array(cluster)
save=True
pub=True
if save:
    np.save("cluster.npy", cluster)
if pub:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("clustered.ply", pcd)