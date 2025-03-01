from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pickle
import pandas as pd
import open3d as o3d
import math
import argparse

def displacement(path):
    if len(path) < 2:
        return 0
    return np.linalg.norm(path[-1] - path[0])

# Functions
def line(point0, point1):
    x0, y0 = point0
    x1, y1 = point1
    a = y0 - y1
    b = x1 - x0
    c = x0*y1 - x1*y0
    return a, b, c

def intersect(line0, line1):
    a0, b0, c0 = line0
    a1, b1, c1 = line1
    D = a0*b1 - a1*b0 # D==0 then two lines overlap
    if D==0: D = np.nan
    x = (b0*c1 - b1*c0)/D
    y = (a1*c0 - a0*c1)/D
#    x[np.isnan(x)] = np.inf
#   y[np.isnan(y)] = np.inf
    return np.array([x, y])

# p1 is np array of x,y, v1 is numpy array of velx, vely
def ttc(p1, v1, p2, v2):
    line1 = line(p1, p1+v1)
    line2 = line(p2, p2+v2)
    p_impact = intersect(line1, line2)

    # never intersect
    if np.any(np.isnan(p_impact)): return np.inf

    # calculate ttc for vehicle 1 (would be same as vehicle 2)
    dist = np.linalg.norm(p1 - p_impact)
    vel = np.linalg.norm(v1)

    return dist / vel



def iou(center1, volume1, center2, volume2):
    """ Compute the IoU of two 3D bounding boxes defined by center and volume.
    Parameters:
        center1, center2: (x, y, z) - The center coordinates of the boxes.
        volume1, volume2: (w, h, d) - The width, height, and depth of the boxes.
    Returns:
        IoU value (float)
    """
    x1, y1, z1 = center1
    w1, h1, d1 = volume1

    x2, y2, z2 = center2
    w2, h2, d2 = volume2

    # Convert to (min, max) format
    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
    z1_min, z1_max = z1 - d1 / 2, z1 + d1 / 2

    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2
    z2_min, z2_max = z2 - d2 / 2, z2 + d2 / 2

    # Compute the intersection boundaries
    x_min_inter = max(x1_min, x2_min)
    y_min_inter = max(y1_min, y2_min)
    z_min_inter = max(z1_min, z2_min)
    x_max_inter = min(x1_max, x2_max)
    y_max_inter = min(y1_max, y2_max)
    z_max_inter = min(z1_max, z2_max)

    # Compute intersection dimensions
    inter_w = max(0, x_max_inter - x_min_inter)
    inter_h = max(0, y_max_inter - y_min_inter)
    inter_d = max(0, z_max_inter - z_min_inter)
    
    inter_volume = inter_w * inter_h * inter_d
    union_volume = np.prod(volume1) + np.prod(volume2) - inter_volume

    return inter_volume / union_volume

# # r_earth = 6.378137*10^6 meters
# center_pixel = [6397, 6510]
# center_latlong = [51.749780400568554, -1.2339014542218842]
# r_earth = 6365.06 * 10**3 # at oxford
# def dy(latitude):
#     d_lat = latitude - center_latlong[0]
#     d_lat_rads = np.radians(d_lat)
#     dy = r_earth*d_lat_rads
#     return dy
# def dx(longitude):
#     lat_rads = np.radians(center_latlong[0])
#     d_lon = longitude - center_latlong[1]
#     d_lon_rads = np.radians(d_lon)
#     dx = r_earth*np.cos(lat_rads)*d_lon_rads
#     return dx
# def latlong_to_xy(lat, long):
#     return (dx(long), dy(lat))

        # # TODO remove cars with >2m center, small volume
        # # cluster_df = pd.DataFrame({
        # #     'x': cluster[:, 0], 
        # #     'y': cluster[:, 1],
        # #     'z': cluster[:, 2],
        # #     'class': cluster[:, 4],
        # #     'instance': cluster[:, 5],
        # #     })
        # is_veh = (cluster[:,4] == (1 or 4 or 5))
        # for inst in np.unique(cluster[:,5][np.where(is_veh)]):
        #     # filtered = cluster_df[(cluster_df['class'].isin([1,4,5])) & 
        #     #                       (cluster_df['instance'] == inst)]
        #     # inst_points = filtered[['x', 'y', 'z']].to_numpy()
        #     # inst_points = cluster[np.where(is_veh and is_this_inst)]

P = pyproj.Proj(proj='utm', zone=30, ellps='WGS84', preserve_units=True)
def LonLat_To_XY(Lon, Lat):
    return P(Lon, Lat)    
def XY_To_LonLat(x,y):
    return P(x, y, inverse=True) 

TRACKING_CLASSES = [
'car', 'bicycle', 'motorcycle', 'truck', "other-vehicle", 'person',
        'bicyclist', 'motorcyclist'
]
class_id_to_name = {
    1: "car",
    2: "bicycle",                                                         
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle", # including bus, "on-rails"
    6: "person", 
    7: "bicyclist", 
    8: "motorcyclist",
}

map_names = ['centerloop', 'southloop', 'northloop']
pixels = {}
latlongs = {}
m_pixels_to_xy = {}
m_xy_to_pixels = {}
###### pixel to latlong correspondences
pixels['southloop'] = np.array([[6219,2650], [5465,10596], [5279,760], [6068,6635], [8417,7056]])
latlongs['southloop'] = np.array([
                        [51.74960715841192, -1.243214497244512],
                        [51.75071111059321, -1.2245179389007195],
                        [51.75097942312646, -1.2476701027214614],
                        [51.749780400568554, -1.2339014542218842],
                        [51.746395207234215, -1.2328352056374066]])
pixels['centerloop'] = np.array([[5715,7648],[10258,6421],[5367,1863],[1563,8015]])
latlongs['centerloop'] = np.array([
                        [51.75261904374165, -1.2533070808953204],
                        [51.74590450080673, -1.2562663545776893],
                        [51.75312972870785, -1.2671889014739395],
                        [51.7587914819614, -1.252375464429015]
])

pixels['northloop'] = np.array([[9564,4956],[12601,10447],[282,4254],[13138,7986]])
latlongs['northloop'] = np.array([
                        [51.76025773543008, -1.2619623649918965],
                        [51.755687413511836, -1.2485586519323157],
                        [51.77423672354839, -1.2636557069241308],
                        [51.75486709646299, -1.2545476364368084],
])
# create the pixels-to-xy transform
for map_name in map_names:
    xy = []
    for i in latlongs[map_name]: xy.append(list(LonLat_To_XY(i[1], i[0])))
    xy = np.array(xy)
    # make it 3 dimensional with last dimension ones, to add the '+ b'
    pixels_extended = np.concatenate((pixels[map_name], np.ones((pixels[map_name].shape[0],1))), axis=1)
    xy_extended = np.concatenate((xy, np.ones((xy.shape[0],1))), axis=1)

    m_pixels_to_xy[map_name] = np.linalg.solve(pixels_extended[:3], xy[:3])
    m_xy_to_pixels[map_name] = np.linalg.solve(xy_extended[:3], pixels[map_name][:3])


def pixels_to_latlong(pixels, map_name):
    pixels_extended = np.concatenate((pixels, np.ones((pixels.shape[0],1))), axis=1)
    xy = np.matmul(pixels_extended, m_pixels_to_xy[map_name])
    converted_pixels = []
    for i in xy: 
        lonlat = XY_To_LonLat(i[0], i[1])
        latlon = [lonlat[1], lonlat[0]]
        converted_pixels.append(latlon)
    return np.array(converted_pixels)

def latlong_to_pixels(lats, longs, map_name):
    xy = []
    for i in range(len(lats)):
        lat = lats[i]
        lon = longs[i]
        xy.append(LonLat_To_XY(lon, lat))
    xy = np.array(xy)
    xy_extended = np.concatenate((xy, np.ones((xy.shape[0],1))), axis=1)
    pixels = np.matmul(xy_extended, m_xy_to_pixels[map_name])
    return pixels

def point_cloud_volume(points):
    dx = np.max(points[:,0]) - np.min(points[:,0])
    dy = np.max(points[:,1]) - np.min(points[:,1])
    dz = np.max(points[:,2]) - np.min(points[:,2])
    return np.array([dx,dy,dz])
def point_cloud_center(points):
    cx = np.mean(points[:,0])
    cy = np.mean(points[:,1])
    cz = np.mean(points[:,2])
    return np.array([cx,cy,cz])

# is the vehicle within the bounds where it's illegal to be so close to cyclist?
# law is: must maintain 2-3s stopping distance behind cyclist, 
# car can go in bike lane when bike is visible in rearview mirror (6m, 10-20ft)
def within_illegal_box(p, center, orientation, rot_offset):
    straight_orientation = np.matmul(orientation,    # same as WND to ENU orientation, but rotate extra 90deg to get from left lidar
        o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, rot_offset-np.pi/2)))                  # to straight ahead orientation

    # rotate points so that straight ahead is (1,0) in pixel space
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.arctan2(straight_orientation[1], straight_orientation[0])))
    p0 = p - center # make the center (where the bike is at) 0,0
    p0 = np.pad(p0, (0,1)) # add 0 at the end for z dim
    p_straight = np.matmul(p0, R) # make the orientation (where bike is facing) 1,0
    #uk law
    xmin = -8 # behind, 7.5px, 18m (conversion is px = 5/12*meters)
    xmax = 3 # in front, 2.5px, 6m
    ymin, ymax = -1, 1 #left, right, 0.625px, 1.5m

    if (xmin <= p_straight[0] <= xmax) and (ymin <= p_straight[1] <= ymax):
        return True
    else:
        return False

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_name", help='northloop southloop or centerloop', type=str)
    parser.add_argument("--local", action="store_true")    
    args = parser.parse_args()
    
    ############ params to change ##########################
    map_name = 'northloop' if args.map_name is None else args.map_name
    local = args.local

    short_name = map_name.split('loop')[0]
    if short_name == 'center': short_name = 'central'
    with open('{}_loop_concat_to_right_translation.pkl'.format(short_name), 'rb') as handle:
        # filename_to_timestamp = pickle.load(handle)
        bad_dict = pickle.load(handle)
    bad_keys = [k  for  k in  bad_dict.keys()]
    bad_keys.sort()
    bad_vals = [v  for  v in  bad_dict.values()]
    bad_vals.sort()
    filename_to_timestamp = {}
    start_frame = 100 if map_name != 'northloop' else 1000
    for i in range(start_frame, len(bad_keys)):
        key = bad_keys[i][:16]
        val = bad_vals[i]
        filename_to_timestamp[key] = val

    plot_cluster = False
    print_info = False

    iou_threshold = 0.25
    displacement_threshold = 0 # currently including parked cars # based on plot, 1.3m disp means not parked

    if map_name == 'northloop':
        if local:
            csv_dir = 'NorthLoop-2024-10-18'
            instance_seg_dir = 'clustering_instance_seg_labeled'
        else:
            csv_dir = '/Volumes/scratchdata/robotcycle_exports/2024-10-18-15-10-24/motion'
            instance_seg_dir = 'north_loop_clustering_instance_seg_labeled'
        xoffset = 20
        yoffset = -20
        rot_offset = 0.1*np.pi
        iou_threshold = 0.3
        persistency_threshold = 4
    
    elif map_name == 'centerloop':      # only implemented for remote
        csv_dir = '/Volumes/scratchdata/robotcycle_exports/2024-11-08-11-14-25/motion'
        instance_seg_dir = 'central_loop_clustering_instance_seg_labeled'
        xoffset = 0
        yoffset = 0
        rot_offset = -0.1*np.pi
        persistency_threshold = 4
    
    elif map_name == 'southloop':       # only implemented for local
        csv_dir = 'south_loop_csvs' 
        instance_seg_dir = 'south_loop_clustering_instance_seg_labeled'
        xoffset = 0
        yoffset = 0
        rot_offset = 0
        # in 3 frames, can catch a car that's gone 2m (because of the overlap (1m per frame, 50% overlap per frame))
        persistency_threshold = 2 
    else:
        raise ValueError('map_name must be one of: {}'.format(map_names))

    ########################################################
    print(map_name)

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

    REMOVE_LATER = 0
    # read image map
    im_map = plt.imread('oxford_' + map_name + '.png') # do for northloop and southloop too
    is_cycleonly = im_map[:,:,0].copy()
    is_cycleonly[np.where(im_map[:,:,1]+im_map[:,:,2] > 0)] = 0

    is_cyclelane = im_map[:,:,2].copy()
    is_cyclelane[np.where(im_map[:,:,0]+im_map[:,:,1] > 0)] = 0

    im_map[np.where(im_map[:,:,2] != 1)] = 0  # remove  except shared cycle_lanes and outlines
    road_lines = im_map[:,:,1]

    heatmap = np.zeros(im_map[:,:,0].shape)
    dist_heatmap = np.zeros(im_map[:,:,0].shape) + np.inf

    # read instance_seg files
    instance_seg_files = [f for f in listdir(instance_seg_dir) if isfile(join(instance_seg_dir, f))]

    vehicles_in_last_frame = {} # key is track_id, value is most recent (center, volume) where center is (x,y,z) and volume is (w,h,d)
    track_ids_per_pixel = {} # key is (x,y) tuple, value is set of track_ids (use set.add() to add)
    ttc_per_pixel = {} # key is (x,y) tuple, value is min ttc at that pixel
    num_frames_per_track_id = {}
    path_per_track_id = {}
    track_id_incr = 0
    old_center = None
    vehicles_per_frame = []

    for f in sorted(instance_seg_files, key=(lambda x : int(x.split('/')[-1].split('.')[0]))):
        filename = (instance_seg_dir + '/' + f)
        filenumber = filename.split('/')[-1].split('.')[0]
        filenumber = filenumber[:16]
        if filenumber not in filename_to_timestamp:
            print(filenumber + ' not in dict')
            continue

        timestamp = int(filename_to_timestamp[filenumber])
        print('doing file ' + filenumber + ' timestamp {}'.format(timestamp))
        latlong = interpolate(gps_timestamps, timestamp, 'gps')
        if print_info: print(latlong)
        orientation = interpolate(imu_timestamps, timestamp, 'imu') # [x, y, z]

        cluster = np.load(filename) # Nx6 where 6 is x,y,z,intensity, sem_label, inst_id

        # lidars are originally flipped between x and y, and z is opposite
        # lidar_to_imu_cluster = np.stack((cluster[:,1], cluster[:,0], -cluster[:,2]), axis=1)
        # OR.... don't flip - rotate
        # lidars are originally rotated 90deg around z axis and 180deg around x axis
        R_toIMU = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, np.pi / 2))
        lidar_to_imu_cluster = np.matmul(cluster[:,:3], R_toIMU)

        # transform cluster's xyz into 0,0,0 orientation
        R = o3d.geometry.get_rotation_matrix_from_xyz(tuple(orientation))
        rotated_points = np.matmul(lidar_to_imu_cluster, R)

        # it's supposed to be NED, 
        # vehs_top_down = np.stack((rotated_points[:,1][vehicle_ids], rotated_points[:,0][vehicle_ids]), axis=1)
        # but it's rotated to the left 90 degrees
        # which means it's WND
        # vehs_top_down = np.stack((-rotated_points[:,0][vehicle_ids], rotated_points[:,1][vehicle_ids]), axis=1)
        # but don't flip - rotate:
        R_fromWND_toENU = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, rot_offset))
        enu_points = np.matmul(rotated_points, R_fromWND_toENU)
        # R_fromWND_toNED = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
        # ned_points = np.matmul(rotated_points, R_fromWND_toNED)

        inst_id_to_centers = {}
        inst_id_to_volumes = {}
        # remove cars with >2m height center, small volume
        is_veh = np.where((cluster[:,4] == (1 or 4 or 5)))
        for inst_id in np.unique(cluster[:,5][is_veh]):
            is_inst = np.where(cluster[:,5] == inst_id)
            inst_points = enu_points[is_inst][:,:3]
            pc_volume  = point_cloud_volume(inst_points)
            pc_center = point_cloud_center(inst_points)
            if ((np.prod(pc_volume) < 1) or # less than 1m cubed
                (pc_center[-1] > 2)): # higher than 2m 
                for i in is_inst: cluster[i,4] = 0 # put it in unknown class
                if print_info: print('inst_id {} is too small {} or high {}'.format(inst_id, np.prod(pc_volume), pc_center[-1]))
            else:
                inst_id_to_centers[inst_id] = pc_center
                inst_id_to_volumes[inst_id] = pc_volume
                    
        # get vehicle_ids
        vehicle_ids = np.where((cluster[:,4] == (1 or 4 or 5)))
        if len(vehicle_ids[0]) == 0:
            print('no vehicles here')
            continue
        enu_vehs = np.concatenate((enu_points[:,:][vehicle_ids], cluster[:,3:][vehicle_ids]), axis=1)
        enu_vehs_top_down = enu_points[:,:2][vehicle_ids]
        # ned_vehs_top_down = ned_points[:,:2][vehicle_ids]

        # save indexes for each segmented instance
        inst_id_to_indexes = {}
        for inst_id in np.unique(enu_vehs[:,5]):
            inst_id_to_indexes[inst_id] = np.where(enu_vehs[:,5] == inst_id)

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
        new_center = new_center.astype(np.int64)
        new_center[0] += xoffset
        new_center[1] += yoffset

        ppoints = vehs_top_down_pixels.astype(np.int64)

        # Do Tracking
        vehicles_in_this_frame = {}
        for inst_id in inst_id_to_indexes:
            center = inst_id_to_centers[inst_id]
            volume = inst_id_to_volumes[inst_id]
            inst_points = ppoints[inst_id_to_indexes[inst_id]]
            inst_points = np.vstack([np.array(u) for u in set([tuple(p) for p in ppoints])])
            inst_points[:,0] += yoffset; inst_points[:,1] += xoffset

            # check if this object is already being tracked
            track_id = -1
            for tid_last in vehicles_in_last_frame:
                center_tid, volume_tid = vehicles_in_last_frame[tid_last]
                if iou(center, volume, center_tid, volume_tid) > iou_threshold: # threshold is 0.3
                    assert track_id == -1, 'two objects are being mapped to same track {} - increase iou threshold'.format(track_id)
                    track_id = tid_last
            # if not, assign a new track id
            if track_id == -1:
                track_id = track_id_incr
                track_id_incr += 1

            # add each vehicle in this frame
            vehicles_in_this_frame[track_id] = (center, volume)

            # uptick frame count of this vehicle and add frame to trackid
            if track_id not in num_frames_per_track_id:
                num_frames_per_track_id[track_id] = 0
                path_per_track_id[track_id] = []
            num_frames_per_track_id[track_id] += 1
            path_per_track_id[track_id].append(center)

            inst_ttc = np.inf
            inst_path = path_per_track_id[track_id]
            if len(inst_path) > 1 and old_center is not None:
                ego_vel = new_center - old_center
                inst_vel = inst_path[-1] - inst_path[-2]
                inst_ttc = ttc(p1=new_center[:2], v1=ego_vel, #ego bicycle position and velocity
                               p2=center[:2], v2=inst_vel[:2]) # track_id vehicle's position and velocity

            # mark its points as seen by this track_id
            for p in inst_points:
                key = (p[0], p[1])
                if key not in track_ids_per_pixel:
                    track_ids_per_pixel[key] = set()
                    ttc_per_pixel[key] = np.inf
                track_ids_per_pixel[key].add(track_id)
                ttc_per_pixel[key] = min(ttc_per_pixel[key], inst_ttc)

        vehicles_in_last_frame = vehicles_in_this_frame
        vehicles_per_frame.append(vehicles_in_this_frame)
        old_center = new_center
        if print_info: print('vehicles in this frame: ', vehicles_in_this_frame.keys())

        # remove duplicates
        upoints = np.vstack([np.array(u) for u in set([tuple(p) for p in ppoints])])
        upoints[:,0] += yoffset
        upoints[:,1] += xoffset
        for p in upoints:
            if is_cycleonly[p[0], p[1]]:
                heatmap[p[0], p[1]] += 1
            elif is_cyclelane[p[0], p[1]] and within_illegal_box(p, new_center, orientation, rot_offset):
                heatmap[p[0], p[1]] += 1
            dist = np.linalg.norm(p - new_center)
            dist_heatmap[p[0], p[1]] = min(dist_heatmap[p[0], p[1]], dist)
            

        if plot_cluster or REMOVE_LATER == 100: 
            # # heatmap_mask = (heatmap > 0).astype(np.int64)
            # # im_map[:,:,0] = heatmap_mask
            # # im_map[new_center[0], new_center[1]] = [1,0,0,1]
            # # fig, ax = plt.subplots()
            # # ax.imshow(im_map)
            # # ax.scatter(vehs_top_down_pixels[:,1]+xoffset, vehs_top_down_pixels[:,0]+yoffset, s=1, c='r')
            # # # ax.set_xlim((new_center[1]-100,new_center[0]+100))
            # # # ax.set_ylim((new_center[1]-100,new_center[0]+100))
            # # plt.show()

            
            # fig, ax = plt.subplots()
            # ax.axis('off')  # command for hiding the axis. 
            # ps = enu_points[np.where(cluster[:,4] != 0)]
            # cs = cluster[np.where(cluster[:,4] != 0)]
            # ax.scatter(np.flip(ps[:,0]), np.flip(ps[:,1]), s=(50/np.flip(cs[:,5])), edgecolors='k', linewidths=0.01, cmap='magma_r', c=np.flip(cs[:,5]))
            # # ax.set_xlim((-25,25))
            # # ax.set_ylim((-25,25))
            # plt.show()


            cluster_to_plot = cluster[:,:3]
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
        
            import pdb; pdb.set_trace()
        
        # REMOVE_LATER = REMOVE_LATER + 1
        # if REMOVE_LATER == 100:
        #     import pdb; pdb.set_trace()
            
    # heatmap_mask = (heatmap > 0).astype(np.int64)
    # im_map[:,:,0] = heatmap_mask
    if print_info:
        with open('{}_path_per_track_id.pkl'.format(map_name), 'wb') as handle:
            pickle.dump(path_per_track_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    persisting_tracks = [k for k in num_frames_per_track_id if 
            (num_frames_per_track_id[k] > persistency_threshold   # persistency threshold = 7 frames for northloop
            and displacement(path_per_track_id[k]) > displacement_threshold)] # be sure car isn't parked (2m threshold)

    tracked_occupancy_heatmap = np.zeros(im_map[:,:,0].shape)
    ttc_heatmap = np.zeros(im_map[:,:,0].shape) + np.inf
    for key in track_ids_per_pixel:
        all_track_ids = track_ids_per_pixel[key] # set of track_ids
        real_track_ids = all_track_ids.intersection(persisting_tracks) 
        tracked_occupancy_heatmap[key[0], key[1]] = len(real_track_ids)

        ttc_heatmap[key[0], key[1]] = ttc_per_pixel[key]
    
    if displacement_threshold > 0:
        map_name = '{}_noparked'.format(map_name)

    with open('{}_tracked_occupancy_heatmap.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(tracked_occupancy_heatmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('{}_ttc_heatmap.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(ttc_heatmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('{}_heatmap.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(heatmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('{}_dist_heatmap.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(dist_heatmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('{}_im_map.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(im_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    num_vehicles_per_frame = []
    num_moving_vehicles_per_frame = []
    for vehicles in vehicles_per_frame:
        num_vehicles_per_frame.append(len(vehicles))
        num_moving_vehicles_per_frame.append(len(set(vehicles.keys()).intersection(persisting_tracks)))

    print('average number of vehicles per frame (possible spurious detections): {}'.format(
            np.mean(np.array(num_vehicles_per_frame))))
    print('average number of vehicles per frame (persisting tracks that are not parked): {}'.format(
            np.mean(np.array(num_moving_vehicles_per_frame))))


      


if __name__ == '__main__':
    main()