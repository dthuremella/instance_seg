from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import pickle
import pandas as pd
import open3d as o3d
import math

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
    return dx * dy * dz
def point_cloud_center(points):
    cx = np.mean(points[:,0])
    cy = np.mean(points[:,1])
    cz = np.mean(points[:,2])
    return np.array([cx,cy,cz])

def main():
    ############ params to change ##########################
    with open('central_loop_concat_to_right_translation.pkl', 'rb') as handle:
        filename_to_timestamp = pickle.load(handle)
    #     bad_dict = pickle.load(handle)
    # bad_keys = [k  for  k in  bad_dict.keys()]
    # bad_keys.sort()
    # bad_vals = [v  for  v in  bad_dict.values()]
    # bad_vals.sort()
    # filename_to_timestamp = {}
    # for i in range(400, len(bad_keys)):
    #     key = bad_keys[i]
    #     val = bad_vals[i]
    #     filename_to_timestamp[key] = val

    map_name = 'centerloop'

    csv_dir = '/Volumes/scratchdata/robotcycle_exports/2024-11-08-11-14-25/motion' #/Volumes/scratchdata/robotcycle_exports/2024-{etc}/motion

    instance_seg_dir = 'central_loop_clustering_instance_seg_labeled'

    plot_cluster = False

    # offset = 20 # calculated by eye for map, for x and y (20 for northloop)
    xoffset = 0   # by eye, 20 for northloop
    yoffset = 0     # by eye, -20 for northloop
    rot_offset = -0.1*np.pi   # by eye, 0.1*np.pi for north
    print_info = False

    ########################################################

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
    im_map[np.where(im_map[:,:,2] != 1)] = 0  # remove  except cycle_lanes and outlines
    is_cyclelane = im_map.copy()
    is_cyclelane[np.where(im_map[:,:,2]+im_map[:,:,1]+im_map[:,:,0] == 3)] = 0 #remove outlines
    is_cyclelane = is_cyclelane[:,:,2]
    road_lines = im_map[:,:,1]
    heatmap = np.zeros(im_map[:,:,0].shape)
    heatmap_points = []

    # read instance_seg files
    instance_seg_files = [f for f in listdir(instance_seg_dir) if isfile(join(instance_seg_dir, f))]

    for f in sorted(instance_seg_files, key=(lambda x : int(x.split('/')[-1].split('.')[0]))):
        filename = (instance_seg_dir + '/' + f)
        filenumber = filename.split('/')[-1].split('.')[0]
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

        # remove cars with >2m height center, small volume
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
        if len(vehicle_ids[0]) == 0:
            print('no vehicles here')
            continue
        enu_vehs_top_down = enu_points[:,:2][vehicle_ids]
        # ned_vehs_top_down = ned_points[:,:2][vehicle_ids]

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
        upoints[:,0] += yoffset
        upoints[:,1] += xoffset
        for p in upoints:
            if is_cyclelane[p[0], p[1]]:
                heatmap[p[0], p[1]] += 1
                heatmap_points.append([p[0], p[1]])

        if plot_cluster or REMOVE_LATER == 100: 
            heatmap_mask = (heatmap > 0).astype(np.int64)
            im_map[:,:,0] = heatmap_mask
            new_center = new_center.astype(np.int64)
            im_map[new_center[0]+xoffset, new_center[1]+yoffset] = [1,0,0,1]
            fig, ax = plt.subplots()
            ax.imshow(im_map)
            ax.scatter(vehs_top_down_pixels[:,1]+xoffset, vehs_top_down_pixels[:,0]+yoffset, s=1, c='r')
            # ax.set_xlim((new_center[1]-100,new_center[0]+100))
            # ax.set_ylim((new_center[1]-100,new_center[0]+100))
            plt.show()

            import pdb; pdb.set_trace()
            
            # fig, ax = plt.subplots()
            # ax.scatter(enu_vehs_top_down[:,0], enu_vehs_top_down[:,1], s=1)
            # # ax.set_xlim((-25,25))
            # # ax.set_ylim((-25,25))
            # plt.show()

            # cluster_to_plot = enu_points
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # xs = cluster_to_plot[:,0][np.where(cluster[:,4] == 1)]
            # ys = cluster_to_plot[:,1][np.where(cluster[:,4] == 1)]
            # zs = cluster_to_plot[:,2][np.where(cluster[:,4] == 1)]
            # colors = cluster[:,5][np.where(cluster[:,4] == 1)]
            # ax.scatter(xs, ys, zs, c=colors, s=1)
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.show()
        
        
        # REMOVE_LATER = REMOVE_LATER + 1
        # if REMOVE_LATER == 100:
        #     import pdb; pdb.set_trace()
            
    # heatmap_mask = (heatmap > 0).astype(np.int64)
    # im_map[:,:,0] = heatmap_mask

    with open('{}_heatmap.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(heatmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}_im_map.pkl'.format(map_name), 'wb') as handle:
        pickle.dump(im_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

      


if __name__ == '__main__':
    main()