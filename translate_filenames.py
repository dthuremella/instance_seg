
from os import listdir
from os.path import isfile, join
import pickle

concat_dir = '/Volumes/scratchdata/efimia/robotcycle_central_loop/concat_pc_bin_files'
concat_files = sorted([f for f in listdir(concat_dir) if isfile(join(concat_dir, f))], key=(lambda x : int(x.split('/')[-1].split('.')[0])))
# right_dir = '../right'
# right_files = sorted([f for f in listdir(right_dir) if isfile(join(right_dir, f))], key=(lambda x : int(x.split('/')[-1].split('.')[0])))

right_txt_file = '/Volumes/scratchdata/robotcycle_exports/2024-11-08-11-14-25/timestamps/hesai/right.txt'
my_file = open(right_txt_file, "r")
data = my_file.read()
right_timestamps_list = sorted(data.split("\n"))
my_file.close()

translation = {}
for i in range(len(right_timestamps_list)):
    concat_file = concat_files[i].split('.')[0]
    right_file = right_timestamps_list[i]

    translation[concat_file] = right_file

with open('concat_to_right_translation.pkl', 'wb') as handle:
    pickle.dump(translation, handle, protocol=pickle.HIGHEST_PROTOCOL)
