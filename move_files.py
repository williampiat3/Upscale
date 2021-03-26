from os import listdir
from os.path import isfile, join
import os
import shutil
import time

def move(path_departure,path_arrival):

	shutil.move(path_departure, path_arrival)


def list_files_in_folder(mypath):
	return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def change_ssd_to_dd():
	mypath = "/media/will/227E8A467E8A1329/Users/willi/Documents/MX/results/"
	path_arrival = "/home/will/Téléchargements/MX/"
	list_files = list_files_in_folder(mypath)
		# time.sleep(1000)
	for file in list_files:
		move(mypath+file,path_arrival+file)

def change_todo_done():
	mypath = "/media/will/227E8A467E8A1329/Users/willi/Documents/MX/frames/class_1/"
	path_arrival = "/home/will/Téléchargements/MX/"
	# list_files = list_files_in_folder(mypath)
	list_files_2 = list_files_in_folder(path_arrival)
	list_files = list_files_in_folder(mypath)
		# time.sleep(1000)
	for file in list_files_2:
		if file in list_files:
			move(mypath+file,"/media/will/227E8A467E8A1329/Users/willi/Documents/MX/done/"+file)

if __name__ == "__main__":
	change_ssd_to_dd()
	change_todo_done()