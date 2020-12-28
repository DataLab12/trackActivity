import sys
import os
import re
import subprocess

rootdir = "./MID" # IPATCH Mid-Level Challenge

for subdir, dirs, files in os.walk(rootdir):
	vidfile_found = False
	annofile_found = False
	already_processed = False
	vidpath = ""
	annopath = ""
	for file in files:
		text = os.path.join(subdir, file)
		vid_regex = "(Sc[0-9][a-z]*_Tk[0-9].*mp4$)"
		anno_regex = "(Sc[0-9][a-z]*_Tk[0-9].*/Sc[0-9][a-z]*_Tk[0-9].*json$)"
		merge_regex = "(Sc[0-9][a-z]*_Tk[0-9].*_merged.json$)"
		video = re.search(vid_regex, text)
		annos = re.search(anno_regex, text)
		merged = re.search(merge_regex, text)
		if video is not None:
			vidfile_found = True
			vidpath = os.path.join(rootdir, video.group())
		if annos is not None:
			print (annos.group())
			annofile_found = True
			annopath = os.path.join(rootdir, annos.group())
		if merged is not None:
			already_processed = True

	if vidfile_found and annofile_found:
		if not already_processed:
			output_dest = annopath[:-5] + "_merged.json"
			subprocess.call(" python tracking_with_activities.py " 
			             + vidpath + " " + annopath + " " + output_dest, shell=True)
		else:
			print ("Skipped " + vidpath + "; already processed")