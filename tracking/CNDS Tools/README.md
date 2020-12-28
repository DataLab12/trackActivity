# Annotation Merging

Applies CenterNet Deep Sort to activity-annotated maritime scenarios to yield JSON annotations that relate activity categories to frames, bounding box coordinates, and trajectory IDs.

```bash
conda activate [my_envornment]
cp *.py [path_to/centerNet-deep-sort]
cd [path_to/centerNet-deep-sort]
```

## tracking_with_activities.py
Runs this [demo](https://github.com/kimyoon-young/centerNet-deep-sort/blob/master/demo_centernet_deepsort.py) with added code to read in the activity annotations and export a JSON file in the following format:
```
[
 {
  "frameNo",
  "activities": [
  	{
  	  "activityID"
  	}
  ],
  "tracks": [
  	{
  	  "trackID",
  	  "bbox": {"x1", "y1", "x2", "y2"}
  	}
  ]
 }
]
```

Before running, the paths for CenterNet, detection model, and default video must be manually changed. 

## merge_all.py
Performs an OS walk of a specified directory containting all videos and corresponding annotation files. 

The "root" for the OS walk may need to be changed before running.

The video and annotation files are expected to be within folders with the naming scheme:

Sc%_Tk%_CAM%%_h264-1/

    Sc%_Tk%_CAM%%_h264-1.mp4

    Sc%_Tk%_CAM%%_h264-1.json

Where the % are replaced by scenario#, take#, cam#, e.g. Sc1_Tk1_CAM12_h264-1

After running, each folder will contain the merged annotation file: Sc%_Tk%_CAM%%_h264-1_merged.json
