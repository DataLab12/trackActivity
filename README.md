# Overhead Object Tracking to Activity Recognition
* Maritime object detection, identification, and tracking
* Multi-camera overhead track projection and merge
* Activity recognition from tracks 

## People
* [Jelena Tešić](jtesic.github.io), Assistant Professor, Computer Science
* [George Strauch](https://george-strauch.github.io/), B.Sc. Fall 2021
* [Jiajian Lin](https://www.linkedin.com/in/jaxlin/), Summer 2020 REU student from UCLA
* [Sebastian Santana](cross_flag.github.io), B.Sc. Fall 2020
* [Alan Turner](mailto:alan@txstate.edu), M.Sc. Spring 2020

## Publications 

* G. Strauch, J. Lin, J. Tešić, ``Overhead Projection Approach For Multi-Camera Vessel Activity Recognition'' IEEE BigData 2021. [paper](https://datalab12.github.io/documents/2021BigDataREU_activity.pdf) [slides](https://datalab12.github.io/documents/2021BigDataREU_activityslides.pdf)

* J. Tešić, D. Tamir et. al, ``Computing with Words in Maritime Piracy and Attack Detection Systems'', HCII 2020, [pdf](https://link.springer.com/chapter/10.1007/978-3-030-50439-7_30)

## Project 

This project is continuation of the [Aerial](https://github.com/DataLab12/AerialPipeline) project on localizing and identifying objects in maritime videos. First, we annotate activities in the videos: follow, speed up, loiter, seperate, and merge. Then we identify and track maritime vehicles using [CenterNetDeepSort](tracking/Util) package.
Next, we automatically map the tracks to system-centric [overhead](overhead) view for multiple cameras in the system. We merge the [tracks](tracking) from the same vehicle accross different cameras, and activity recognition reduces to describing and identifying tracks of the maritime vehicles. Next, we propose feature-based and location-based approach to classify tracks into [activities](trackActivity), and augment the training data by creating more synthetic tracks using the same paradigm. 

## About 

Recent rise of maritime piracy and attacks on transportation ships has cost the global economy several billion dollars.  To counter the threat, researchers have proposed agent-driven modeling to capture dynamics of maritime transportation system, and to score the potential of a range of piracy countermeasures. Combining information from on-board sensors and cameras with intelligence from external sources for early piracy threat detection has shown promising re-sults but lacks real-time update for situational context.  Such systems can benefit from the early warnings such as “a boat is approaching the ship and accelerating”, “a boat is circling the ship,” or “two boats are diverging close to the ship”.  Existing on-board cameras capture these activities, but there is no automated processing of this type of  patterns to inform early warning system.  Visual data feed is used by crew only after they have been alerted of possible attack: camera sensors are inexpensive but transforming the incoming video data streams into actionable items still demands expensive human processing.  We propose to apply the recent advances in deep learning to design and train algorithms to localize, identify, and track small maritime objects under varying conditions (e.g. snowstorm, high glare, night), and several methods to identify threating activities where lack of training data prohibits the use of deep learning. 

## Object Identification and Tracking

[Tracking](tracking) module uses CenterNetDeepsort framework for object identification and tracking with specialized model trained for small maritime objects.  We summarize the modeling output on test data in JSON format (vehicle ID) and CSV format that contains coordinate and distance from camera for each vehicle ID. [Util](tracking/Util) folder contains the scripts to run CenterNetDeepSort and CSV converters. Script [centernet_deepsort2_OG.py](tracking/Util/centernet_deepsort2_OG.py), instructions found in [overhead](overhead) produces a video  with all recognized and tracked vehicles visually labeled, and CSV file with all relevant ID's, activity, and distance data.

## Activity Annotation

Activities were logged using the [VIA4JSON](https://github.com/DataLab12/VIA-JSON) repository. Each individual scene is composed of at most four distinct cameras. Camera twelve faces the stem of the boat with cameras ten, eleven, and fourteen facing starboard and slightly towards the bow. Each camera records its own length of footage which is uploaded to Via-JSON and then analysed by hand. Boat activity is logged in 10s intervals, and saved as JSON file for each of the invidividual cameras. Final posprocessing step compiles the data into single JSON per event, in JSON COCO format with Activity ID, video ID, start time, stop time. 

## Multi camera overhead projection

[Overhead](overhead) module merges overhead projection from multiple cameras. [Clicks.py](overhead/clicks.py) script outputs coordinates from each of the responding cameras: two horizon corner points and three points in the bottom middle left and right of the frame. In order to run this program from yourself first you will have to create a basic overhead CADfile similar to: [ship2b3](overhead/ship2b3.png)

Then you can use [Clicks.py](overhead/clicks.py) to match points from your cadfile camera angles to the corresponding angles of your video. Once you have your points you run [json_homographizer.py](overhead/json_homographizer.py) and then merge all of your corresponding CSV files together. This will output a singular CSV file that can be used as an input for [drawfromcsv.py](overhead/drawfromcsv.py) which will create a png image showing the vehicle boat paths and a small video output showing the paths being drawn.


## Activity identification and recognition 

[Activity](activity) - The last piece in integrated end-to-end system that analyzes multi-camera ship video feed, localizes maritime vessels in the video feed, identifies the maritime vessel over multiple cameras, maps the vessel track onto an overhead plane is to identify anomalous vessel movement around the ship. We focus on a specific activity detection approach in maritime vessel overhead tracks and on synthetic data generation to realistically model maritime boat movements around onboard ship cameras using real-world examples. We propose and compare three novel modes of trajectory analysis and activity classification, using Computing with Words [CWW](activity/baseline/computing-with-words.py), a Markov trajectory feature classifier [MTFC](activity/baseline/hmm.py), and Na¨ıve Bayes Radial Classifier [NBRC](activity/radial/activity-classifier.py) to detect the activity of vessel approaching the ship, vessel chasing another vessel, and vessel circling around the ship.
 

