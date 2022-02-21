# Paper

* G. Strauch, J. Lin, J. Tešić, ``Overhead Projection Approach For Multi-Camera Vessel Activity Recognition'' IEEE BigData 2021. [paper](https://datalab12.github.io/documents/2021BigDataREU_activity.pdf) [slides](https://datalab12.github.io/documents/2021BigDataREU_activityslides.pdf)

## About

The last piece in integrated end-to-end system that analyzes multi-camera ship video feed, localizes maritime vessels in the video feed, identifies the maritime vessel over multiple cameras, maps the vessel track onto an overhead plane is to identify anomalous vessel movement around the ship. We focus on a specific activity detection approach in maritime vessel overhead tracks and on synthetic data generation to realistically model maritime boat movements around onboard ship cameras using real-world examples. We propose and compare three novel modes of trajectory analysis and activity classification, using Computing with Words [CWW](baseline/computing-with-words.py), a Markov trajectory feature classifier [MTFC](baseline/hmm.py), and Na¨ıve Bayes Radial Classifier [NBRC](radial/activity-classifier.py) to detect the activity of vessel approaching the ship, vessel chasing another vessel, and vessel circling around the ship.

* [baseline](baseline/README.md) - baseline approaches using feature extraction, markov models, and computing with words 
* [radial](radial/README.md) - latest code to generate synthetic data and to detect activities using radial approach

# Generating Synthetic Data-Set
The synthetic data consists of 4 different types of paths: circling, approaching, chase, and random. Each
is generated with a specified number of points to generate (n_clusters), and number of pixels the trajectory moves 
between each point (cluster_size). The total number of points produced in the csv files may be less than n_clusters
because it is possible for a point that is outside the domain to be generated and will not be recorded in the final csv.
Also, the distance traveled between each point is also not exactly the same since each pixel level move is determined using a 
probability distribution.

To generate synthetic datasets, specify the variables for n_clusters and cluster_size if needed and run the following
commands in the terminal.
This should take approximately 5 minutes to generate the 500 samples for each path type.

```bash
>> mkdir "datafiles/csv_files"
>> python radial/path_generator.py
```



