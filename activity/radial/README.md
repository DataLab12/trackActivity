# Naive Bayes Radial Classifier (NBRC)


## Synthetic Data
* [path_generator.py](path_generator.py)

Before running the script to generate a synthetic dataset of boat paths, makes sure the directory where the csv files
are saved exists. By default, this is <em>'../datafiles/csv_files'</em> defined in the parameters of the function
<code>generate_dataset()</code>

The following should be at the bottom of the script.

```python
if __name__ == '__main__':
    sample_frequencies = [15, 25, 35, 45]
    n_processes = 5
    samples_per_proc = 100

    for cs in sample_frequencies:
        processes = []
        for i in range(n_processes):
            args_ = {'samples_per_proc': samples_per_proc, 'thid': i, 'cluster_size': cs, 'n_clusters': 50}
            x = multiprocessing.Process(target=generate_dataset, kwargs=args_)
            x.start()
            processes.append(x)
        for j in processes: j.join()

```
This uses multiprocessing to improve execution time, so the total number of samples of each type is <code> n_processes * samples_per_proc</code>.
In the code above, it is generating 4 datasets, one for each of the sample frequencies and each containing 500 samples of each path type. 
When a new process is created to call <code>generate_dataset()</code>, the arguments for the function call are defined by
<code>args_</code>.



## NBRC Model

* [activity-classifier.py](activity-classifier.py)

To train a naive bayes classifier model, the program will first extract features from the data. To do this it will convert
all points in each trajectory to polar coordinates with the origin in the center of the overhead.
The angle is the abs angle from the direction the boat is heading, so the angle will range from 0 to pi
as opposed to 0 to 2pi with traditional polar coordinates.

To train and score a model, uncomment the following code at the bottom of [svc-activity-classifier.py](activity-classifier.py)
and run the program.
```python
if __name__ == '__main__':
    n_points = 50
    point_size = 25
    train_single_model(cached=True, n_points=n_points, cluster_size=point_size)
```

To train models with different cluster sizes and path lengths to compare them,
uncomment the following code and run the program. If this is the first time this is being run,
and the scores are not being read from the file, it will take around 30 minutes to complete

```python
if __name__ == '__main__':
    fname = 'datafiles/model_scores.npy'
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            scores = np.load(f, allow_pickle=True)
    else:
        scores = compare_models()
    show_scores(scores)
```
