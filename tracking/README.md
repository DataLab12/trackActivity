# Multi-Camera Multi-Target Tracking

CenterNet Deepsort is trained on an original model using information from the IPATCH video dataset. 
Secondary models have been trained using AIinTheSky.


* [CenterNet Deep Sort](CenterNetDeepSort.md) - how to setup and run the [original code](https://github.com/kimyoon-young/centerNet-deep-sort)

* [CNDS Tools](CNDS Tools) - Data Lab modifications and contributions to CenterNet Deep Sort. 
  Input: activity-annotated maritime scenarions
  Output: JSON annotation that relate CenterNettDeepSort activity to frames, bounding boxes, and trajectory IDs. 
* Changes to centernet include distance forumula calculations as well as additions to bounding box tracking and small changes to generate console output and CSV output. As well as tweaking changes to detection and Detector factory.

* Running without a model path

  * If you have a completed and annotated COCO-JSON dataset and are attempting to run a new model:
  * Update your detector factory for centernet by adding your training data paths to the detector_factory.py file in the centernet 	  library after downloading centernet. Make sure to create a data file in the data directory that is the same name as the path added to detector_factory.py
  * NEXT STEP: 
    ```bash
      >>main.py ctdet --exp_id 'your ID here' --batch_size 'your batch #' --master_batch 15 --lr 'desired learning rate' --gpus 0
    ```

  * To continue training on an already existing model:

    ```bash 
    >>test.py ctdet --exp_id 'your ID here' --keep_res --load_model 'model_path here'
    ```
