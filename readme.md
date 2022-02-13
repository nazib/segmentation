Here a Unet model is implemented with Falsk api for pruction deployment. 
Here are some facts about this model:
    1. The model trained on micrograph data. For quick testing and debugging
    we changed the input image resolution 512 to 128.
    2. MOdel needs to trained with batch_size=1 and more than 50 epochs
    3. Prediction require to provide the directory of test images, and predicted segmentations will be in the
    "predictions" directory.

File Descriptions:
     1. 