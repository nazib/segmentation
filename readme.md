## Unet for micrograph segmentation
Here a Unet model is implemented with Falsk api for pruction deployment. 
Here are some facts about this model:
1. The model trained on micrograph data. For quick testing and debugging
    wechanged the input image resolution 512 to 128.

2. MOdel needs to trained with batch_size=1 and more than 50 epochs

3. Prediction require to provide the directory of test images, and predicted segmentations will be in the
    "predictions" directory.

## File List and their purposes

1. model.py --- Unet model definition file.
2. train.py --- Training code for the model.
3. predict.py --- Prediction code.
4. losses.py --- loss functions for training.
5. DataReader.py  --- data loading code.
6. app.py --- REST API using Flask of the model training and prediction
---

## Training Unet
 The training requires to provide data folder, number of epochs, batch size, learning rate 
 image size and what validation percentage. The validation percentage defines what proportion of data is to be used for training and what is for validation. During testing debugging 10% images are removed for testing. The command to train the model is: 

 ```python
        python train.py --dir DATA_DIR --epochs 100 --dim 128 --batch_size 1 --validation 10
 ### Training the model
```    
Similarly the Json data for REST API for training is:
```Json
{
	"epochs": 1,
	"dimension": 128,
	"lr": 0.0001,
	"batch_size": 1,
	"dir": "micrograph_data",
	"validation": 10
}
```
The Command line for prediction:
```python
    python predict.py --dim 128 --data_dir DATA_DIR --md SAVED_MODEL_DIR --with_label False
```
Similarly the joson payload is like:

```Json
   {
	"data_dir": "test",
	"model_dir": "chkpoints",
	"with_label": "False",
	"image_dim": 128
}
```
