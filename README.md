# UNet

## Requirements
#### 1.) Ubuntu </br>
#### 2.) Make sure you have python3 installed , which should be there by default </br>
#### 3.) Dataset should contain two folders , one named "images" containing the images and second should be named "masks" , containing the ground truth mask labels corresponding to each image.

## Usage
#### 1.) Clone the repository </br>
#### 2.) Enter the following in command line </br>
```
cd UNet
bash ./packages.sh
```
#### 3.) For training , </br>
```
python3 train.py
Enter data path
"Enter the path to your dataset" 
```
#### 4.) After training , the following prompt will show , asking you to name the newly trained model
```
Select name for the model
"model name of your choice"
```
#### 5.) Model will be saved in Models folder , which is created when you run train.py. From there , you can use that model for prediction
#### 6.) For prediction , run
```
python3 predict.py
```
#### 7.) The following prompt will show
```
Enter image path
```
#### 8.) Here, you will enter the path to your test image
#### 9.) The predicted mask will be shown in new window and also will be saved in the "Predictions" folder

