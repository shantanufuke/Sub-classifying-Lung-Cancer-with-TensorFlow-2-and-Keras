# Sub-classifying-Lung-Cancer-with-TensorFlow-2-and-Keras
![image](https://github.com/shantanufuke/Sub-classifying-Lung-Cancer-with-TensorFlow-2-and-Keras/assets/104629474/1b2e0aff-91b4-48aa-a027-575af1df7895)

Lung cancer continues to be a significant healthcare challenge. It is the leading cause of cancer death among men and the second leading cause of cancer death among women worldwide. 
Non-small cell lung cancer represents 85 % of all lung cancer cases. 

Due to the recent availability of advanced targeted therapies, it is imperative to not only detect but also properly sub-classify non-small lung cancer into two major sub-types: squamous 
cell carcinoma and adenocarcinoma, which can be challenging at times even for experienced pathologists.

I have trained and tested a machine learning (ML) model to sub-classify non-small cell lung cancer images into squamous cell carcinoma and adenocarcinoma using TensorFlow 2 and Keras. 
I have followed below steps:

1. Prepare training, validation, and testing data directories
2. Import libraries
3. Specify paths to the training, validation, and testing dataset directories
4. Normalize images and generate batches of tensor image data for training, validation, and testing
5. Visualize samples of training images (optional)
6. Build the convolutional network model
7. Compile and train the model
8. Evaluate the model
9. Assess trained model performance on the testing dataset

I used an image dataset containing 5000 color images of lung squamous cell carcinoma and 5000 color images of lung adenocarcinoma from the LC25000 dataset, which is freely available for ML researchers.
Since I was using the Keras flow_from_directory method from the ImageDataGenerator class to generate batches of tensor image data for our model, I needed to organize the dataset into a specific directory 
structure.

The model architecture consists of three convolutional layers with increasing filter sizes (32, 64, and 128) followed by max-pooling layers to downsample the feature maps. Each convolutional layer is 
followed by a ReLU activation function to introduce non-linearity. After the third convolutional layer, a Flatten layer is used to convert the 3D feature maps into a 1D vector. The flattened output 
is then fed into two fully connected (Dense) layers with 128 neurons and a sigmoid activation function in the final layer for binary classification. A dropout layer with a dropout rate of 0.2 is added 
after the first Dense layer to prevent overfitting. The model overrides the get_config and from_config methods to enable model serialization and deserialization.

In summary, the trained model was able to classify previously unseen (testing dataset) non-small cell lung carcinoma images into squamous cell carcinoma and adenocarcinoma with 96% accuracy.












