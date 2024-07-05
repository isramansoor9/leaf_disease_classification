# Leaf Disease Classification

## Overview

This project focuses on classifying leaf diseases into 38 different classes using deep learning techniques. The classes include diseases such as Apple___Apple_scab, Apple___Black_rot, and others across various plant species. The model is trained using TensorFlow/Keras and evaluated on a dataset sourced from Kaggle "https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset".

## Dataset

The dataset used for training and evaluation consists of images collected from various sources, encompassing a wide range of leaf diseases and healthy states. The dataset is organized into training, validation, and test sets, allowing for rigorous model evaluation.

## Model Architecture

The classification model is built using TensorFlow and Keras, leveraging convolutional neural networks (CNNs) for image feature extraction and classification. The architecture includes multiple convolutional layers followed by max-pooling, batch normalization, and dense layers with softmax activation for multi-class classification.

## Training

The model is trained on a GPU-enabled environment to expedite training times, utilizing techniques like data augmentation to enhance model generalization. Training progress is monitored using validation accuracy and loss metrics, with early stopping and model checkpointing to prevent overfitting and retain the best performing model.

## Evaluation

Model performance is evaluated using various metrics such as accuracy and loss on a separate test set. Additionally, predictions on sample images are visualized to assess the model's ability to correctly classify different leaf diseases.

## Results

Epoch 1/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 56s 30ms/step - accuracy: 0.3701 - loss: 2.2364 - val_accuracy: 0.7591 - val_loss: 0.7558

Epoch 2/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 28ms/step - accuracy: 0.8122 - loss: 0.5988 - val_accuracy: 0.8804 - val_loss: 0.3644

Epoch 3/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.8853 - loss: 0.3553 - val_accuracy: 0.8893 - val_loss: 0.3428

Epoch 4/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9104 - loss: 0.2707 - val_accuracy: 0.9060 - val_loss: 0.2961

Epoch 5/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 51s 29ms/step - accuracy: 0.9272 - loss: 0.2142 - val_accuracy: 0.9187 - val_loss: 0.2559

Epoch 6/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 51s 29ms/step - accuracy: 0.9421 - loss: 0.1772 - val_accuracy: 0.9314 - val_loss: 0.2174

Epoch 7/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9467 - loss: 0.1575 - val_accuracy: 0.9035 - val_loss: 0.3271

Epoch 8/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 28ms/step - accuracy: 0.9545 - loss: 0.1336 - val_accuracy: 0.9314 - val_loss: 0.2427

Epoch 9/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 51s 29ms/step - accuracy: 0.9558 - loss: 0.1308 - val_accuracy: 0.9272 - val_loss: 0.2602

Epoch 10/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9617 - loss: 0.1124 - val_accuracy: 0.9258 - val_loss: 0.2679

Epoch 11/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9665 - loss: 0.0960 - val_accuracy: 0.9414 - val_loss: 0.2128

Epoch 12/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 82s 29ms/step - accuracy: 0.9671 - loss: 0.0941 - val_accuracy: 0.9482 - val_loss: 0.1823

Epoch 13/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 51s 29ms/step - accuracy: 0.9713 - loss: 0.0813 - val_accuracy: 0.9439 - val_loss: 0.2093

Epoch 14/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9727 - loss: 0.0807 - val_accuracy: 0.9288 - val_loss: 0.2729

Epoch 15/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 51s 29ms/step - accuracy: 0.9724 - loss: 0.0793 - val_accuracy: 0.9418 - val_loss: 0.2198

Epoch 16/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 28ms/step - accuracy: 0.9772 - loss: 0.0678 - val_accuracy: 0.9481 - val_loss: 0.2027

Epoch 17/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9772 - loss: 0.0665 - val_accuracy: 0.9451 - val_loss: 0.2075

Epoch 18/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 51s 29ms/step - accuracy: 0.9738 - loss: 0.0791 - val_accuracy: 0.9480 - val_loss: 0.1920

Epoch 19/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9767 - loss: 0.0697 - val_accuracy: 0.9353 - val_loss: 0.2686

Epoch 20/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 51s 29ms/step - accuracy: 0.9800 - loss: 0.0591 - val_accuracy: 0.9482 - val_loss: 0.1944

Epoch 21/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9799 - loss: 0.0602 - val_accuracy: 0.9548 - val_loss: 0.1917

Epoch 22/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9829 - loss: 0.0520 - val_accuracy: 0.9491 - val_loss: 0.2030

Epoch 23/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9809 - loss: 0.0575 - val_accuracy: 0.9547 - val_loss: 0.1815

Epoch 24/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 29ms/step - accuracy: 0.9802 - loss: 0.0593 - val_accuracy: 0.9458 - val_loss: 0.2286

Epoch 25/25
1758/1758 ━━━━━━━━━━━━━━━━━━━━ 50s 28ms/step - accuracy: 0.9805 - loss: 0.0578 - val_accuracy: 0.9513 - val_loss: 0.2154

### Test accuracy: 0.9504894018173218

![image](https://github.com/isramansoor9/leaf_disease_classification/assets/135416295/f9fbb975-fed5-4a25-a216-05d4de338457)

![image](https://github.com/isramansoor9/leaf_disease_classification/assets/135416295/c04af7a6-3491-4c4b-b7ab-786b5e2eb3ee)



## Usage
To run the project locally, ensure you have TensorFlow and other dependencies installed. You can train the model by running the training script and evaluate its performance using the provided evaluation scripts. Sample code for predicting disease classes from new images is also included.

## Dependencies

TensorFlow
NumPy
Matplotlib

## Future Improvements
Future enhancements to the project may include deploying the model as a web application or integrating it with IoT devices for real-time disease detection in agricultural settings.
