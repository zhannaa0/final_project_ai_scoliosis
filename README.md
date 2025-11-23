# AI-based Scoliosis Detection using Vision Transformer 
# Zhanna Arystanbekova, AAI-2502M

Latex report also attached in moodle.

# Introduction
This project aims to develop an artificial intelligence model for detecting scoliosis from spinal X-ray images. Scoliosis is a medical condition characterized by abnormal curvature of the spine, and early detection is essential for effective treatment. Manual diagnosis by radiologists can be subjective and time-consuming.  

By leveraging deep learning and computer vision techniques, this project automates scoliosis detection and classification into Normal and Scoliosis categories. The approach improves diagnostic speed, consistency, and accessibility in clinical settings.

# Dataset
The dataset used in this project was obtained from Kaggle: "The Vertebrae X-ray Images" by Yasser Hessein. It contains spinal X-ray images labeled into two classes: Normal and Scoliosis.
The dataset was split into 70% training set, 15% validation set and 15% test set.

# Model
A pretrained Vision Transformer (ViT-B/16) was used, based on ImageNet weights ViT\_B\_16\_Weights.IMAGENET1K\_V1.  
The model was trained and validated over 5 epochs, recording accuracy and loss for both datasets at each epoch. 

This training loop iterates through the dataset several times, performing all key steps of the learning process.
For each batch, it moves data to the device, resets accumulated gradients with optimizer.zero\_grad(), runs the model to obtain predictions, and computes the loss.
Then loss.backward() propagates the error through the network to compute gradients, and optimizer.step() updates the model’s weights.
Throughout the loop, it also counts correct predictions and accumulates the loss to calculate overall training accuracy and loss for the epoch.

This loop performs the main training procedure of the neural network.  
At each epoch the model is set to training mode with model.train() so that layers such as dropout or batch normalization behave appropriately.  
Each batch of images and labels is passed through the model to obtain predictions, and the loss between the predicted and true labels is computed by the chosen criterion.  

torch.max(outputs, 1) is used to extract the predicted class for each sample in the batch.
It finds the maximum value along dimension 1 (the class dimension) and returns both the values and their indices; the indices represent the class with the highest predicted probability.
These predicted class indices are then compared with the true labels to compute training accuracy.


# Results
Accuracy and loss through epoches:
<img width="2091" height="793" alt="image" src="https://github.com/user-attachments/assets/0770b369-7fa2-48f5-912e-6dd1060bb79a" />

On the unseen test dataset, the model achieved the following metrics:
  Accuracy: 95%
  Precision (Normal / Scoliosis): 1.00 / 0.94
  Recall (Normal / Scoliosis): 0.83 / 1.00
  F1-score (Normal / Scoliosis): 0.91 / 0.97

Overall about test performance, the model well classifies the normal and scoliosis cases. There are normal vertebra images with some curve because of graphical biases, nevertheless, model finds it accurately as normal case.  


# Conclusion
This project successfully implemented an AI-based scoliosis detection system using the Vision Transformer (ViT-B/16) model. The model achieved a test accuracy of 95\%, demonstrating the potential of transformer-based architectures for medical image analysis. The use of transfer learning allowed the system to leverage knowledge from large-scale datasets like ImageNet, enabling accurate feature extraction even with a limited number of medical images.


