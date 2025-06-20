# Personnal Professional Project: Facial Recognition System

## Overview

This project aims to build an accurate and efficient facial recognition system. We experimented with multiple traditional and deep learning-based approaches to identify the most suitable method for our needs.

## Approaches Explored

Over the course of development, we tried several methods for facial detection and recognition:

- **CNN + HOG (Histogram of Oriented Gradients)**  
  Combined deep learning-based classification with traditional feature extraction. Provided decent performance but lacked robustness in varying lighting and poses.

- **HOG + SVM**  
  A classical pipeline using HOG features and a Support Vector Machine classifier. While lightweight and easy to implement, it underperformed on complex, real-world data.

- **OpenFace**  
  A deep learning model that uses facial embeddings and metric learning. It showed moderate accuracy but struggled with subtle facial differences and edge cases.

- **SVM + LBP (Local Binary Patterns)**  
  A texture-based approach combined with SVM. It worked well for controlled environments but did not generalize well to more diverse datasets.

- **Haar Cascade + CNN**  
  Used Haar Cascades for face detection followed by CNN-based classification. Detection was fast, but the system was sensitive to occlusions and required well-aligned faces.

## Final Approach: DeepFace

After thorough testing and performance evaluation, we chose **DeepFace** as the final solution. DeepFace provides an easy-to-use API with access to powerful pre-trained models (e.g., VGG-Face, Facenet, ArcFace), offering high accuracy and robustness.

### Why DeepFace?

- Outperformed all custom-built pipelines in terms of **recognition accuracy**
- Combines face detection, alignment, and recognition in one framework
- Easily supports face verification and identification tasks
- Flexible backend selection with support for multiple models


