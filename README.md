## Wheat Disease Detection Using Deep Learning

Overview

This project aims to detect diseases in wheat plants using deep learning and transfer learning techniques. Convolutional Neural Networks (CNNs) were implemented using pre-trained architectures such as VGG19, InceptionV3, and Xception with ImageNet weights to extract meaningful visual features from wheat leaf images. Transfer learning was chosen to accelerate training and enhance performance, leveraging the strong feature representations learned from large-scale image datasets.

# Dataset and Preprocessing

The dataset consists of labeled wheat leaf images categorized into healthy and diseased classes. Images were preprocessed to meet the input requirements of each model, including resizing, normalization, and data augmentation. Augmentation techniques were applied to improve generalization and reduce overfitting. The dataset was split into training and testing sets to ensure objective evaluation of model performance.

# Model Architecture and Training

Multiple deep learning models were trained and evaluated, including VGG19, InceptionV3, and Xception. These models were fine-tuned on the wheat dataset to adapt pre-trained features to the disease detection task. The best-performing model achieved approximately 98% training accuracy and 96% test accuracy, indicating effective learning and strong generalization to unseen images.

# Model Evaluation and Visualization

Beyond accuracy metrics, the project includes detailed visual analysis of training behavior. Training and validation accuracy and loss curves were generated to monitor convergence and detect potential overfitting or underfitting. These visualizations provide clear insight into the learning process and help validate the reliability of the trained model.

# Deployment and User Interface

The trained model was deployed using Streamlit to create an interactive web application. Users can upload wheat leaf images and receive real-time predictions. The application displays the predicted disease class along with a probability-based bar chart, allowing users to understand the model’s confidence across different classes. Screenshots of the deployed application and sample prediction outputs are included to demonstrate usability and real-world performance.

Key Features

Transfer learning using VGG19, InceptionV3, and Xception

Image preprocessing with normalization and augmentation

High classification accuracy on training and test data

Interactive Streamlit-based deployment

Probability-based prediction visualization

Training and validation accuracy and loss graphs

# Challenges

One of the main challenges was ensuring proper generalization while achieving high accuracy. Preventing overfitting required careful dataset splitting, data augmentation, and monitoring of loss curves. Additionally, maintaining consistent preprocessing between training and deployment was essential to ensure reliable real-time predictions.

# Conclusion

This project demonstrates an end-to-end deep learning pipeline for wheat disease detection, covering data preprocessing, model training, evaluation, and deployment. By combining transfer learning with an interactive web interface, the system provides an effective and practical solution for agricultural disease identification. The project serves as a strong demonstration of applied computer vision and deep learning skills suitable for academic, internship, and professional portfolio use.
