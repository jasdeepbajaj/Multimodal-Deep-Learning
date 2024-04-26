# Multimodal-Deep-Learning

## Abstract

This report presents a comprehensive study on multimodal digit classification using handwritten images and spoken audio data from the MNIST dataset. The goal was to develop a deep learning model that fuses visual and auditory inputs to accurately predict digit labels. The approach involved preprocessing multimodal data, designing a fusion model architecture, and implementing training strategies to enhance prediction capabilities. The model showed promising accuracy and competitive performance, surpassing the benchmark F1 score on a held-out test dataset.

## Introduction

The world is inherently multimodal—we perceive it through multiple senses. This project explores the challenge of digit recognition using a multimodal approach, integrating both visual and auditory data. The objective was to develop a deep learning model capable of processing and integrating these heterogeneous data sources to predict digit labels from 0 to 9.

### Problem Statement

The task involves recognizing digits presented in both written and spoken formats, necessitating a model architecture that can effectively merge these modalities.

### Solution Approach

The `FusionModel` framework integrates feature vectors from an Image Encoder (CNN) and an Audio Encoder (CNN), enhancing performance over unimodal systems.

### Comparative Analysis

This model integrates audiovisual data for a more comprehensive understanding of digits, resolving ambiguities that visual data alone might present.

### Results Overview

The FusionModel significantly outperformed baseline unimodal systems, achieving an F1 score of 0.999 on the Kaggle test dataset.

## Methods

### Data Preprocessing

Images and audio files were standardized and normalized. Images were reshaped and pixel values normalized, while audio files were amplitude normalized using sklearn's `StandardScaler`.

### Model Design

The `FusionModel` includes:
- **ImageEncoder:** A CNN pathway with dropout layers to prevent overfitting.
- **AudioEncoder:** A 1D CNN to capture temporal dynamics from audio.
- **Fusion Layer:** Integrates features from both encoders for classification.

### Model Training

Training involved batch processing, using the Adam optimizer, and cross-entropy loss function. Hyperparameters were tuned to optimize accuracy and reduce overfitting.

## Results

### Model Performance

Achieved an F1 Score of 0.999, with detailed performance metrics over training epochs.

### Comparison with Baseline Models

Demonstrated superior performance compared to unimodal systems.

### Kaggle Competition Performance

Ranked 1st in the class Kaggle competition, validating the model's effectiveness.

### Evaluation and Discussion

The results highlight the benefits of multimodal integration in improving the robustness of classification tasks.

### Clustering Analysis of Multimodal Embeddings

Used KMeans and t-SNE to analyze and visualize the embeddings, showing distinct clusters corresponding to digit labels.

## Conclusion

The FusionModel effectively integrates auditory and visual data, outperforming traditional unimodal systems and providing insights into multimodal learning.

### Reflection on Design Choices

The architecture and strategies employed were validated by the performance metrics and competitive rankings.

### Future Work

Exploration of more complex integration techniques and expanding the dataset could further enhance performance.

## References

1. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal Machine Learning: A Survey and Taxonomy. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 41(2), 423-443.
2. Ramachandram, D., & Taylor, G. W. (2017). Deep Multimodal Learning: A Survey on Recent Advances and Trends. _IEEE Signal Processing Magazine_, 34(6), 96-108.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT press.
4. O’Shea, K., & Nash, R. (2015). An Introduction to Convolutional Neural Networks. [Link](https://doi.org/10.48550/arXiv.1511.08458).
5. [More References and Resources](https://medium.com/stackademic/introduction-to-multimodal-deep-learning-c2d521d0a4cf)

