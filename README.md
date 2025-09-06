# Classification Model for Satellite Image Segmentation

A machine learning project focused on satellite image segmentation with emphasis on rare-class detection using crowd-sourced data and human-reported information from internet sources.

## 🌍 Project Overview

This project develops a specialized segmentation method for identifying rare geographical features in satellite imagery, leveraging human-reported data from news articles and blogs as training data points. The approach is particularly useful for detecting uncommon but critical geographical phenomena that traditional automated methods might miss.

## 🎯 Problem Statement

The project addresses the challenge of rare-class segmentation in satellite imagery by:
- Utilizing human-reported data from internet news and blogs as ground truth
- Implementing custom loss functions that combine standard segmentation losses with crowd-confidence weighting
- Focusing on rare geographical features that are difficult to detect with conventional methods

## 🛠️ Technical Approach

### Custom Loss Function
The core innovation lies in a hybrid loss function that:
- Combines standard segmentation losses (Focal Loss, Dice Loss)
- Incorporates crowd-confidence weighting
- Prioritizes rare-class detection in the loss calculation

### Key Features
- **Satellite Image Processing**: Advanced image segmentation techniques
- **Crowd-Sourced Learning**: Integration of human-reported data
- **Edge Detection**: Specialized algorithms for boundary identification
- **Multi-class Segmentation**: Support for various geographical feature types

## 📁 Repository Structure

```
├── optimal_satellite_segmentation.py  # Main segmentation algorithm
├── import cv2.py                      # Image processing utilities
├── tyu.py                            # Additional processing functions
├── wer.py                            # Supplementary algorithms
├── images/                           # Source satellite images
├── masks/                            # Ground truth segmentation masks
├── test_images/                      # Test dataset
├── example_edges.png                 # Edge detection example
└── Bharatiya Antariksh Hackathon 2025 Idea Submission PPT.pdf
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tensorflow  # or pytorch
```

### Usage

1. **Basic Segmentation**:
```python
python optimal_satellite_segmentation.py
```

2. **Image Processing**:
```python
python import\ cv2.py
```

## 🔬 Methodology

1. **Data Collection**: Gather satellite images and corresponding human-reported observations
2. **Preprocessing**: Apply edge detection and image enhancement techniques
3. **Model Training**: Use custom loss function with crowd-confidence weighting
4. **Validation**: Test on rare geographical features
5. **Post-processing**: Refine segmentation masks

## 🏆 Competition Context

This project was developed for the **Bharatiya Antariksh Hackathon 2025**, focusing on innovative applications of satellite imagery analysis for geographical feature detection.

## 📊 Results

The model demonstrates improved performance in:
- Rare geological feature detection
- Boundary precision for uncommon land formations
- Integration of human expertise with automated analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Algorithm improvements
- New rare-class detection methods
- Enhanced crowd-sourcing integration
- Performance optimizations

## 📝 License

This project is open source and available under the MIT License.

## 📞 Contact

For questions or collaboration opportunities, please reach out through GitHub issues or discussions.

---

**Note**: This project represents ongoing research in satellite image analysis and rare-class segmentation. Results and methodologies are continuously being refined and improved.