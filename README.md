# AI-Powered FMCW Radar False Target Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Dockerized](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Made with Keras](https://img.shields.io/badge/Made%20with-Keras-D00000.svg?logo=keras)](https://keras.io/)
[![Model: 1D CNN](https://img.shields.io/badge/Model-1D%20CNN-purple.svg)](#)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)](#)

This project implements a modular system for simulating FMCW radar signals, injecting false targets, and using AI-based classification to detect spoofed radar returns. It integrates radar signal modeling, signal injection, and a deep learning-based classifier, all accessible through a FastAPI backend.

---

## Project Structure


<img width="569" height="647" alt="image" src="https://github.com/user-attachments/assets/a0230706-0a9f-46f4-be8f-387cd88a0d3c" />

---

## Features

- FMCW radar signal generation

- Signal-to-noise ratio modeling
 
- False target injection at arbitrary ranges
  
- Preprocessing and dataset creation
  
- 1D CNN classifier for spoofed vs real signal discrimination
  
- REST API to serve classification results

---

## Setup and Deployment

### Using Docker (Recommended)

```bash
cd api/
docker compose up --build
```

# Once running,access the API at 

http://localhost:8000/docs

### Manual Local Setup (Python 3.8+)

python3 -m venv venv

source venv/bin/activate

pip install -r api/requirements.txt

uvicorn main:app --reload

## API Usage

### Endpoint: /classify

POST raw signal array to classify it.

### Sample Input

```bash
{
  "signal": [0.1, 0.3, -0.2, 0.5, 0.7, ...] #any fmcv signal sample 
}
```
### Sample Output

```bash
{
  "prediction": "spoofed",
  "confidence": 0.9842
}
```

# View interactive docs at: http://localhost:8000/docs

## Model Training 
to train your own classifier
```bash
cd ai_module/
python train_classifier.py
```

### The model will be saved as:

radar_classifier.h5

You can replace the default deployed model in api/.

##  Visualizations

Confusion Matrix: ai_module/conf_matrix.png

FMCW Chirp: radar_simulator/plot_utils.py

Spoofed vs Real Sample: Use inject_false_target.py and simulate_target_return.py for visualization.

### The following plot shows how false targets are injected into the original FMCW radar echo:


<img width="1200" height="500" alt="false_target" src="https://github.com/user-attachments/assets/8b8e37ae-c0a3-433d-af72-fbfbbbca11f9" />


- **Blue**: Original radar echo without spoofing.
- 
- **Orange**: Echo with artificially injected false targets (spoofed signal).
- 
- Notice the phase distortion and interference patterns introduced by spoofing attempts.

These signal differences form the basis for training the AI classifier to detect spoofing attacks.



## Model Evaluation

### Confusion Matrix


<img width="500" height="400" alt="conf_matrix" src="https://github.com/user-attachments/assets/29adfa5a-517f-4fb9-8ee9-89e2e842b910" />

Classifiction Report

| Metric    | Class 0 (Real) | Class 1 (Spoofed) | Overall |
| --------- | -------------- | ----------------- | ------- |
| Precision | 0.97           | 0.97              | 0.97    |
| Recall    | 0.97           | 0.97              | 0.97    |
| F1-Score  | 0.97           | 0.97              | 0.97    |
| Accuracy  | –              | –                 | **97%** |

The model achieves 97% accuracy in distinguishing real vs spoofed radar signals, demonstrating

strong generalization on synthetic test data.





## Disclaimer
This repository is strictly for academic, ethical, and defensive research purposes.

Usage for real-world spoofing or malicious interference with radar systems is prohibited.


## License

MIT License or Academic Non-Commercial License

## Authors
Ajinkya Patil

GitHub:[https://github.com/Ajinkya-001](https://github.com/Ajinkya-001)

Email:ajinkyapatilckl@gmail.com








