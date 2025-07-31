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

-AI-Powered-FMCW-Radar-False-Target-Injector/
├── ai_module/
│   ├── conf_matrix.png
│   ├── dataset.py
│   ├── model.py
│   ├── noise.py
│   ├── radar_classifier.h5
│   ├── train_classifier.py
│   ├── utils.py
│   ├── X_noisy_v2.npy
│   └── y_noisy_v2.npy
├── api/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── main.py
│   ├── radar_classifier.h5
│   └── requirements.txt
├── best_model.h5
├── docker-compose.yml
├── Dockerfile
├── f.py
├── logs
├── main.py
├── radar_simulator/
│   ├── __init__.py
│   ├── false_target_injector.py
│   ├── generate_radar_pulse.py
│   ├── inject_false_target.py
│   ├── plot_utils.py
│   ├── signal_noise_model.py
│   ├── simulate_target_return.py
│   └── utils.py
├── README.md
├── requirements copy.txt
├── requirements.txt
├── save_dataset.py
├── test_loader.py
└── v.py




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


## Disclaimer
This repository is strictly for academic, ethical, and defensive research purposes.

Usage for real-world spoofing or malicious interference with radar systems is prohibited.


## License

MIT License or Academic Non-Commercial License

## Authors
Ajinkya Patil

GitHub:[https://github.com/Ajinkya-001](https://github.com/Ajinkya-001)

Email:ajinkyapatilckl@gmail.com






