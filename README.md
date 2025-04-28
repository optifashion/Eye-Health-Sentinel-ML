Description

Eye-Health Sentinel ML is an end-to-end eye-strain detection and mitigation platform. It includes research pipelines in Python 3.12, edge deployment via TensorFlow Lite, cloud inference with ONNX on AWS Wavelength GPU micro-instances (Graviton K3, CUDA 11), a FastAPI service, and Kubeflow orchestration. Key performance targets:

AUROC: ≥ 0.91 on blink-rate & blue-light datasets

Inference latency: < 45 ms (on-device & cloud)

Privacy: Federated continual learning with privacy safeguards

Role: AI Scientist (McGill Co-op)

EyeHealthSentinelML/
├── README.md
├── requirements.txt
├── docker/
│   ├── Dockerfile.edge  # TFLite + FastAPI
│   └── Dockerfile.cloud # ONNX + FastAPI
├── research/
│   ├── data_pipeline.py
│   ├── train.py
│   └── model.py
├── edge/
│   ├── inference.py     # TFLite runtime
│   └── prepare_tflite.py
├── cloud/
│   ├── inference.py     # ONNX runtime
│   └── prepare_onnx.py
├── api/
│   └── app.py           # FastAPI server
└── kubeflow/
    └── pipeline.py      # Kubeflow pipeline definition

    # Eye-Health Sentinel ML

## Overview
This repository implements an eye-health monitoring system with data ingestion, model training & distillation, on-device & cloud inference, and federated continual learning.

### Components
- **research/**: Data pipelines, model training (Python 3.12)
- **edge/**: TensorFlow Lite conversion & real-time inference (<45 ms)
- **cloud/**: ONNX export & GPU inference on AWS Wavelength (Graviton K3, CUDA 11)
- **api/**: FastAPI microservice for inference endpoints
- **kubeflow/**: Pipeline to automate training & deployment

## Prerequisites
- Python 3.12
- Docker & Docker Compose
- AWS CLI configured with access to Wavelength zones
- Kubeflow running on EKS or GKE
- CUDA 11 toolkit (for cloud builds)

## Setup

1. **Clone repo**
   ```bash
   git clone https://github.com/yourorg/EyeHealthSentinelML.git
   cd EyeHealthSentinelML
