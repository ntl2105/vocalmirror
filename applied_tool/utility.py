# utility.py
import os
from datetime import datetime

def log_step(message, log_file="./logs/project_steps.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def log_metric(metric_name, value, log_file="./logs/metrics.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {metric_name}: {value}\n")

def log_experiment(params, results, log_file="./logs/experiments.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] Experiment Parameters: {params}\n")
        f.write(f"[{timestamp}] Results: {results}\n\n")

