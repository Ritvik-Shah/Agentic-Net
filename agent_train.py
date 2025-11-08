import os
import subprocess
import uuid
from ollama import Ollama

OLLAMA_MODEL = "gpt-oss"
JOB_DIR = "/tmp/agent_jobs"

def make_prompt(dataset_path):
    return f"""
You are a code generator. Generate a Python script that:
- Loads the CSV dataset at '{dataset_path}'
- Trains a PyTorch MLP classifier for multi-class classification
- Splits dataset into train/val (80/20)
- Logs metrics to stdout
- Saves model to '/job/output/model.pt'
- Only uses torch, pandas, numpy
- Output ONLY Python script content
"""

def run_agent_job(dataset_path):
    os.makedirs(JOB_DIR, exist_ok=True)
    client = Ollama()
    prompt = make_prompt(dataset_path)
    
    resp = client.generate(model=OLLAMA_MODEL, prompt=prompt, max_tokens=1500)
    script_text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")

    job_id = uuid.uuid4().hex
    job_path = os.path.join(JOB_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)
    os.makedirs(os.path.join(job_path, "output"), exist_ok=True)
    
    script_path = os.path.join(job_path, "train.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)

    docker_cmd = [
        "docker", "run", "--rm",
        "--name", f"agent_job_{job_id}",
        "--cpus", "2", "--memory", "4g",
        "-v", f"{dataset_path}:/data:ro",
        "-v", f"{job_path}:/job",
        "agentic_ai_image",
        "python", "/job/train.py"
    ]

    print("Starting training container...")
    proc = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Training container exited with code {ret}")
    print("Job finished. Check", os.path.join(job_path, "output"))
    return job_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent_train.py /absolute/path/to/dataset")
        exit(1)
    run_agent_job(sys.argv[1])
