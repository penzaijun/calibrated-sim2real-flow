FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# gcc: torch.compile (Inductor / Triton) builds small C launcher modules at run time; the runtime image has no compiler
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir h5py==3.11.0 scipy==1.11.4
WORKDIR /workspace
COPY . /workspace
