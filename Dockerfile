FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace/ops-ass-5

RUN pip install --no-cache-dir -U pip

COPY requirements.txt /workspace/ops-ass-5/requirements.txt
RUN pip install --no-cache-dir -r /workspace/ops-ass-5/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    WANDB_SILENT=true

CMD ["bash"]

