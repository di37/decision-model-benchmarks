FROM vllm/vllm-openai:v0.19.0

RUN python3 -m pip install --no-cache-dir "contrastive-lm==0.1.0"

ENTRYPOINT ["clm-serve"]
