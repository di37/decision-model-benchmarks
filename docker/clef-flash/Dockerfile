FROM local/laya-server:0.3.21
RUN pip install --no-cache-dir "transformers==5.10.2" accelerate pillow
RUN pip install --no-cache-dir --no-deps "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
ENTRYPOINT ["python"]
