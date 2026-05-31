@echo off

cd /d "D:\Feature Detection in Satellite Imagery"

set PIP_CACHE_DIR=D:\Feature Detection in Satellite Imagery\cache\pip

set HF_HOME=D:\Feature Detection in Satellite Imagery\cache\huggingface
set TRANSFORMERS_CACHE=D:\Feature Detection in Satellite Imagery\cache\transformers
set TORCH_HOME=D:\Feature Detection in Satellite Imagery\cache\torch

set TMP=D:\Feature Detection in Satellite Imagery\cache\tmp
set TEMP=D:\Feature Detection in Satellite Imagery\cache\tmp

call .venv\Scripts\activate

code .