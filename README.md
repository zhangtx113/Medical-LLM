# Get Start
```python
conda create -n medical-llm python=3.10 -y
conda activate medical-llm
```

Download Model
```python
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Models/Qwen3-0.6B
```

Deploy Model
```pyhton
pip install vllm
MODEL_PATH=./Models/Qwen3-0.6B
vllm serve $MODEL_PATH \
  --served-model-name Qwen3-0.6B \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 8
```

Download Dataset
```python
cd eval
python download_medxpertqa.py
```

evaluate
```python
python eval_medxpertqa.py \
  --data-path medxpertqa_text.jsonl \
  --model local \
  --medical-task "Diagnosis","Treatment","Basic Medicine" \
  --body-system "Cardiovascular" \
  --question-type Reasoning,Understanding \
  --max-samples 10

python eval_medxpertqa_local.py \
  --data-path medxpertqa_text.jsonl \
  --medical-task "Diagnosis","Treatment","Basic Medicine" \
  --body-system "Cardiovascular" \
  --question-type "Reasoning","Understanding" \
  --max-samples 10 \
  --output-path results/predictions_qwen3_8b_local.jsonl


```

构建知识点
提取问题
答案:COT,answer,type,key_word