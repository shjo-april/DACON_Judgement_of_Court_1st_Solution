# [Private 1st] DACON Judgement of Court

# 1. Goal
- TBD

# 2. Overview & Results
- TBD

# 3. Reproducibility
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

```bash
python3 preprocess.py

CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_trainval.py --model google/bigbird-pegasus-large-bigpatent --tag bigbird-pegasus-large-bigpatent
CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_trainval.py --model google/rembert --tag rembert
CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_trainval.py --model microsoft/deberta-v2-xxlarge --tag deberta-v2-xxlarge
CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_trainval.py --model albert-xxlarge-v2 --tag albert-xxlarge-v2

CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_test.py --model google/bigbird-pegasus-large-bigpatent --tag bigbird-pegasus-large-bigpatent
CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_test.py --model google/rembert --tag rembert
CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_test.py --model microsoft/deberta-v2-xxlarge --tag deberta-v2-xxlarge
CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_test.py --model albert-xxlarge-v2 --tag albert-xxlarge-v2

CUDA_VISIBLE_DEVICES=0 python3 extract_embs_for_llm.py --file ./open/train.json
CUDA_VISIBLE_DEVICES=1 python3 extract_embs_for_llm.py --file ./open/test.json
python3 generate_qa_list_for_llm.py # for few-shot learning

CUDA_VISIBLE_DEVICES=0 python3 infer_classification_models.py \
--model_names rembert,albert-xxlarge-v2,deberta-v2-xxlarge,bigbird-pegasus-large-bigpatent
```

```bash
deactivate

cd llm

python3 -m venv venv
source ./venv/bin/activate

pip3 install --upgrade pip
pip3 install -e .

git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-v1.3

python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path vicuna-13b-v1.3 --port 21002

python3 -m run_llm \
--controller-address "http://localhost:21001" --model-name vicuna-13b-v1.3 \
--temperature 0.001 --max-new-tokens 100
```

```bash
python3 ensemble_all_results.py
```

# 4. Training

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --model bigbird-pegasus-large-bigpatent
CUDA_VISIBLE_DEVICES=0 python3 train.py --model rembert
CUDA_VISIBLE_DEVICES=0 python3 train.py --model deberta-v2-xxlarge
CUDA_VISIBLE_DEVICES=0 python3 train.py --model albert-xxlarge-v2
```