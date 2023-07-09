# [Private 1st] DACON Judgement of Court
* official link (korean): https://dacon.io/competitions/official/236112/overview/description

# 1. Goal
- This challenge aims to develop an AI that predicts legal case outcomes. The significance is a crucial step in exploring how AI can be effectively utilized in the field of law.

# 2. Overview & Results
- TBD

# 3. Reproducibility
- Install libraries for text classification models.
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

- Preprocess all training and testing samples.
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
python3 generate_qa_list_for_llm.py
```

- Run text classification models (i.e., RemBeRT, ALBERT, DeBERTa, and BigBirdPegasus)
- Please download pretrained weights following this [link](https://drive.google.com/file/d/1B_litWreHZnkRN4VZrOCczbgePl4Szxb/view?usp=sharing).
```bash
CUDA_VISIBLE_DEVICES=0 python3 infer_classification_models.py \
--model_names rembert,albert-xxlarge-v2,deberta-v2-xxlarge,bigbird-pegasus-large-bigpatent
```

- Install libraries for large langage models (i.e., vicuna).
```bash
deactivate

cd llm

python3 -m venv venv
source ./venv/bin/activate

pip3 install --upgrade pip
pip3 install -e .

git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-v1.3
```

- Run vicuna-13b-v1.3 using the dataset for few-shot learning.
```
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path vicuna-13b-v1.3 --port 21002
python3 -m run_llm \
--controller-address "http://localhost:21001" --model-name vicuna-13b-v1.3 \
--temperature 0.001 --max-new-tokens 100
```

- Produce the final result by unifying two results from classification and language models.
```bash
python3 ensemble_all_results.py
```

# 4. Training
- Train four classification models.
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --model bigbird-pegasus-large-bigpatent
CUDA_VISIBLE_DEVICES=0 python3 train.py --model rembert
CUDA_VISIBLE_DEVICES=0 python3 train.py --model deberta-v2-xxlarge
CUDA_VISIBLE_DEVICES=0 python3 train.py --model albert-xxlarge-v2
```

# 5. Acknowledgement
- Thanks to the authors of [Vicuna-13B-v1.3](https://github.com/lm-sys/FastChat) used in this respository.

If you have any question or find any bug, please email [me](shjo.april@gmail.com).