---
language:
- en
- he
tags:
- translation
license: cc-by-4.0
datasets:
- quickmt/quickmt-train.he-en
model-index:
- name: quickmt-en-he
  results:
  - task:
      name: Translation eng-heb
      type: translation
      args: eng-heb
    dataset:
      name: flores101-devtest
      type: flores_101
      args: eng_Latn heb_Hebr devtest
    metrics:
    - name: BLEU
      type: bleu
      value: 34.32
    - name: CHRF
      type: chrf
      value: 62.37
    - name: COMET
      type: comet
      value: 87.91
---

<a href="https://huggingface.co/spaces/quickmt/quickmt-gui"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Open in Spaces"></a>

# `quickmt-en-he` Neural Machine Translation Model 

`quickmt-en-he` is a reasonably fast and reasonably accurate neural machine translation model for translation from `en` into `he`.


## Try it on our Huggingface Space

Give it a try before downloading here: https://huggingface.co/spaces/quickmt/QuickMT-gui


## Model Information

* Trained using [`eole`](https://github.com/eole-nlp/eole) 
* 200M parameter transformer 'big' with 8 encoder layers and 2 decoder layers
* 32k separate Sentencepiece vocabs
* Expested for fast inference to [CTranslate2](https://github.com/OpenNMT/CTranslate2) format
* Training data: https://huggingface.co/datasets/quickmt/quickmt-train.he-en/tree/main

See the `eole` model configuration in this repository for further details and the `eole-model` for the raw `eole` (pytorch) model.


## Usage with `quickmt`

You must install the Nvidia cuda toolkit first, if you want to do GPU inference.

Next, install the `quickmt` [python library](github.com/quickmt/quickmt). 

```bash
git clone https://github.com/quickmt/quickmt.git
pip install ./quickmt/
```

Finally, use the model in python:

```python
from quickmt import Translator
from huggingface_hub import snapshot_download

# Download Model (if not downloaded already) and return path to local model
# Device is either 'auto', 'cpu' or 'cuda'
t = Translator(
    snapshot_download("quickmt/quickmt-en-he", ignore_patterns="eole-model/*"),
    device="cpu"
)

# Translate - set beam size to 1 for faster speed (but lower quality)
sample_text = 'Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.'

t(sample_text, beam_size=5)
```

> 'ד"ר אהוד אור, פרופסור לרפואה באוניברסיטת דלהוזי בהליפקס, נובה סקוטיה ויו"ר המחלקה הקלינית והמדעית של האגודה הקנדית לסוכרת, הזהיר כי המחקר נמצא עדיין בימיו הראשונים.'

```python
# Get alternative translations by sampling
# You can pass any cTranslate2 `translate_batch` arguments
t([sample_text], sampling_temperature=1.2, beam_size=1, sampling_topk=50, sampling_topp=0.9)
```

> 'פרופסור לרפואה באוניברסיטת דלהוסרה בהליפקס בנובה סקוטיה ויו"ר החטיבה הקלינית והמחקרית של אגודת הסוכרת הקנדית הזהיר כי המחקר נמצא כיום בתחילתו.'

The model is in `ctranslate2` format, and the tokenizers are `sentencepiece`, so you can use `ctranslate2` directly instead of through `quickmt`. It is also possible  to get this model to work with e.g. [LibreTranslate](https://libretranslate.com/) which also uses `ctranslate2` and `sentencepiece`. A model in safetensors format to be used with `eole` is also provided.


## Metrics

`bleu` and `chrf2` are calculated with [sacrebleu](https://github.com/mjpost/sacrebleu) on the [Flores200 `devtest` test set](https://huggingface.co/datasets/facebook/flores) ("eng_Latn"->"heb_Hebr"). `comet22` with the [`comet`](https://github.com/Unbabel/COMET) library and the [default model](https://huggingface.co/Unbabel/wmt22-comet-da). "Time (s)" is the time in seconds to translate the flores-devtest dataset (1012 sentences) on an Nvidia RTX 4070s GPU with batch size 32.


|                                  |   bleu |   chrf2 |   comet22 |   Time (s) |
|:---------------------------------|-------:|--------:|----------:|-----------:|
| quickmt/quickmt-en-he            |  34.32 |   62.37 |     87.91 |       1.15 |
| facebook/nllb-200-distilled-600M |  23.83 |   53.98 |     84.12 |      25.78 |
| facebook/nllb-200-distilled-1.3B |  29    |   58.64 |     87.23 |      44.79 |
| facebook/m2m100_418M             |  20.53 |   50.74 |     81.38 |      21.7  |
| facebook/m2m100_1.2B             |  23.78 |   53.73 |     83.81 |      41.71 |
