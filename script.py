from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device="cuda:0")

pipe("我叫沃尔夫冈，我住在柏林")


import torch
print(torch.cuda.is_available())
print(torch.__version__)


from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
    "Helsinki-NLP/opus-mt-zh-en"
)

quantizer = ORTQuantizer.from_pretrained(ort_model)


from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained("Helsinki-NLP/opus-mt-zh-en",from_transformers=True)

# Translation model - AutoModelForSeq2SeqLM
from optimum.onnxruntime import ORTQuantizer, ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

onnx_model = ORTModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", from_transformers=True) #export=True)

quantizer = ORTQuantizer.from_pretrained(onnx_model)

dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

model_quantized_path = quantizer.quantize(
    save_dir="path/to/output/model",
    quantization_config=dqconfig,
)


# How to quantize an ONNX model

