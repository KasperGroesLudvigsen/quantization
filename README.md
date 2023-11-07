# Quantization

https://huggingface.co/Helsinki-NLP/opus-mt-zh-en

Objective:
To make inference faster on CPU and GPU.

There are three types of quantization:
1. Post training dynamic quantization
2. Post training static quantization
2. Quantization aware training


Post training dynamic quantization computes the range for each activation on the fly at run time. This can therefore be slower than static quantization, but static quantization often leads to a drop in model accuracy compared to dynamic quantization. 

Since we're dealing with pre-trained models, we're opting for either #1 or #2. 

Performance (speed) is important, but the accuracy of the model output is even more
important. We're willing to choose a slower option to optimize for accuracy. 


Other speed up considerations:
- Are the models in ONNX? What inference engine are we using?


To do:
- Find test data
- write test script
- test quantized model accuracy compared to unquantized 
- are our models based on "marian" ? KeyError: 'marian is not supported yet with the onnx backend. Only [] are supported. If you want to support onnx please propose a PR or open up an issue.'


# Other to do:
- Convert Lehmann to ONNX and use an inference engine https://onnxruntime.ai/ https://medium.com/axinc-ai/overview-of-onnx-and-operators-9913540468ae 
