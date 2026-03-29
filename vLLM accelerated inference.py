import os
# 强制离线，彻底屏蔽HuggingFace校验
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 你的原始模型路径（完全不动）
MODEL_PATH = r"E:\big model use\model\Qwen2.5\qwen\Qwen2___5-0___5B-Instruct"

# 切换目录，绕开所有路径校验
os.chdir(MODEL_PATH)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载本地模型
tokenizer = AutoTokenizer.from_pretrained(".", local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(".", local_files_only=True, trust_remote_code=True, device_map="auto")

# 推理
prompt = "讲个周幽王烽火戏诸侯的故事"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 🔥 修复参数：max_tokens → max_new_tokens
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # 修复这里！
        temperature=0.5,
        top_p=0.5,
        repetition_penalty=1.05
    )

# 输出结果
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print(response)