from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 👉 已自动替换成你的本地模型路径
model_path = r"E:\big model use\model\Qwen2.5\qwen\Qwen2__5-0__5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.5,
    top_p=0.5,
    repetition_penalty=1.05,
    max_tokens=512
)

prompt = "讲个周幽王烽火戏诸侯的故事"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
llm = LLM(model=model_path, trust_remote_code=True)
outputs = llm.generate([text], sampling_params)

for output in outputs:
    print(output.outputs[0].text)