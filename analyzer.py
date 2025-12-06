import os
from huggingface_hub import InferenceClient
from prompts import SYSTEM_PROMPT

class Analyzer:
    def __init__(self, api_key=None, base_url=None, model="Qwen/Qwen2.5-Coder-32B-Instruct"):
        token = api_key or os.environ.get("HF_TOKEN")
        self.client = InferenceClient(api_key=token)
        self.model = model

    def analyze_code(self, code_content):
        """
        Sends the code content to the LLM for analysis against the production framework.
        """
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is the code to analyze:\n\n{code_content}"}
            ]
            
            response = self.client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during analysis: {str(e)}"
