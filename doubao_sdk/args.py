from volcenginesdkarkruntime import Ark
from openai import OpenAI
from tqdm import tqdm
import time

AK = "YOUR_AK"
SK = "YOUR_SK"
API_KEY= "YOUR API KEY"
MODEL_ID = {
    "doubao-lite-4k": "ep-20240614100427-s8nmn",
    "doubao-pro-4k": "ep-20241111192110-lc4wp",

}

MODEL_API = {
  "deepseek": "DEEPSEEK_API",
  "gpt": "GPT_API"
}

def get_api_key(model_name):
  return MODEL_API[model_name]

def get_client(model_name="deepseek"):
  client = None
  if model_name == "deepseek":
    client = Ark(api_key=get_api_key(model_name))
  elif model_name == "gpt":
    client = OpenAI(api_key=get_api_key(model_name), base_url="https://YOUR_BASE_URL")
  return client

def get_model(model_name):
  model_map = {
    "deepseek": "deepseek-v3-250324",
    "gpt": "gpt-4o-mini"
  }
  model = model_map[model_name]
  return model

def get_response_with_retries(client, prompt, model_name="deepseek", TEMPERATURE=0.8, retries=5):
  model = get_model(model_name)
  for attempt in range(retries):
    try:
      response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
      )
      print(response)
      return response.choices[0].message.content.strip()
    except Exception as e:
      wait = 2 ** attempt
      tqdm.write(f"[!] Error: {e}. Retrying in {wait}s...")
      time.sleep(wait)
  raise RuntimeError("Exceeded max retries.")