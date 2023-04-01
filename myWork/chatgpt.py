from tqdm import tqdm
import jsonlines
import openai
import math
import json
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

data = {}
with jsonlines.open('data/dev.jsonl') as f:
    for line in f.iter():
        # data[line["index"]] = "You have to explain the following cause and effect in less than 20 words.\n\nCause: "+line["cause"]+"\nEffect: "+line["effect"]
        data[line["index"]] = line["cause"]+" "+line["effect"]+" Explain the reason for this in less than 20 words."

openai.api_key_path = "/raid/ai20btech11011/ecare-rag/openai2"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    # return openai.Completion.create(**kwargs)
    return openai.ChatCompletion.create(**kwargs)


response = {}
i = 0
for key, content in tqdm(data.items()):
    reply = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    )
    response[key] = reply['choices'][0]['message']['content']
    time.sleep(1.5)

with open("data/chatgpt_prediction.json", "w") as f:
    json.dump(response, f)
