import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(
base_url="https://api.groq.com/openai/v1", #<-Check ZOOM chat for change
api_key=os.getenv("GITHUB_TOKEN"),
)
r = client.chat.completions.create(
model="llama-3.3-70b-versatile", #<-Check ZOOM chat for change
messages=[{"role": "user", "content": "What is my name?"}],
)
# print(r) # uncomment to see the whole messy response
print(r.choices[0].message.content)