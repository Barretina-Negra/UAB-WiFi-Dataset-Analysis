import requests

# La teva API key (obtinguda de PublicAI)
API_KEY = "zpka_4d26fbf3602644d1a719050b1f901e2f_0030d1b5"

# Configuració
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "UAB-THE-HACK/1.0"
}

# Fer una pregunta al model
payload = {
    "model": "BSC-LT/ALIA-40b-instruct_Q8_0",
    "messages": [
        {"role": "user", "content": "Hola! Com puc crear un chatbot?"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}

response = requests.post(
    "https://api.publicai.co/v1/chat/completions",
    headers=headers,
    json=payload
)
print("Status code:", response.status_code)
print("Response text:", response.text)

print(response.json()["choices"][0]["message"]["content"])
