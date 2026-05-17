import json, urllib.request

req = {
    "message": ["Hello"],
    "system_prompt": "You are a highly intelligent Mixture of Experts (MoE) Large Language Model. You provide accurate, helpful, and concise answers.",
    "use_rag": False,
    "use_web": False,
    "stream": True,
    "temperature": 0.1,
    "max_tokens": 512,
    "top_k": 40
}
try:
    url = "http://127.0.0.1:8080/chat/stream"
    data = json.dumps(req).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req) as f:
        print("Status:", f.status)
        for line in f:
            print(line.decode('utf-8'))
except Exception as e:
    print(e)
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))
