import requests

# 1. SETUP
URL = "http://localhost:8000/predict"
# Ensure the path is correct
FILE_PATH = r"C:\Users\jrzem\Downloads\Screenshot 2025-12-30 101207.png"

print(f"🚀 Sending image to {URL}...")
print(f"📁 File: {FILE_PATH}")

try:
    # 2. OPEN AND SEND FILE
    # CHANGE: The key must be "image" to match the FastAPI endpoint
    with open(FILE_PATH, "rb") as f:
        response = requests.post(URL, files={"image": f})

    # 3. PRINT RESULTS
    print(f"\n✅ Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\n📊 PREDICTION RESULTS:")
        # Pretty print the JSON
        import json
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n❌ Error: {response.text}")

except FileNotFoundError:
    print(f"\n❌ Error: Could not find the file at {FILE_PATH}")
except Exception as e:
    print(f"\n❌ Connection Error: {e}")