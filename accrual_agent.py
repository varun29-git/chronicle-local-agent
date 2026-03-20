import mlx_lm
import json
import sqlite3
import re
from mlx_lm import load, generate
from accrual_schema import initialize_database

print("🧠 Loading Accrual Brain into RAM (One-time setup)...")
model_path = "mlx-community/gemma-3-4b-it-4bit"
model, tokenizer = load(model_path)
initialize_database()
print("✅ Brain is warm. Ready for transactions.")

def process_transaction(raw_text):
    # Forced format to avoid model chatter
    prompt = f"""<start_of_turn>user
Extract to JSON: "{raw_text}"
Rules: ONLY JSON. No stories.
Format: {{"name": "str", "total": int, "paid": int, "udhaar": int}}<end_of_turn>
<start_of_turn>model
{{"name":"""

    # We pre-fill '"name":' to force the model into the data structure immediately
    response = generate(model, tokenizer, prompt=prompt, max_tokens=60)
    full_str = '{"name":' + response.strip()
    
    # 1. Scrub hidden control characters
    clean_str = "".join(char for char in full_str if ord(char) >= 32)
    
    # 2. Regex to grab exactly what is between brackets
    match = re.search(r'(\{.*\})', clean_str)
    
    if match:
        try:
            data = json.loads(match.group(1))
            data = normalize_transaction(data)
            save_to_ledger(data)
        except Exception as e:
            print(f"❌ Parse Error: {e}. Model output was: {full_str}")
    else:
        print(f"❌ No valid structure found in: {full_str}")

def normalize_transaction(data):
    total = int(data.get("total", 0) or 0)
    paid = int(data.get("paid", 0) or 0)
    udhaar = data.get("udhaar")

    if udhaar is None:
        udhaar = total - paid

    udhaar = int(udhaar)

    # Keep ledger math consistent even when the model emits a bad udhaar value.
    expected_udhaar = total - paid
    if udhaar != expected_udhaar:
        udhaar = expected_udhaar

    normalized_name = str(data.get("name", "")).strip()
    if not normalized_name:
        raise ValueError("Missing customer name")

    if total < 0 or paid < 0 or udhaar < 0:
        raise ValueError("Negative amounts are not supported")

    if paid > total:
        raise ValueError("Paid amount cannot exceed total amount")

    return {
        "name": normalized_name,
        "total": total,
        "paid": paid,
        "udhaar": udhaar,
    }

def save_to_ledger(data):
    initialize_database()
    conn = sqlite3.connect('accrual_ledger.db')
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO customers (name) VALUES (?)", (data['name'],))
    cursor.execute("UPDATE customers SET total_udhaar = total_udhaar + ? WHERE name = ?", 
                   (data['udhaar'], data['name']))
    cursor.execute("INSERT INTO transactions (customer_id, amount, note) VALUES ((SELECT id FROM customers WHERE name=?), ?, ?)",
                   (data['name'], data['udhaar'], f"Paid {data['paid']} of {data['total']}"))
    conn.commit()
    conn.close()
    print(f"✅ Accrual verified: {data['name']} owes ₹{data['udhaar']} more.")

if __name__ == "__main__":
    print("\n--- ACCRUAL PERSISTENT AGENT ---")
    print("Type your transaction and hit Enter. Type 'exit' to stop.")
    while True:
        user_input = input("\n📝 Transaction: ")
        if user_input.lower() == 'exit':
            break
        process_transaction(user_input)
