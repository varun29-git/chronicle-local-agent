import mlx_lm
import json
import sqlite3
from mlx_lm import load, generate

# 1. Load the Sovereign Brain
model_path = "mlx-community/gemma-3-4b-it-4bit"
model, tokenizer = load(model_path)

def process_transaction(raw_text):
    # The "System Prompt" forces the model to stop the story and start the logic
    prompt = f"""
    You are the core logic engine for Accrual. 
    Task: Extract financial data from the text into JSON.
    Format: {{"name": str, "total": int, "paid": int, "udhaar": int}}
    
    Text: {raw_text}
    JSON:"""

    # Generate the JSON response
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
    
    try:
        # Parse the JSON from the text
        data = json.loads(response.strip())
        save_to_ledger(data)
        return data
    except:
        print("❌ Model failed to generate valid JSON. Retrying...")
        return None

def save_to_ledger(data):
    conn = sqlite3.connect('accrual_ledger.db')
    cursor = conn.cursor()
    
    # 1. Ensure customer exists or create them
    cursor.execute("INSERT OR IGNORE INTO customers (name) VALUES (?)", (data['name'],))
    
    # 2. Update their total Udhaar
    cursor.execute("UPDATE customers SET total_udhaar = total_udhaar + ? WHERE name = ?", 
                   (data['udhaar'], data['name']))
    
    # 3. Log the specific transaction
    cursor.execute("SELECT id FROM customers WHERE name = ?", (data['name'],))
    c_id = cursor.fetchone()[0]
    cursor.execute("INSERT INTO transactions (customer_id, amount, note) VALUES (?, ?, ?)",
                   (c_id, data['udhaar'], f"Purchased items. Paid {data['paid']}"))
    
    conn.commit()
    conn.close()
    print(f"✅ Accrual complete. {data['name']}'s ledger updated by ₹{data['udhaar']}.")

if __name__ == "__main__":
    # Test the loop
    user_input = "Suresh bought 10kg sugar for 400 but only paid 100."
    process_transaction(user_input)