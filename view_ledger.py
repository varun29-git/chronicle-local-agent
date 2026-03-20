import sqlite3
from accrual_schema import initialize_database

def display_ledger():
    initialize_database()
    conn = sqlite3.connect('accrual_ledger.db')
    cursor = conn.cursor()
    
    print("\n--- ACCRUAL SOVEREIGN LEDGER ---")
    print(f"{'ID':<4} | {'Customer':<15} | {'Total Udhaar (₹)':<15}")
    print("-" * 40)
    
    cursor.execute("SELECT id, name, total_udhaar FROM customers")
    for row in cursor.fetchall():
        print(f"{row[0]:<4} | {row[1]:<15} | {row[2]:<15}")
    
    conn.close()

if __name__ == "__main__":
    display_ledger()
