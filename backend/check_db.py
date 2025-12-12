import sqlite3

conn = sqlite3.connect("restaurant.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in database:")
for table in tables:
    print(f"  - {table[0]}")

# Check if employees table has the right columns
if any("employees" in t for t in tables):
    cursor.execute("PRAGMA table_info(employees)")
    cols = cursor.fetchall()
    print("\nEmployees table columns:")
    for col in cols:
        print(f"  - {col[1]}: {col[2]}")
else:
    print("\n⚠️  employees table does NOT exist!")

conn.close()
