import sqlite3

conn = sqlite3.connect(r"C:\Users\sinsa\Downloads\cube\test.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM reviews WHERE id = ?" , (9,))

conn.commit()
conn.close()

print("Row deleted successfully!")
