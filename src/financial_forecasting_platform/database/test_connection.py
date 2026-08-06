from financial_forecasting_platform.database.connection import get_connection


connection = get_connection()

cursor = connection.cursor()

cursor.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema='public';
""")

tables = cursor.fetchall()

print(tables)

cursor.close()
connection.close()