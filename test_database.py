import time
import threading
import mysql.connector

def test_query():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="people_detection"
    )
    cur = conn.cursor()
    start = time.time()
    cur.execute("SELECT * FROM cctv;")
    cur.fetchall()
    end = time.time()
    print(f"Query time: {end - start:.4f} seconds")
    cur.close()
    conn.close()

threads = []
for _ in range(50):  # 50 concurrent queries
    t = threading.Thread(target=test_query)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
