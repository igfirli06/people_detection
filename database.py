import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",   # isi password MySQL-mu
        database="people_detection"
    )
