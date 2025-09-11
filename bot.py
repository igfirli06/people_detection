import requests
import json

TELEGRAM_API_TOKEN = "8283970823:AAE1653ZRnqixbl3bbR-YLNdz2fMc57ZIXE"

url = f"https://api.telegram.org/bot8283970823:AAE1653ZRnqixbl3bbR-YLNdz2fMc57ZIXE/getUpdates"

try:
    response = requests.get(url)
    data = response.json()

    if data["ok"]:
        if data["result"]:
            chat_id = data["result"][-1]["message"]["chat"]["id"]
            print(f"Chat ID Anda adalah: {chat_id}")
        else:
            print("Tidak ada pesan baru. Pastikan Anda sudah memulai percakapan dengan bot.")
    else:
        print("Gagal mendapatkan update dari API Telegram.")

except Exception as e:
    print(f"Terjadi kesalahan: {e}")