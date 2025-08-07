import pywhatkit
import time

def send_whatsapp_message(phone_number, message):
    try:
        print(f"[WA] Mengirim pesan ke {phone_number} ...")
        pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=15, tab_close=True)
        time.sleep(10)  # Tunggu agar pesan terkirim dengan stabil
        print("[WA] Pesan berhasil dikirim.")
    except Exception as e:
        print("[WA ERROR]", e)
