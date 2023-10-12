import threading

def send_frames():
    while True:
        print('haha')

def receive_results():
    while True:
        print("Toan")

send_thread = threading.Thread(target=send_frames)
receive_thread = threading.Thread(target=receive_results)
receive_thread.start()
send_thread.start()