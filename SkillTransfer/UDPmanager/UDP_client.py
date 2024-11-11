import socket
import time

class UDP_Client():
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data_list, client_ip, client_port):
        message =  ','.join(map(str, data_list))
        self.sock.sendto(message.encode("utf-8"), (client_ip, client_port))

    def close(self):
        self.sock.close()

if __name__ == "__main__":
    client_ip = "133.68.108.36"  # 送信先のIPアドレス
    client_port = 31110  # 送信先のポート

    sender = UDP_Client()

    while True:
        for i in range(100):
            data_list = [0.5+i/100, 0.5+i/100, 0.6+i/100, 0.7+i/100, 1.5+i/100, 1.6+i/100, 1.7+i/100, 0.5+i/100, 0.6+i/100, 0.7+i/100, 1.5+i/100, 0.5+i/100, 0.6+i/100, 0.7+i/100, 1.5+i/100]
            sender.send(data_list,client_ip,client_port)
            print(data_list)
            print(time.time())
            time.sleep(0.01)
        