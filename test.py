import socket

# UDPソケットを設定
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 送信先のIPアドレスとポート番号
mac_address = ('133.68.108.26', 8000)

# メッセージを送信
message = b's'  # 送信するメッセージ
udp_sock.sendto(message, mac_address)

print("Message sent.")
