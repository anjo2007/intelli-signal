import socket

def run_udp_client():
    # 1. Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("",5005))

    
    
    target_ip = '192.168.1.2'  # Fixed: added quotes around IP address
    target_port = 5005

    message = input("Message: ")

    # The location is passed here as a tuple: (target_ip, target_port)
    sock.sendto(message.encode('utf-8'), (target_ip, target_port))
    
    print("Message sent! Message=",message)
    
    # Close the socket
    sock.close()
while True:
    run_udp_client()
