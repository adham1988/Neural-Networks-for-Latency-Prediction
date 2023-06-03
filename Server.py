import socket
import threading
import base64
import binascii

def handle_tcp_client(client_socket):
    # Receive the video name from the client
    a = client_socket.recv(1024).decode('utf-8')
    x = a.split(",")
    video_name = x[0]
    encoding = x[2]
    print("video requested : ",video_name, "encoding",encoding)

    # Open the video file and encode its data
    with open(video_name, 'rb') as video_file:
        #open the video file and read its contents into a byte string variable
        video_data = video_file.read()
        if encoding == "Base64":
            #print("encoding base")
            encoded_data = base64.b64encode(video_data)
        elif encoding == "binary":
            #print("encoding binary")
            encoded_data = ''.join(format(byte, '08b') for byte in video_data)
        

        #binary encoding
        #binary_data = ''.join(format(byte, '08b') for byte in video_data)

        #encoded_data = base64.b64encode(video_data)
        # Convert the video data to hexadecimal and send it to the client
        #hex_string = binascii.hexlify(video_data).decode('utf-8')
        
        

   
    # Send the encoded data to the client using hexadecimal
    #client_socket.sendall(hex_string.encode('utf-8'))
    if encoding == "Base64":
        #print("sending base")
        client_socket.sendall(encoded_data)
    elif encoding == "binary":
        #print("sending binary")
        client_socket.sendall(encoded_data.encode())
    # Send the encoded data to the client using base64
    #client_socket.sendall(encoded_data)
    #client_socket.sendall(binary_data.encode())

    
    # Close the client socket
    client_socket.close()

def handle_udp_client(data, client_address, server_socket):
    # Receive the video name from the client
    a = data.decode('utf-8')
    x = a.split(",")
    video_name = x[0]
    
    chunk_size = int(x[1])
    encoding = x[2]
    print("video requested : ",video_name, "encoding",encoding)
    #print("video requested : ",video_name + "and UDP chunk : ",chunk_size + "encoding: ",encoding)
    
    with open(video_name, 'rb') as video_file:
        video_data = video_file.read()
        if encoding == "Base64":
            #print("encode base")
            encoded_data = base64.b64encode(video_data)
        else:
            #print("encode binary")
            encoded_data = ''.join(format(byte, '08b') for byte in video_data)
    
# Send the encoded data to the client in chunks of 1300 bytes
   
    num_chunks = len(encoded_data) // chunk_size + 1
    #print(num_chunks)
    if encoding == "Base64":
        for i in range(num_chunks):
            #print("send chunks")
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(encoded_data))
            #print("sending base")
            server_socket.sendto(encoded_data[start:end], client_address)
            #print("shunk sent")
    else:
        for i in range(num_chunks):
            #print("send chunks")
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(encoded_data))
            #print("sending binary")
            server_socket.sendto(encoded_data[start:end].encode(), client_address)

    server_socket.sendto(b"END", client_address)
    
    


    

def tcp_server():
    # Create TCP server socket and listen for incoming connections
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    TCP_port = 33100
    server_socket.bind(('0.0.0.0', TCP_port))
    server_socket.listen(1)
    print('TCP server listening on port ',TCP_port)

    while True:
        # Accept incoming connection
        client_socket, client_address = server_socket.accept()
        print(f'Received TCP connection from {client_address}')

        # Spawn a new thread to handle the client request
        tcp_thread = threading.Thread(target=handle_tcp_client, args=(client_socket,))
        tcp_thread.start()

def udp_server():
    # Create UDP server socket and bind to a specific address
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    UDP_port = 33111
    server_socket.bind(('0.0.0.0', UDP_port))
    print('UDP server listening on port ',UDP_port)

    while True:
        # Receive incoming data and client address
        data, client_address = server_socket.recvfrom(1024)
        print(f'Received UDP connection from {client_address}')

        # Spawn a new thread to handle the client request
        udp_thread = threading.Thread(target=handle_udp_client, args=(data, client_address, server_socket))
        udp_thread.start()

def main():
    # Start the TCP and UDP server threads
    tcp_thread = threading.Thread(target=tcp_server)
    udp_thread = threading.Thread(target=udp_server)
    tcp_thread.start()
    udp_thread.start()

if __name__ == '__main__':
    main()