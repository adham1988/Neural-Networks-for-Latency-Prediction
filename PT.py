import socket
import random
import base64
import time
import numpy as np
import csv
import binascii
import subprocess

def tcp_client(video_name,encoding,IP,port,Timeout):
    # Create TCP client socket and connect to server
    protocol = "TCP"
    starttime = time.perf_counter()
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((IP, port))
    time1 = time.perf_counter() - starttime
    client_socket.settimeout(5)


    # Send the video name to the server
    client_socket.send(video_name.encode('utf-8'))
   
    BufferSize = 900
    
    timeout_count = Timeout
    
    with open("temp.txt", "wb") as f:
        while True:
            try:
                text = client_socket.recv(BufferSize)
            #print(text)
                if not text:
                    break
                f.write(text)
            except socket.timeout:
                print("Socket timed out while receiving data.")
                timeout_count = 1
                break
        
    time2 = time.perf_counter() - starttime - time1
    
    with open("temp.txt", "rb") as f:
        temp = f.read()
        if encoding == "Base64":
            #print("len of base",len(temp))
            temp = base64.b64decode(temp)
        else:
            #print("not base")
            #print("len of binary",len(temp))
            temp = bytes(int(temp[i:i+8], 2) for i in range(0, len(temp), 8))


        #temp = bytes(int(temp[i:i+8], 2) for i in range(0, len(temp), 8))
        #temp = base64.b64decode(temp)
        
        f.close()
    
    time3 = time.perf_counter() - starttime - time2 - time1

    with open("TCP.mp4", "wb") as fh:
        fh.write(temp)
        fh.close()
    time4 = time.perf_counter() - starttime - time3 - time2 - time1
    time5 = time.perf_counter() - starttime 





    # Decode the video data and save it to file
    #video_data = base64.b64decode(encoded_data)

    print(f'Received video data for "{video_name}" from TCP server')
    return [protocol, time1, time2, time3, time4, time5,timeout_count]
               
def udp_client(video_name,encoding,IP,port,Timeout):
    protocol = "UDP"
    time1 = 0
    starttime = time.perf_counter()
    # Create UDP client socket and send the video name to the server

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.sendto(video_name.encode('utf-8'), (IP, port))
    client_socket.settimeout(5)
    #print("send")
    encoded_data = b""

    """
    while True:
        data, server_address = client_socket.recvfrom(1300)
        if not data:
            break
        encoded_data += data
        if data == b"END":
            break"""
    timeout_count = Timeout
    while True:
        try:
            data, server_address = client_socket.recvfrom(1300)
        except socket.timeout:
            timeout_count = 1
            print("Timeout occurred")
            break

        if not data:
            break

        encoded_data += data
        if data == b"END":
            break
    
    if timeout_count == 0:
        encoded_data = encoded_data[:-3]
        # Add padding to the encoded data if necessary
        padding_length = 4 - (len(encoded_data) % 4)
        if padding_length != 4:
            encoded_data += b"=" * padding_length
            #print("received")
    
    
# Decode the encoded data and save it to a file
    #Sdecoded_data = base64.b64decode(encoded_data)
        with open("temp.txt", "wb") as temp_file:
            temp_file.write(encoded_data)
        time2 = time.perf_counter() - starttime 
    # Read the temp file.
    
    

        with open("temp.txt", "rb") as f:
            temp = f.read()
            if encoding == "Base64":
                temp = base64.b64decode(temp)
            elif encoding == "binary":
                temp = bytes(int(temp[i:i+8], 2) for i in range(0, len(temp), 8))
            f.close()

        """
        if encoding == "Base64":
            print("base")
            temp = base64.b64decode(temp)
        else:
            print("not base")
            temp = bytes(int(temp[i:i+8], 2) for i in range(0, len(temp), 8))"""
    
        time3 = time.perf_counter() - starttime - time2
        with open("UDP.mp4", "wb") as fh:
            fh.write(temp)
            fh.close()
        time4 = time.perf_counter() - starttime - time3 - time2
        time5 = time.perf_counter() - starttime

        print(f'Received video data for "{video_name}" from UDP server')
    elif timeout_count != 0:
        return [protocol, 0, 0, 0, 0, 0, 1]

   
    return [protocol, time1, time2, time3, time4, time5,Timeout]

   

def main(video_name,encoding,IP,port1,port2,Timeout):
    
    
    # Coin toss to randomly choose TCP or UDP
    #times = udp_client(video_name,encoding,IP,UDP_port,Timeout)
    #times = tcp_client(video_name,encoding,IP,TCP_port,Timeout)
    """
    if random.randint(0, 1):
        #tcp_client(video_name)
        times = tcp_client(video_name,encoding,IP,TCP_port,Timeout)
    else:
        #udp_client(video_name)
        times = udp_client(video_name,encoding,IP,UDP_port,Timeout)
    #udp_client('20.mp4')
    #times = udp_client(video_name)
    #times = tcp_client(video_name)
    """
    return times

def get_wifi_info():
    # Run the netsh wlan show interface command and get the output
    output = subprocess.check_output(['netsh', 'wlan', 'show', 'interface'])

    # Decode the output from bytes to string
    output_str = output.decode('utf-8')

    # Find the line containing the receive rate
    rx_rate_line = next((line for line in output_str.split('\n') if 'Receive rate (Mbps)' in line), None)
    # Find the line containing the transmit rate
    tx_rate_line = next((line for line in output_str.split('\n') if 'Transmit rate (Mbps)' in line), None)
    # Find the line containing the RSSI
    rssi_line = next((line for line in output_str.split('\n') if 'Signal' in line), None)

    # Extract the rate value from the matched lines
    rx_rate = float(rx_rate_line.split(':')[1].strip()) if rx_rate_line else None
    tx_rate = float(tx_rate_line.split(':')[1].strip()) if tx_rate_line else None
    rssi = float(rssi_line.split(':')[1].strip()[:-1]) if rssi_line else None
    if rssi:
        #rssi = round((rssi / 2) - 100, 2)
        rssi = rssi*0.01

    # Return the receive and transmit rates and RSSI as a tuple
    return rx_rate, tx_rate, rssi



if __name__ == '__main__':
    IP = socket.gethostbyname(socket.gethostname())
    
    TCP_port = 33100
    UDP_port = 33111
    #this is olny for UDP

    chunk_size = 1300
    iters = 1000
    
    n_bins = 10

    min_DL = 6
    max_DL = 401

    min_Tr = 2
    max_Tr = 821
    
    bin_edges_DL = np.linspace(min_DL, max_DL, n_bins + 1)
    bin_edges_Tr = np.linspace(min_Tr, max_Tr, n_bins + 1)
    
    
    T1_std = 15.8529
    T1_mean = 10.9362
    T1_min = 0
    T1_max = 176.9971999892732

    T2_std = 45.9239
    T2_mean = 60.5327
    T2_min = 2.511
    T2_max = 227.142

    T3_std = 4.9899
    T3_mean = 7.57097
    T3_min = 0.25529
    T3_max =  70.2520

    T4_std = 1.36451
    T4_mean = 1.42012
    T4_min = 0.4011
    T4_max = 47.9598





    rowss = []
    
    RSS_mean = 0.77924
    RSS_std = 0.152123

    DL_mean = 131.71
    DL_std = 113.553
   

    Transmission_mean = 84.3749
    Transmission_std = 150.01
    
    DL_binned_mean = 3.24256
    DL_binned_std = 2.45961
    DL_min = 1 
    DL_max = 10

    Transm_binned_mean = 1.67537
    Transmission_binned_std = 1.68696
    Tr_min = 1
    Tr_max = 10
    
    MTU_mean = 844.036
    MTU_std = 416.64
    MTU_min = 340
    MTU_max = 1340

    Fsize_mean = 12.38
    Fsize_std = 11.7626
    Fsize_min = 1
    Fsize_max = 30


    Indexs = np.array([]).astype(int)

    #arrays for database
    Location_Fjer = np.array([]).astype(int)
    Location_Frankfurt = np.array([]).astype(int)
    Location_Paris = np.array([]).astype(int)
    Technology_WiFi = np.array([]).astype(int)
    RSSI = np.array([],dtype=np.float64)
    Protocol_UDP = np.array([]).astype(int)
    DownLink_stand = np.array([],dtype=np.float64)
    Filesize_stand = np.array([],dtype=np.float64)
    Transmission_stand = np.array([],dtype=np.float64)
    MTU_stand = np.array([],dtype=np.float64)
    Encoding_binary = np.array([]).astype(int)
    T1 = np.array([],dtype=np.float64)


    #arrays for timestamps
    location = np.array([]).astype(str)
    tech = np.array([]).astype(str)
    RSSI = np.array([])
    DownLink = np.array([]).astype(int)
    UpLink = np.array([])
    MTU = np.array([]).astype(int)
    protocol = np.array([]).astype(str)
    file_size = np.array([]).astype(int)
    encoding = np.array([]).astype(str)
    Transmission = np.array([]).astype(int)
    after_conn = np.array([])
    after_recv = np.array([])
    after_stor = np.array([])
    after_deco = np.array([])
    total = np.array([])
    Timeout = np.array([]).astype(int)
    
    my_table = {
    (1, 340, 'Base64'): 5,
    (1, 340, 'binary'): 28,
    (1, 740, 'Base64'): 2,
    (1, 740, 'binary'): 12,
    (1, 1340, 'Base64'): 2,
    (1, 1340, 'binary'): 7,
    (10, 340, 'Base64'): 46,
    (10, 340, 'binary'): 274,
    (10, 740, 'Base64'): 20,
    (10, 740, 'binary'): 118,
    (10, 1340, 'Base64'): 11,
    (10, 1340, 'binary'): 64,
    (30, 340, 'Base64'): 137,
    (30, 340, 'binary'): 820,
    (30, 740, 'Base64'): 59,
    (30, 740, 'binary'): 352,
    (30, 1340, 'Base64'): 32,
    (30, 1340, 'binary'): 190
}

    for x in range(iters):
        print(x)

        name = int(np.random.randint(3, size=1)) + 1
        filesize_array = np.array([1,10,30])
        #name = 3
        if name == 1:
            name2 = 1
            file_size = np.append(file_size,filesize_array[0])
        elif name == 2:
            name2 = 10
            file_size = np.append(file_size,filesize_array[1])
        elif name == 3:
            name2 = 30
            file_size = np.append(file_size,filesize_array[2])
        #name2 = 30
        enc = int(np.random.randint(2, size=1)) 
        enco = np.array(["Base64","binary"])
        #filename_req = str(name2)+".mp4"+","+ str(chunk_size) +","+ enco[enc]# Requested file name
        #filename_req = str(name2)+".mp4" # Requested file name
        filename_req = str(name2)+".mp4"+","+ str(chunk_size)+","+str(enco[enc])
        #video_name = '10.mp4'
        encoding = np.append(encoding, str(enco[enc]))  #input any distance value
        #print(enco[1])
        Timeoutval = 0
        rx_rate, tx_rate, rssi = get_wifi_info()
        DownLink = np.append(DownLink, rx_rate)
        UpLink = np.append(UpLink, tx_rate)
        RSSI = np.append(RSSI, rssi)
        
        loc = np.array(["AAU","Fjer","Frankfurt","Paris"])
        loc_value = loc[0]
        location = np.append(location, loc_value)  #input any distance value

        technology = np.array(["WiFi","4G", "5G"])
        tech_value = technology[0]
        tech = np.append(tech, tech_value)

        mss_size = np.array([300, 700, 1300])
        
        mss_value = mss_size[2] + 40
        MTU = np.append(MTU, mss_value)
        
        num_transmissions = my_table[(name2, mss_value, enco[enc])]
        #print("Number of transmission ",num_transmissions) # prints 5

        Transmission = np.append(Transmission, num_transmissions)

        if loc_value == "Fjer":
            Location_Fjer= np.append(Location_Fjer, 1)   
        else:
            Location_Fjer= np.append(Location_Fjer, 0)
        
        if loc_value == "Frankfurt":
            Location_Frankfurt = np.append(Location_Frankfurt,1)
        else:
            Location_Frankfurt = np.append(Location_Frankfurt,0)
        
        if loc_value == "Paris":
            Location_Paris = np.append(Location_Paris,1)
        else:
            Location_Paris = np.append(Location_Paris,0)


        
        if tech_value == "WiFi":
            Technology_WiFi = np.append(Technology_WiFi, 1)
        else:
            Technology_WiFi = np.append(Technology_WiFi, 0)
        
        RSSI = np.append(RSSI, (rssi - RSS_mean)/RSS_std)


        #this for normalizing 
        if name2 == 1:
            Filesize_stand = np.append(Filesize_stand, -0.964075)
        elif name2 == 10:
            Filesize_stand = np.append(Filesize_stand, -0.198354)
        elif name2 == 30:
            Filesize_stand = np.append(Filesize_stand, 1.50325)
        
        #T1 = np.append(T1, 0.901307)

        
        bin_index_Tr = np.digitize(num_transmissions, bin_edges_Tr)

        #Transmission_norm = np.append(Transmission_norm, (num_transmissions - Transmission_mean)/Transmission_std)
        Transmission_stand = np.append(Transmission_stand, (bin_index_Tr - Transm_binned_mean)/(Transmission_binned_std))
        
       
               
        """
        if mss_value == 340:
            MTU_norm = np.append(MTU_norm, -1.21167)
        elif mss_value == 740:
            MTU_norm = np.append(MTU_norm, -0.252497)
        elif mss_value == 1340:
            MTU_norm = np.append(MTU_norm, 1.18626)
        """
        
        if mss_value == 340:
            MTU_stand = np.append(MTU_stand, -1.21167)
        elif mss_value == 740:
            MTU_stand = np.append(MTU_stand, -0.252497)
        elif mss_value == 1340:
            MTU_stand = np.append(MTU_stand, 1.18626)



        if enco[enc] == "binary":
            Encoding_binary = np.append(Encoding_binary, 1)
            #print("binary")
        else:
            Encoding_binary = np.append(Encoding_binary, 0)
            #print("base")
        
        
        bin_index_DL = np.digitize(rx_rate, bin_edges_DL)
        
        #DownLink_norm = np.append(DownLink_norm, (rx_rate - DL_mean)/DL_std )
        DownLink_stand = np.append(DownLink_stand, (bin_index_DL - DL_binned_mean)/(DL_binned_std))
        #print("DL:",bin_index_DL)
        




        rows1 = [] # initialize empty list

        
        if random.randint(0, 1):
            Protocol_UDP = np.append(Protocol_UDP, 0)
            rows1.append([Location_Fjer[-1], Location_Frankfurt[-1], Location_Paris[-1], Technology_WiFi[-1],
            Protocol_UDP[-1], Encoding_binary[-1], DownLink_stand[-1], Filesize_stand[-1], MTU_stand[-1], Transmission_stand[-1]]) 
            with open("database.txt", 'a') as f_object:
                writer_object = csv.writer(f_object)
                writer_object.writerows(rows1)
                f_object.close()
            time.sleep(0.5)
            times = tcp_client(filename_req,enco[0],IP,TCP_port,Timeoutval)    

        else:
            Protocol_UDP = np.append(Protocol_UDP, 1)
            rows1.append([Location_Fjer[-1], Location_Frankfurt[-1], Location_Paris[-1], Technology_WiFi[-1],
            Protocol_UDP[-1], Encoding_binary[-1], DownLink_stand[-1], Filesize_stand[-1], MTU_stand[-1], Transmission_stand[-1]])    
            with open("database.txt", 'a') as f_object:
                writer_object = csv.writer(f_object)
                writer_object.writerows(rows1)
                f_object.close()
            time.sleep(0.5)
            times = udp_client(filename_req,enco[0],IP,UDP_port,Timeoutval)
        
        

        

       
        protocol = np.append(protocol, times[0])
        after_conn = np.append(after_conn, times[1]*1000) 
        after_recv = np.append(after_recv, times[2]*1000)
        after_stor = np.append(after_stor, times[3]*1000)
        after_deco = np.append(after_deco, times[4]*1000)
        total = np.append(total, times[5]*1000)
        Timeout = np.append(Timeout, times[6])

        
        
        last_value = after_recv[-1]
        
        new_value = after_recv[-1]
        
        
        with open('C.csv', mode='a', newline='') as f:
            # create a csv writer object
            writer = csv.writer(f)
            # write the data row
            writer.writerow([x+1, after_conn[-1], after_recv[-1], after_stor[-1], after_deco[-1], total[-1]])
        
        rows3 = [] # initialize empty list
        rows3.append([(after_conn[-1]-T1_mean)/T1_std, (after_recv[-1]-T2_mean)/T2_std, (after_stor[-1]-T3_mean)/T3_std, (after_deco[-1]-T4_mean)/T4_std])
        

        with open("TrueDelays.txt", 'a') as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerows(rows3)
            f_object.close()

        time.sleep(2)
        Indexs = np.append(Indexs, x+1)
    
    rowss.append(
        ["Index","T1", "T2", "T3", "T4", "Total"])
    
    for x in range(len(after_conn)):
        rowss.append([Indexs[x],after_conn[x],  after_recv[x], after_stor[x], after_deco[x], total[x]])

  
    print("the number of timeout is : ",np.sum(Timeout),"out of",iters)


