"""Existing ip extraction operator."""
import os
import binascii
import random
from scapy.all import rdpcap, IP, TCP, UDP, IPv6, PacketList
import math
import ipaddress
dataset_dir = ''
min_plen = 128

def random_ipv4():
    IPV4_MAX = ipaddress.IPv4Address._ALL_ONES
    ip_int = random.randint(0, IPV4_MAX)
    ip_str = ipaddress.IPv4Address._string_from_ip_int(ip_int)
    return ip_str

def random_ipv6():
    IPV6_MAX = ipaddress.IPv6Address._ALL_ONES
    ip_int = random.randint(0, IPV6_MAX)
    ip_str = ipaddress.IPv6Address._string_from_ip_int(ip_int)
    return ip_str

def random_field(bits):
    field_max = 2 ** bits - 1
    field_int = random.randint(0, field_max)
    return field_int

def random_ip_port(packets: PacketList):
    if IP in packets[0]:
        client_ip = packets[0][IP].src
        server_ip = packets[0][IP].dst
        r_client_ip = random_ipv4()
        r_server_ip = random_ipv4()
    elif IPv6 in packets[0]:
        client_ip = packets[0][IPv6].src
        server_ip = packets[0][IPv6].dst
        r_client_ip = random_ipv6()
        r_server_ip = random_ipv6()
    else:
        print('other L3 protocol')
    r_client_port = random_field(16)
    r_server_port = random_field(16)
    for packet in packets:
        if IP in packet:
            is_client = packet[IP].src == client_ip
            packet[IP].src = r_client_ip if is_client else r_server_ip
            packet[IP].dst = r_server_ip if is_client else r_client_ip
            if TCP in packet:
                packet[TCP].sport = r_client_port if is_client else r_server_port
                packet[TCP].dport = r_server_port if is_client else r_client_port
            if UDP in packet:
                packet[UDP].sport = r_client_port if is_client else r_server_port
                packet[UDP].dport = r_server_port if is_client else r_client_port
        if IPv6 in packet:
            is_client = packet[IPv6].src == client_ip
            packet[IPv6].src = r_client_ip if is_client else r_server_ip
            packet[IPv6].dst = r_server_ip if is_client else r_client_ip
            if TCP in packet:
                packet[TCP].sport = r_client_port if is_client else r_server_port
                packet[TCP].dport = r_server_port if is_client else r_client_port
            if UDP in packet:
                packet[UDP].sport = r_client_port if is_client else r_server_port
                packet[UDP].dport = r_server_port if is_client else r_client_port
    return packets

def cut(obj, sec):
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except Exception as e:
        remanent_count = 0
        print(1)
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result

def bigram_generation(packet_datagram, packet_len, flag=True, num_interval=2):
    result = ''
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for sub_string_index in range(0, len(generated_datagram), num_interval):
        if sub_string_index != len(generated_datagram) - 1:
            token_count += num_interval
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    if flag == True:
        result = result.rstrip()
    return result

def get_feature_packet(label_pcap, payload_len):
    if os.path.getsize(label_pcap) == 0:
        print('Current File Size = 0 !')
        return -1
    feature_data = []
    pcap_reader = rdpcap(label_pcap)
    flow_size = len(pcap_reader)
    flow_data_string = ''
    if flow_size < 1:
        print('no pkts in flow')
        return -1
    if flow_size == 1:
        plen = 128
    else:
        plen = math.ceil(128 / flow_size)
        plen = max(plen, min_plen)
    pcap_reader = random_ip_port(pcap_reader)
    for packet in pcap_reader:
        if not packet.haslayer('IP'):
            continue
        p_data = packet['IP']
        packet_data = binascii.hexlify(bytes(p_data))
        packet_string = packet_data.decode()
        new_packet_string = packet_string[0:]
        packet_data_string = bigram_generation(new_packet_string, packet_len=plen, flag=True, num_interval=2)
        flow_data_string += packet_data_string
        flow_data_string += '@'
        if len(flow_data_string) > 3000:
            break
    if len(flow_data_string) == 0:
        return -1
    flow_data_string = flow_data_string[:-1] + '#' + str(flow_size)
    feature_data.append(flow_data_string)
    return feature_data
