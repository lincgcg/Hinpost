"""Existing field shuffle extraction operator."""
import os
import binascii
import random
from scapy.all import rdpcap, PacketList, IP, TCP, UDP, IPv6, Packet
import ipaddress
import copy
import math
dataset_dir = ''
aug_time = 1
selected_rate = 0.1
min_plen = 32

def random_ipv4():
    IPV4_MAX = ipaddress.IPv4Address._ALL_ONES
    ip_int = random.randint(0, IPV4_MAX)
    ip_str = ipaddress.IPv4Address._string_from_ip_int(ip_int)
    return ip_str

def mac_str_to_int(mac_str):
    mac_str = mac_str.replace(':', '')
    return int(mac_str, 16)

def mac_int_to_str(mac_int):
    mac_hex = format(mac_int, '012x')
    return ':'.join([mac_hex[i:i + 2] for i in range(0, 12, 2)])

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
    r_client_mac = mac_int_to_str(random_field(48))
    r_server_mac = mac_int_to_str(random_field(48))
    for packet in packets:
        if IP in packet:
            is_client = packet[IP].src == client_ip
            if 'Ethernet' in packet:
                packet['Ethernet'].src = r_client_mac if is_client else r_server_mac
                packet['Ethernet'].dst = r_server_mac if is_client else r_client_mac
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

def shuffle_N_Key(packet_string, field_pos, field_len, selected_rate=0.2, shuffle_num=1):
    result = []
    keys_name = list(field_pos.keys())
    key_num = int(selected_rate * len(keys_name))
    keys = random.sample(keys_name, key_num)
    keys.sort(key=lambda x: field_pos[x])
    begin_index = 0
    raw_packet_string = copy.deepcopy(packet_string)
    unchanged_packet_string = []
    key_str = []
    for key in keys:
        unchanged_packet_string.append(packet_string[begin_index:int(field_pos[key])])
        begin_index = int(field_pos[key] + field_len[key])
    unchanged_packet_string.append(packet_string[begin_index:])
    for key in keys:
        key_str.append(packet_string[int(field_pos[key]):int(field_pos[key] + field_len[key])])
    raw_key_str = copy.deepcopy(key_str)
    for i in range(shuffle_num):
        random.shuffle(key_str)
        shuffle_packet_string = unchanged_packet_string[0]
        for i in range(len(key_str)):
            shuffle_packet_string = shuffle_packet_string + key_str[i] + unchanged_packet_string[i + 1]
        result.append(shuffle_packet_string)
        key_str = copy.deepcopy(raw_key_str)
    return result

def parse_pkt_fields(packet: Packet):
    from hinpost.data.fields import field_offsets
    return field_offsets(packet)

def get_flow_feats(packets: PacketList | list):
    flow_size = len(packets)
    if flow_size == 1:
        plen = 256
    else:
        plen = math.ceil(128 / math.log10(flow_size))
    plen = max(plen, min_plen)
    flow_str_list = [''] * (aug_time + 1)
    for packet in packets:
        if packet.haslayer('IP'):
            p_data = packet['IP']
        else:
            continue
        (fp, fl) = parse_pkt_fields(p_data)
        packet_data = binascii.hexlify(bytes(p_data))
        packet_string = packet_data.decode()
        new_packet_string = packet_string[0:]
        packet_data_string = bigram_generation(new_packet_string, packet_len=plen, flag=True, num_interval=2)
        flow_str_list[0] += packet_data_string + ' '
        shuffle_pkt_str = shuffle_N_Key(new_packet_string, fp, fl, selected_rate, aug_time)
        for i in range(aug_time):
            temp = bigram_generation(shuffle_pkt_str[i], packet_len=plen, flag=True, num_interval=2)
            flow_str_list[i + 1] += temp
            flow_str_list[i + 1] += ' '
        if len(flow_str_list[0]) > 6000:
            break
    if len(flow_str_list[0]) == 0:
        return -1
    flow_data_string = '@@'.join([s.strip() for s in flow_str_list])
    return flow_data_string

def get_feature_packet(label_pcap, payload_len):
    if os.path.getsize(label_pcap) == 0:
        print('Current File Size = 0 !')
        return -1
    feature_data = []
    packets = rdpcap(label_pcap)
    flow_size = len(packets)
    if flow_size < 1:
        print('no pkts in flow')
        return -1
    packets = random_ip_port(packets)
    packets = list(packets)
    flow_data_string = get_flow_feats(packets)
    if flow_data_string == -1:
        return -1
    feature_data.append(flow_data_string)
    return feature_data
