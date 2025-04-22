from scapy.all import sniff, ARP, DNS, IP, UDP
import pandas as pd
import time

data = []

def process_packet(pkt):
    features = {}
    features['timestamp'] = time.time()
    
    if ARP in pkt:
        features['type'] = 'ARP'
        features['src_ip'] = pkt[ARP].psrc
        features['dst_ip'] = pkt[ARP].pdst
        features['src_mac'] = pkt[ARP].hwsrc
        features['dst_mac'] = pkt[ARP].hwdst
        features['op_code'] = pkt[ARP].op
    
    elif DNS in pkt and pkt.haslayer(DNSRR):
        features['type'] = 'DNS'
        features['src_ip'] = pkt[IP].src
        features['dst_ip'] = pkt[IP].dst
        features['dns_query'] = pkt[DNS].qd.qname.decode()
        features['dns_response_ip'] = pkt[DNSRR].rdata
        features['ttl'] = pkt[IP].ttl
    
    if features:
        data.append(features)
        df = pd.DataFrame(data)
        df.to_csv("network_log.csv", index=False)

sniff(prn=process_packet, store=False)

This is to store real time info onto csv file to train the model
