"""Offsets for the fields already used by the supplied augmentation operators."""


def field_offsets(packet):
    """Return hexadecimal offsets, accounting for variable IP header length.

    The eligible fields match the legacy parser: Ethernet, IPv4 and TCP
    descriptors with integral hexadecimal offsets and lengths, excluding options,
    TCP checksum and urgent pointer. Options are retained but never selected.
    """
    positions, lengths = {}, {}
    offset = 0
    for layer_name, prefix in (("Ethernet", "Ethernet"), ("IP", "IP"), ("TCP", "TCP")):
        if not packet.haslayer(layer_name):
            continue
        layer = packet[layer_name]
        start = offset
        for field in layer.fields_desc:
            if field.name == "options":
                continue
            length = field.sz * 2
            name = prefix + field.name
            if offset % 1 == 0 and length % 1 == 0 and name not in {"TCPchksum", "TCPurgptr"}:
                positions[name], lengths[name] = int(offset), int(length)
            offset += length
        if layer_name == "IP":
            # The serialized IHL includes IP options, even when Scapy's ihl is None.
            header_bytes = (bytes(layer)[0] & 0x0F) * 4
            offset = start + header_bytes * 2
    return positions, lengths
