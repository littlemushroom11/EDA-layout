#! anaconda/bin/python3
# **************************************************************************************
# NOTICE OF COPYRIGHT AND OWNERSHIP OF SOFTWARE:
# This file is part of the AutoCell project.
# Copyright (c) 2020-2020 Primarius Technologies Co., Ltd. All Rights Reserved.
#
# This computer program is the property of Primarius Technologies Co., Ltd. of
# B-5, No. 1768 Xinluo Avenue,High-Tech Industrial Development Zone, Jinan 250101, P. R. China
# Any use, copy, publication, distribution, display, modification, or transmission of this computer
# program in whole or in part in any form or by any means without
# the prior express written permission of Primarius Technologies Co., Ltd. is
# strictly prohibited.
#
# Except when expressly provided by Primarius Technologies Co., Ltd. in writing,
# possession of this computer program shall not be construed to confer any license
# or rights under any of Primarius Technologies Co., Ltd.'s intellectual property
# rights, whether by estoppel, implication, or otherwise.
#
# ALL COPIES OF THIS PROGRAM MUST DISPLAY THIS NOTICE OF COPYRIGHT AND OWNERSHIP IN FULL.
# ****************************************************************************************/

from enum import Enum
from typing import Any, Tuple

import re
import numpy
import math
import networkx as nx


def is_power(net: str):
    return net.upper() == "VDD" or net.upper() == "VSS"

def str_equal(s1: str, s2: str) -> bool:
    return s1.upper() == s2.upper()

class ChannelType(Enum):
    NMOS = 0,
    PMOS = 1


class NetRange:
    def __init__(self, x1, x2):
        self.x1 = x1 if x1 < x2 else x2
        self.x2 = x2 if x2 > x1 else x1

    def update(self, x):
        if x < self.x1:
            self.x1 = x
        elif x > self.x2:
            self.x2 = x

    def length(self):
        return self.x2 - self.x1

    def center(self):
        return (self.x2 - self.x1) / 2


class Transistor:
    """
    Abstract representation of a MOS transistor.
    """

    def __init__(self, channel_type: ChannelType,
                 source_net: str, gate_net: str, drain_net: str,
                 channel_width, name: str):
        """
        params:
        left: Either source or drain net.
        right: Either source or drain net.
        """
        self.name = name
        self.channel_type = channel_type
        self.source_net = source_net
        self.gate_net = gate_net
        self.drain_net = drain_net
        self.channel_width = channel_width

    def terminals(self) -> Tuple[Any, Any, Any]:
        """ Return a tuple of all terminal names.
        :return:
        """
        return self.source_net, self.gate_net, self.drain_net

    def __key(self):
        return self.name, self.channel_type, self.source_net, self.gate_net, self.drain_net, self.channel_width

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, y):
        return self.__key() == y.__key()


class TransistorRef:

    def __init__(self, transistor, flip, width):
        self.transistor = transistor
        self.flip = flip
        self.width = width

    def source(self):
        return self.transistor.drain_net if self.flip else self.transistor.source_net

    def drain(self):
        return self.transistor.source_net if self.flip else self.transistor.drain_net

    def gate(self):
        return self.transistor.gate_net

    def __repr__(self):
        return "({}, {}, {})".format(self.source(), self.gate(), self.drain())


class Cell:
    """ Dual row cell.
    """

    def __init__(self, name, pins):
        self.name = name
        self.pins = pins
        self.net_pos = dict()
        self.upper = []
        self.lower = []

        self.ref_width = 0
        self.width = 0
        self.bbox = 0
        self.pin_access = 0
        self.symmetric = 10
        self.drc = 10
        self.score = 0

    def reset(self, width):
        # self.ref_width = ref_width
        self.width = width
        self.upper = [None] * width
        self.lower = [None] * width

    def add_net(self, net, x):

        if net in self.net_pos.keys():
            self.net_pos[net].append(x)
        else:
            self.net_pos[net] = [x]

    def add_transistor(self, t: TransistorRef, x):
        if t.transistor.channel_type == ChannelType.PMOS:
            self.upper[x] = t
        else:
            self.lower[x] = t

        self.add_net(t.source(), x - 0.5)
        self.add_net(t.gate(), x)
        self.add_net(t.drain(), x + 0.5)

    def check(self, ref_dic) -> bool:
        # check transistor width
        transistor_width = {}
        for t in self.upper + self.lower:
            if t:
                if t.transistor.name not in transistor_width.keys():
                    transistor_width[t.transistor.name] = 0
                transistor_width[t.transistor.name] += t.width
                if t.width > 220 or t.width < 120:
                    print("ERROR: transistor %s width %d unlegal\n" % (t.transistor.name, t.width))
                    self.score = 0
                    return False

        for name, transistor in ref_dic.items():
            if transistor_width[name] != transistor.channel_width:
                print("ERROR: transistor %s width lost\n" % name)
                self.score = 0
                return False
        # check diffusion sharing
        for i in range(self.width):
            pmos1 = self.upper[i]
            nmos1 = self.lower[i]
            if (pmos1 and not nmos1) or (not pmos1 and nmos1):
                self.symmetric -= 1

            if pmos1 and nmos1 and pmos1.gate() != nmos1.gate():
                print("ERROR: position %d gate not match\n" % i)
                return False

            if i < self.width - 1:
                pmos2 = self.upper[i + 1]
                nmos2 = self.lower[i + 1]
                if pmos1 and pmos2 and pmos1.drain() != pmos2.source():
                    print("ERROR: position %d diffusion unsharing\n" % i)
                    return False
                if nmos1 and nmos2 and nmos1.drain() != nmos2.source():
                    print("ERROR: position %d diffusion unsharing\n" % i)
                    return False

                if i > 0:
                    pmos0 = self.upper[i - 1]
                    nmos0 = self.lower[i - 1]
                    if pmos1 and pmos2 and pmos0 and pmos1.width < pmos2.width and pmos1.width < pmos0.width:
                        self.drc -= 10
                    if nmos1 and nmos2 and nmos0 and nmos1.width < nmos2.width and nmos1.width < nmos0.width:
                        self.drc -= 10

        return True

    def get_pin_access(self):
        pin_coords = []

        for net, r in self.net_pos.items():
            if net in self.pins:
                pin_coords.append(r[0])
                max_distance = 0
                for pos in r:
                    distance = 0
                    another_pos = []
                    for another_n, another_r in self.net_pos.items():
                        if another_n in self.pins and another_n != net:
                            another_pos.extend(another_r)

                    if not another_pos:
                        break
                    another_pos.sort()
                    if another_pos[0] > pos:
                        distance = abs(another_pos[0] - pos)
                    elif another_pos[-1] < pos:
                        distance = abs(another_pos[-1] - pos)
                    else:
                        for i in range(0, len(another_pos) - 1):
                            if another_pos[i] < pos < another_pos[i + 1]:
                                distance = min(abs(another_pos[i] - pos), abs(another_pos[i + 1] - pos))
                                break
                    if distance > max_distance:
                        max_distance = distance
                        pin_coords[-1] = pos

        pin_coords.sort()
        if not pin_coords or len(pin_coords) == 1:
            self.pin_access = 1
        else:
            pin_spacing = []
            left_spacing = pin_coords[0] + 0.5
            right_spacing = self.width - 0.5 - pin_coords[-1]
            if left_spacing > 1:
                pin_spacing.append(left_spacing / self.width)
            if right_spacing > 1:
                pin_spacing.append(right_spacing / self.width)
            for i in range(0, len(pin_coords) - 1):
                pin_spacing.append((pin_coords[i + 1] - pin_coords[i]) / self.width)

            self.pin_access = numpy.std(numpy.array(pin_spacing))

    def evaluate(self, runtime):
        for net, r in self.net_pos.items():
            r.sort()
            if not is_power(net):
                self.bbox += r[-1] - r[0]
        self.get_pin_access()

        ws = 40 * (1 - (self.width - self.ref_width) / (self.ref_width + 20))
        bs = min(20.0, 20 * (1 - (self.bbox - self.ref_width * (len(self.pins) - 1)) / 60))
        ps = 10 * (1 - self.pin_access)
        ss = self.symmetric
        ds = self.drc
        rs = 10 * (1 / (1 + math.exp(runtime / 3600 - 1)))
        self.score = ws + bs + ps + ss + ds + rs
        print("Cell score %f (width: %d, bbox: %f, pin_access: %f, symmetric: %d, drc: %d, runtime: %ds)" \
              % (self.score, self.width, self.bbox, self.pin_access, self.symmetric, self.drc, runtime))

    def __repr__(self):
        return (
                " | ".join(['{:^16}'.format(str(t)) for t in self.upper]) +
                "\n" +
                " | ".join(['{:^16}'.format(str(t)) for t in self.lower])
        )


class EulerGraph:

    def __init__(self, refs):
        self.refs = refs
        self.graph = nx.MultiGraph()
        self.build_graph()

    def build_graph(self):
        for r in self.refs:
            if r is not None:
                self.graph.add_node(r.source())
                self.graph.add_node(r.drain())
                self.graph.add_edge(r.source(), r.drain())

    def get_odd_num(self):
        odd_nodes = [node for node, degree in self.graph.degree if degree % 2 != 0]
        return len(odd_nodes)


def get_channel_type(model) -> ChannelType:
    return ChannelType.PMOS if str_equal(model, "pch_mac") else ChannelType.NMOS


def get_channel_width(width_str: str):
    words = width_str.split("=")
    w = float(words[1][:-1])
    unit = words[1][-1]
    return w * 1000 if str_equal(unit, 'u') else w


def decompose_transistor_name(tname: str):
    words = tname.split("_finger")
    return words[0], 0 if len(words) == 1 else words[1]


def load_netlist(file, cell_name):
    input_stream = open(file, "r")
    find_cell = False
    transistor_dic = {}
    pin_list = set()
    while True:
        line = input_stream.readline()
        if not line:
            break
        words = line.split()
        if find_cell and words[0] == ".ENDS":
            break

        if find_cell:
            w = 0.00
            for word in words[6:]:
                if word[0] == 'w' or word[0] == 'W':
                    w = get_channel_width(word)
                    break

            t = Transistor(get_channel_type(words[5]), words[3], words[2], words[1], w, words[0][1:])
            transistor_dic[t.name] = t

        if len(words) > 1 and cell_name == words[1]:
            for word in words[2:]:
                if not is_power(word):
                    pin_list.add(word)
            find_cell = True

    return transistor_dic, pin_list


def extract_subckt(file, cell_re):
    input_stream = open(file, "r")
    output_stream = open("visible_cases.spi", "a")

    find_cell = False
    while True:
        line = input_stream.readline()
        if not line:
            break
        words = line.split()
        if find_cell and words[0] == ".ENDS":
            output_stream.write(line)
            find_cell = False
            continue

        if find_cell:
            output_stream.write(line)
            continue

        if len(words) > 1 and re.match(cell_re, words[1]):
            output_stream.write(line)
            find_cell = True


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 4:
        print("ERROR: No placement file provided.\nUsage: python main.py <placement_file> <cell_name> <netlist>")
        exit()

    # read placement file
    placement_file = sys.argv[1]
    placement_stream = open(placement_file, "r")
    placement_dic = json.load(placement_stream)
    placement = placement_dic["placement"]
    cell_name = sys.argv[2]

    # get transistor properties from netlist
    netlist_file = sys.argv[3]
    transistor_dic, pins = load_netlist(netlist_file, cell_name)
    # get ref width
    ref_width = 0
    for transistor_name, t in transistor_dic.items():
        if t.channel_width > 220:
            ref_width += t.channel_width // 200
        else:
            ref_width += 1

    cell = Cell(cell_name, pins)
    # get cell width
    width = 0
    for transistor_name, properties in placement.items():
        if width < int(properties["x"]) + 1:
            width = int(properties["x"]) + 1
    cell.reset(width)
    # add transistor
    for transistor_name, properties in placement.items():
        tname, finger = decompose_transistor_name(transistor_name)
        transistor = transistor_dic[tname]
        ref = TransistorRef(transistor, not str_equal(transistor.source_net, properties["source"]), int(properties["width"]))
        cell.add_transistor(ref, int(properties["x"]))

    # set ref width
    upper_graph = EulerGraph(cell.upper)
    lower_graph = EulerGraph(cell.lower)
    min_gap = max(0.0, (upper_graph.get_odd_num() + lower_graph.get_odd_num() - 4) / 2)
    cell.ref_width = (min_gap + ref_width) / 2

    # check and get score
    if cell.check(transistor_dic):
        cell.evaluate(runtime=0)


