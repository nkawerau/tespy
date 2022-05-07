import logging

from tespy.components import Source, Sink, Merge, Splitter, Subsystem, AlkalineWaterElectrolyzer
from tespy.connections import Connection
from tespy.networks import Network
import numpy as np

import pandas as pd

pd.set_option("display.max_rows", 20)
pd.set_option("display.min_rows", 20)
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 200)
pd.set_option("display.precision", 10)
pd.options.display.float_format = "{:,.10f}".format

""" TESPy subsystems for the fluid network """


class Cell(Subsystem):
    """Class documentation"""

    def __init__(self, label, last_cell):

        if not isinstance(label, str):
            msg = "Subsystem label must be of type str!"
            logging.error(msg)
            raise ValueError(msg)

        elif len([x for x in [";", ", ", "."] if x in label]) > 0:
            msg = "Can't use " + str([";", ", ", "."]) + " in label."
            logging.error(msg)
            raise ValueError(msg)
        else:
            self.label = label
            self.last_cell = last_cell

        self.comps = {}  # dictionary for the attribute components
        self.conns = {}  # dictionary for the attribute connections
        self.create_comps(label, last_cell)
        self.create_conns(label, last_cell)

        self.list_of_pipes = []

    def create_comps(self, label, last_cell):
        """Create the subsystem's components."""

        if not self.last_cell:
            self.comps[self.label + "_splitter"] = Splitter(
                self.label + "_splitter", num_out=2
            )

            self.comps[self.label + "_merge"] = Merge(
                self.label + "_merge", num_in=2
            )

        else:
            self.comps[self.label + "_splitter"] = Splitter(
                self.label + "_splitter", num_out=1
            )

            self.comps[self.label + "_merge"] = Merge(
                self.label + "_merge", num_in=1
            )

        if 'cathode' in self.label:
            self.comps[self.label + "_ael_cell"] = AlkalineWaterElectrolyzer(self.label + "_ael_cell",
                                                                             pr=1,
                                                                             cell_current=800,
                                                                             cell_voltage=1.8,
                                                                             cell_surface=2710,)

    def create_conns(self, label, last_cell):
        """Define the subsystem's connections."""
        if 'cathode' in self.label:
            self.conns["splitter-ael_cell"] = Connection(
                self.comps[self.label + "_splitter"],
                "out1",
                self.comps[self.label + "_ael_cell"],
                'in1',
                self.label + "_n2",
            )
            self.conns["ael_cell-merge"] = Connection(
                self.comps[self.label + "_ael_cell"],
                'out1',
                self.comps[self.label + "_merge"],
                "in1",
                self.label + "_n3",
            )

        """
        n1 = connection source(out1) with splitter(in1)
        n2 = connection splitter(out1) with ael-cell(in1/in2)
        n3 = connection ael-cell(out1/out2) with merge(in1)
        n4 = connection merge(out1) with sink(in1)
        """


"""Set up network for alkaline water electrolyzer tests."""
# system properties
number_of_cells = 1
reference_number_of_cells = 95

H2_mass_fraction = 0

H2O_mass_fraction = 0.7 - H2_mass_fraction

KOH_mass_fraction = 1 - H2O_mass_fraction - H2_mass_fraction

O2_mass_fraction4 = 0

fluid_list = [
    "TS::H2",
    "TS::H2O",
    "TS::KOH",
    "TS::O2",
]

# fluid network
fn = Network(
    fluids=fluid_list,
    m_unit="kg / s",
    p_unit="bar",
    h_unit="kJ / kg",
    T_unit="C",
    v_unit="m3 / h",
    vol_unit="m3 / kg",
    iterinfo=True,
)

# component definition
source_cathode_stream = Source("source_cathode_stream")
sink_cathode_stream = Sink("sink_cathode_stream")
source_anode_stream = Source("source_anode_stream")
sink_anode_stream = Sink("sink_anode_stream")

last_cell = False
list_of_cathode_subsystems = []
list_of_anode_subsystems = []

# system stream elements

for i in range(1, number_of_cells + 1):

    if i == number_of_cells:
        last_cell = True
    # cathode stream system
    # adding subsystem cathode{n} to the list list_of_cathode_subsystems
    list_of_cathode_subsystems.append(Cell(f"cathode{i}", last_cell))

    if i == 1:
        # defining connection between source_cathode_stream and cell1
        fn.add_conns(
            Connection(
                source_cathode_stream,
                "out1",
                list_of_cathode_subsystems[i - 1].comps[f"cathode{i}" + "_splitter"],
                "in1",
                f"cathode{i}" + "_n1",
            )
        )

        # adding subsystem cell1 with 1 connection (n2 - n3) to the network
        fn.add_subsys(list_of_cathode_subsystems[i - 1])

        # defining connection between cell1 and sink_cathode_stream
        fn.add_conns(
            Connection(
                list_of_cathode_subsystems[i - 1].comps[f"cathode{i}" + "_merge"],
                "out1",
                sink_cathode_stream,
                "in1",
                f"cathode{i}" + "_n4",
            )
        )

    if i != 1:
        # defining connections between cells and adding them to the network
        fn.add_conns(
            Connection(
                list_of_cathode_subsystems[i - 2].comps[f"cathode{i - 1}" + "_splitter"],
                "out2",
                list_of_cathode_subsystems[i - 1].comps[f"cathode{i}" + "_splitter"],
                "in1",
                f"cathode{i}" + "_n1",
            )
        )

        # adding subsystem with 1 connections (n2 - n3) to the network
        fn.add_subsys(list_of_cathode_subsystems[i - 1])

        fn.add_conns(
            Connection(
                list_of_cathode_subsystems[i - 1].comps[f"cathode{i}" + "_merge"],
                "out1",
                list_of_cathode_subsystems[i - 2].comps[f"cathode{i - 1}" + "_merge"],
                "in2",
                f"cathode{i}" + "_n4",
            )
        )

    # anode stream system

    # adding subsystem anode{n} to the list list_of_anode_subsystems
    list_of_anode_subsystems.append(Cell(f"anode{i}", last_cell))

    if i == 1:
        # defining connection between source_anode_stream and cell1 (n1)
        fn.add_conns(
            Connection(
                source_anode_stream,
                "out1",
                list_of_anode_subsystems[i - 1].comps[f"anode{i}" + "_splitter"],
                "in1",
                f"anode{i}" + "_n1",
            )
        )

        fn.add_conns(
            Connection(
                list_of_anode_subsystems[i - 1].comps[f"anode{i}" + "_splitter"],
                "out1",
                list_of_cathode_subsystems[i - 1].comps[f"cathode{i}" + "_ael_cell"],
                "in2",
                f"anode{i}" + "_n2",
            )
        )

        fn.add_conns(
            Connection(
                list_of_cathode_subsystems[i - 1].comps[f"cathode{i}" + "_ael_cell"],
                "out2",
                list_of_anode_subsystems[i - 1].comps[f"anode{i}" + "_merge"],
                "in1",
                f"anode{i}" + "_n3",
            )
        )

        # defining connection between cell1 and sink_anode_stream
        fn.add_conns(
            Connection(
                list_of_anode_subsystems[i - 1].comps[f"anode{i}" + "_merge"],
                "out1",
                sink_anode_stream,
                "in1",
                f"anode{i}" + "_n4",
            )
        )

    if i != 1:
        # defining connections between cells and adding them to the network
        fn.add_conns(
            Connection(
                list_of_anode_subsystems[i - 2].comps[f"anode{i - 1}" + "_splitter"],
                "out2",
                list_of_anode_subsystems[i - 1].comps[f"anode{i}" + "_splitter"],
                "in1",
                f"anode{i}" + "_n1",
            )
        )

        fn.add_conns(
            Connection(
                list_of_anode_subsystems[i - 1].comps[f"anode{i}" + "_splitter"],
                "out1",
                list_of_cathode_subsystems[i - 1].comps[f"cathode{i}" + "_ael_cell"],
                "in2",
                f"anode{i}" + "_n2",
            )
        )

        fn.add_conns(
            Connection(
                list_of_cathode_subsystems[i - 1].comps[f"cathode{i}" + "_ael_cell"],
                "out2",
                list_of_anode_subsystems[i - 1].comps[f"anode{i}" + "_merge"],
                "in1",
                f"anode{i}" + "_n3",
            )
        )

        # defining connection between cell1 and sink_anode_stream
        fn.add_conns(
            Connection(
                list_of_anode_subsystems[i - 1].comps[f"anode{i}" + "_merge"],
                "out1",
                list_of_anode_subsystems[i - 2].comps[f"anode{i - 1}" + "_merge"],
                "in2",
                f"anode{i}" + "_n4",
            )
        )

"""Test component properties of alkaline water electrolyzer."""
mass_flow = 1e-2
p0 = 16

for half_cell in ["cathode1_n1", "anode1_n1"]:
    fn.get_conn(half_cell).set_attr(T=70, p=p0, m=mass_flow*number_of_cells,
                                    fluid={'H2': H2_mass_fraction, 'H2O': H2O_mass_fraction, 'KOH': KOH_mass_fraction, 'O2': 0})

for k in range(1, number_of_cells + 1):
    for connections in [f"cathode{k}_n2", f"cathode{k}_n3"]:
        fn.get_conn(connections).set_attr(p0=p0, h0=135, m0=mass_flow / number_of_cells)

    for connections in [f"cathode{k}_n1", f"cathode{k}_n4"]:
        fn.get_conn(connections).set_attr(p0=p0, h0=135, m0=mass_flow - (mass_flow * ((k - 1) / number_of_cells)))

for l in range(1, number_of_cells + 1):
    for connections in [f"anode{l}_n2", f"anode{l}_n3"]:
        fn.get_conn(connections).set_attr(p0=p0, h0=135, m0=mass_flow / number_of_cells)

    for connections in [f"anode{l}_n1", f"anode{l}_n4"]:
        fn.get_conn(connections).set_attr(p0=p0, h0=135, m0=mass_flow - (mass_flow * ((l - 1) / number_of_cells)))

#fn.set_attr(m_range=[1e-2, mass_flow], p_range=[p0, p0])#, h_range=[134.5, 135.5])
#m_range=[1e-2, mass_flow],

fn.solve('design')
#fn.print_results()

# compare each pressure drop of a specific component of all cells with pandas
results = fn.results["Connection"]
results.insert(0, column='name', value=results.index)
results = results.reset_index(drop=True)
results = results.drop(columns=["s", "x", "Td_bp"])

for i in range(len(results)):
    if "v" in results.name[i]:
        results = results.drop(index=[i])

results = results.reset_index(drop=True)

# get data frame for specific connection point
for i in range(1, 4):
    locals()[f"n{i}"] = results.loc[results.name.str.contains(f"n{i}")].reset_index(
        drop=True
    )

pressure_drops = pd.DataFrame(columns=['name', 'total', 'diversion', 'straight'])
for index in range(len(n1)):
    pressure_drops.loc[index, 'name'] = n1.name.loc[index][:-3]
pressure_drops.total = (n1.p - n3.p) * 1000
pressure_drops.diversion = (n2.p - n3.p) * 1000

for index in range(len(n1)-2):
    pressure_drops.straight = (n1.loc[index, 'p'] - n1.loc[index+2, 'p']) * 1000

pressure_drops['relative_flow'] = ''
for i in range(len(pressure_drops)):
    if pressure_drops.name.str.contains("cathode")[i]:
        pressure_drops.loc[i, "relative_flow"] = (n2.m.iloc[i]  / n2.m.iloc[0]) * 100
    else:
        pressure_drops.loc[i, "relative_flow"] = (n2.m.iloc[i]  / n2.m.iloc[1]) * 100


print("\nconnection results:\n", results)
print("\npressure drops:\n", pressure_drops)

