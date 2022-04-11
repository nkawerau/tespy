import logging

from tespy.components import Source, Sink, Merge, Splitter, Pipe, AlkalineWaterElectrolyzer, Valve
from tespy.connections import Connection
from tespy.networks import Network

import pandas as pd

from tespy.tools import fluid_properties as fp

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 200)
pd.set_option("display.precision", 10)

"""Set up network for alkaline water electrolyzer tests."""
nw = Network(['TS::H2', 'TS::H2O', 'TS::KOH', 'TS::O2'], T_unit='C', p_unit='bar')

cell1 = AlkalineWaterElectrolyzer('cell1', pr=1)
cell2 = AlkalineWaterElectrolyzer('cell2', pr=1)
pipe1 = Pipe('pipe1', D=0.1, L=1, ks=1e-6, Q=0)
pipe2 = Pipe('pipe2', D=0.1, L=1, ks=1e-6, Q=0)
cathode1_valve = Valve('cathode1_valve1', zeta=2/(0.1**4))
anode1_valve = Valve('anode1_valve', zeta=2/(0.1**4))
cathode2_valve = Valve('cathode2_valve1', zeta=1/(0.1**4))
anode2_valve = Valve('anode2_valve', zeta=1/(0.1**4))

# component definition
source_cathode_stream = Source("source_cathode_stream")
sink_cathode_stream = Sink("sink_cathode_stream")
sink2_cathode_stream = Sink("sink2_cathode_stream")
source_anode_stream = Source("source_anode_stream")
sink_anode_stream = Sink("sink_anode_stream")
sink2_anode_stream = Sink("sink2_anode_stream")

cathode1_splitter = Splitter("cathode1_splitter", num_out=2)
anode1_splitter = Splitter("anode1_splitter", num_out=2)

cathode1_merge = Merge("cathode1_merge", num_in=2)
anode1_merge = Merge("anode1_merge", num_in=2)

nw.add_conns(Connection(source_cathode_stream, 'out1', cathode1_splitter, 'in1', "cathode1_n1"))
nw.add_conns(Connection(cathode1_splitter, 'out1', cell1, 'in1', "cathode1_n2"))
#nw.add_conns(Connection(cell1, 'out1', cathode1_merge, 'in1', "cathode1_n3"))
nw.add_conns(Connection(cell1, 'out1', cathode1_valve, 'in1', "cathode1_nx"))
nw.add_conns(Connection(cathode1_valve, 'out1', cathode1_merge, 'in1', "cathode1_n3"))
nw.add_conns(Connection(cathode1_merge, 'out1', sink_cathode_stream, 'in1', "cathode1_n4"))

nw.add_conns(Connection(source_anode_stream, 'out1', anode1_splitter, 'in1', "anode1_n1"))
nw.add_conns(Connection(anode1_splitter, 'out1', cell1, 'in2', "anode1_n2"))
#nw.add_conns(Connection(cell1, 'out2', anode1_merge, 'in1', "anode1_n3"))
nw.add_conns(Connection(cell1, 'out2', anode1_valve, 'in1', "anode1_nx"))
nw.add_conns(Connection(anode1_valve, 'out1', anode1_merge, 'in1', "anode1_n3"))
nw.add_conns(Connection(anode1_merge, 'out1', sink_anode_stream, 'in1', "anode1_n4"))

nw.add_conns(Connection(cathode1_splitter, 'out2', cell2, 'in1', "cathode2_n2"))
#nw.add_conns(Connection(cell2, 'out1', cathode1_merge, 'in2', "cathode2_n3"))
nw.add_conns(Connection(cell2, 'out1', cathode2_valve, 'in1', "cathode2_nx"))
nw.add_conns(Connection(cathode2_valve, 'out1', cathode1_merge, 'in2', "cathode2_n3"))
#nw.add_conns(Connection(cell2, 'out1', sink2_cathode_stream, 'in1', "cathode2_n3"))


nw.add_conns(Connection(anode1_splitter, 'out2', cell2, 'in2', "anode2_n2"))
#nw.add_conns(Connection(cell2, 'out2', anode1_merge, 'in2', "anode2_n3"))
nw.add_conns(Connection(cell2, 'out2', anode2_valve, 'in1', "anode2_nx"))
nw.add_conns(Connection(anode2_valve, 'out1', anode1_merge, 'in2', "anode2_n3"))
#nw.add_conns(Connection(cell2, 'out2', sink2_anode_stream, 'in1', "anode2_n3"))


"""Test component properties of alkaline water electrolyzer."""
number_of_cells = 2
mass_flow = 0.5
p0 = 1
T0 = 30


"""Better starting values"""
if False:
    flow = [0 for i in range(4)]
    flow[0] = mass_flow
    flow[1] = p0*1e5
    flow[2] = 1
    flow[3] = {"H2": 0, "H2O": 0.7, "KOH": 0.3, "O2": 0}
    # enthalpy in [J/kg]
    h0 = fp.h_mix_pT(flow, T0+273.15)/1e3

    for connections in ["cathode1_n1", "cathode1_n4", "anode1_n1", "anode1_n4"]:
        nw.get_conn(connections).set_attr(p0=p0, h0=h0, m0=mass_flow - (mass_flow * ((1 - 1) / number_of_cells)))

    for i in range(1, number_of_cells + 1):
        for connections in [f"cathode{i}_n2", f"cathode{i}_n3"]:
            nw.get_conn(connections).set_attr(p0=p0, h0=h0, m0=mass_flow / number_of_cells)

    for i in range(1, number_of_cells + 1):
        for connections in [f"anode{i}_n2", f"anode{i}_n3"]:
            nw.get_conn(connections).set_attr(p0=p0, h0=h0, m0=mass_flow / number_of_cells)


#nw.set_attr(m_range=[1e-3, mass_flow], p_range=[p0 - 0.5, p0])  #, h_range=[h0, h0])

for half_cell in ["cathode1_n1", "anode1_n1"]:
    nw.get_conn(half_cell).set_attr(
        T=T0,
        p=p0,
        m=mass_flow,
        fluid={
            'H2': 0,
            'H2O': 0.7,
            'KOH': 0.3,
            'O2': 0,
        },
    )

# nw.get_conn("anode1_n2").set_attr(m=mass_flow/2)
# nw.get_conn("cathode1_n2").set_attr(m=mass_flow/2)

nw.solve('design')
nw.print_results()
print(nw.results['Connection'])
