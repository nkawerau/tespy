from tespy.components import Sink, Source, Splitter, Merge, Pipe, Valve, TJunctionSplitter
from tespy.connections import Connection
from tespy.networks import Network
import numpy as np


velocity = 1  # m/s
d = 0.1
A_channel = d**2 * (np.pi/4)
volume_flow = 2 * velocity * A_channel # m3/s
rho = 1000  # kg/m3
mass_flow = volume_flow * rho # kg/s

source = Source("source")
sink1 = Sink("sink1")
sink2 = Sink("sink2")

splitter = Splitter("splitter")
new_splitter = TJunctionSplitter("splitter", num_out=2, zeta1=1.001/(d**4), zeta2=1/(d**4))

merge = Merge("merge")

valve1 = Valve("valve1", zeta=2/(d**4))
valve2 = Valve("valve2", zeta=1/(d**4))

pipe1 = Pipe("pipe1", ks=0.000005, L=1, D=d, Q=0)
pipe2 = Pipe("pipe2", ks=0.000005, L=1, D=d, Q=0)

fluid_list = ["HEOS::H2O"]

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

T0 = 30
p0 = 2

fn.set_attr(m_range=[1e-3, mass_flow], p_range=[p0-0.1, p0])

# splitter - valve system with velocity = 1m/s, zeta = 1 --> dp=5 mbar
if False:
    source_splitter = Connection(source, "out1", splitter, "in1", "source-splitter")

    splitter_valve1 = Connection(splitter, "out1", valve1, "in1", "splitter-valve1")
    splitter_valve2 = Connection(splitter, "out2", valve2, "in1", "splitter-valve2")

    valve1_sink1 = Connection(valve1, "out1", sink1, "in1", "valve1-sink1")
    valve2_sink2 = Connection(valve2, "out1", sink2, "in1", "valve2-sink2")

    fn.add_conns(source_splitter, splitter_valve1, splitter_valve2, valve1_sink1, valve2_sink2)

    fn.get_conn("source-splitter").set_attr(T=T0, p=p0, m=mass_flow, fluid={"H2O": 1})
    fn.get_conn("valve1-sink1").set_attr(m=mass_flow/2)

# splitter - valve - merge system with velocity = 1m/s, zeta = 1 --> dp=5 mbar
if False:
    way1 = mass_flow/2
    way2 = mass_flow - way1

    source_splitter = Connection(source, "out1", splitter, "in1", "source-splitter")
    #source_splitter.m.val0 = way1 + way2

    splitter_valve1 = Connection(splitter, "out1", valve1, "in1", "splitter-valve1")
    #splitter_valve1.m.val0 = way1

    splitter_valve2 = Connection(splitter, "out2", valve2, "in1", "splitter-valve2")
    #splitter_valve2.m.val0 = way2

    valve1_merge = Connection(valve1, "out1", merge, "in1", "valve1-merge")
    #valve1_merge.m.val0 = way1

    valve2_merge = Connection(valve2, "out1", merge, "in2", "valve2-merge")
    #valve2_merge.m.val0 = way2

    merge_sink1 = Connection(merge, "out1", sink1, "in1", "valve1-sink1")
    #merge_sink1.m.val0 = way1 + way2

    fn.add_conns(source_splitter, splitter_valve1, splitter_valve2, valve1_merge, valve2_merge, merge_sink1)

    fn.get_conn("source-splitter").set_attr(T=T0, p=p0, m=mass_flow, fluid={"H2O": 1})

# splitter system
if False:
    source_splitter = Connection(source, "out1", new_splitter, "in1", "source-splitter")
    splitter_sink1 = Connection(new_splitter, "out1", sink1, "in1", "splitter-sink1")
    splitter_sink2 = Connection(new_splitter, "out2", sink2, "in1", "splitter-sink2")

    fn.add_conns(source_splitter, splitter_sink1, splitter_sink2)

    fn.get_conn("source-splitter").set_attr(T=T0, p=p0, m=mass_flow, fluid={"H2O": 1})
    fn.get_conn("splitter-sink1").set_attr(m=mass_flow/2)

# splitter - merge system
if True:
    source_splitter = Connection(source, "out1", new_splitter, "in1", "source-splitter")
    #source_splitter.m.val0 = mass_flow

    splitter_merge1 = Connection(new_splitter, "out1", merge, "in1", "splitter-merge1")
    #splitter_merge1.m.val0 = mass_flow/2
    splitter_merge2 = Connection(new_splitter, "out2", merge, "in2", "splitter-merge2")
    #splitter_merge1.m.val0 = mass_flow/2

    merge_sink1 = Connection(merge, "out1", sink1, "in1", "merge-sink")
    #merge_sink1.m.val0 = mass_flow

    fn.add_conns(source_splitter, splitter_merge1, splitter_merge2, merge_sink1)

    fn.get_conn("source-splitter").set_attr(T=T0, p=p0, m=mass_flow, fluid={"H2O": 1}, state='l')
    fn.get_conn("splitter-merge1").set_attr(state='l')
    fn.get_conn("splitter-merge2").set_attr(state='l')
    fn.get_conn("merge-sink").set_attr(state='l')

# splitter - pipe - merge system
if False:
    source_splitter = Connection(source, "out1", new_splitter, "in1", "source-splitter")

    splitter_pipe1 = Connection(new_splitter, "out1", pipe1, "in1", "splitter-pipe1")
    splitter_pipe2 = Connection(new_splitter, "out2", pipe2, "in1", "splitter-pipe2")

    pipe1_merge = Connection(pipe1, "out1", merge, "in1", "pipe1-merge")
    pipe2_merge = Connection(pipe2, "out1", merge, "in2", "pipe2-merge")

    merge_sink1 = Connection(merge, "out1", sink1, "in1", "pipe1-sink1")

    fn.add_conns(source_splitter, splitter_pipe1, splitter_pipe2, pipe1_merge, pipe2_merge, merge_sink1)

    fn.get_conn("source-splitter").set_attr(T=T0, p=p0, m=mass_flow, fluid={"H2O": 1})
    #fn.get_conn("pipe1-sink1").set_attr(m=m0/2)


fn.solve(mode="design")
fn.print_results()