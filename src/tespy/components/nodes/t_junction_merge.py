# -*- coding: utf-8

"""Module of class Merge.


This file is part of project TESPy (github.com/oemof/tespy). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location tespy/components/nodes/merge.py

SPDX-License-Identifier: MIT
"""

import numpy as np

from tespy.components.nodes.base import NodeBase
from tespy.tools.data_containers import DataContainerSimple as dc_simple
from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.document_models import generate_latex_eq
from tespy.tools.fluid_properties import s_mix_pT
from tespy.tools.helpers import num_fluids

from tespy.tools.fluid_properties import v_mix_ph


class TJunctionMerge(NodeBase):
    r"""
    Class for merge points with multiple inflows and one outflow.

    **Mandatory Equations**

    - :py:meth:`tespy.components.nodes.base.NodeBase.mass_flow_func`
    - :py:meth:`tespy.components.nodes.base.NodeBase.pressure_equality_func`
    - :py:meth:`tespy.components.nodes.merge.Merge.fluid_func`
    - :py:meth:`tespy.components.nodes.merge.Merge.energy_balance_func`

    Inlets/Outlets

    - specify number of outlets with :code:`num_in` (default value: 2)
    - out1

    Image

    .. image:: _images/Merge.svg
       :alt: alternative text
       :align: center

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    design_path : str
        Path to the components design case.

    local_offdesign : boolean
        Treat this component in offdesign mode in a design calculation.

    local_design : boolean
        Treat this component in design mode in an offdesign calculation.

    char_warnings : boolean
        Ignore warnings on default characteristics usage for this component.

    printout : boolean
        Include this component in the network's results printout.

    num_in : float, dict
        Number of inlets for this component, default value: 2.

    Example
    -------
    The merge mixes a specified number of mass flows and has a single outlet.
    At the outlet, fluid composition and enthalpy are calculated by mass
    weighted fluid composition and enthalpy of the inlets.

    >>> from tespy.components import Sink, Source, Merge
    >>> from tespy.connections import Connection
    >>> from tespy.networks import Network
    >>> import shutil
    >>> import numpy as np
    >>> fluid_list = ['O2', 'N2']
    >>> nw = Network(fluids=fluid_list, p_unit='bar', iterinfo=False)
    >>> so1 = Source('source1')
    >>> so2 = Source('source2')
    >>> so3 = Source('source3')
    >>> si1 = Sink('sink')
    >>> m = Merge('merge', num_in=3)
    >>> m.component()
    'merge'
    >>> inc1 = Connection(so1, 'out1', m, 'in1')
    >>> inc2 = Connection(so2, 'out1', m, 'in2')
    >>> inc3 = Connection(so3, 'out1', m, 'in3')
    >>> outg = Connection(m, 'out1', si1, 'in1')
    >>> nw.add_conns(inc1, inc2, inc3, outg)

    A merge with three inlets mixes air (simplified) with pure nitrogen and
    pure oxygen. All gases enter the component at the same temperature. As
    mixing effects are not considered, the outlet temperature should thus be
    similar to the three inlet temperatures (difference might occur due to
    rounding in fluid property functions, let's check it for two different
    temperatures). It is e.g. possible to find the required mass flow of pure
    nitrogen given the nitrogen mass fraction in the outlet.

    >>> T = 293.15
    >>> inc1.set_attr(fluid={'O2': 0.23, 'N2': 0.77}, p=1, T=T, m=5)
    >>> inc2.set_attr(fluid={'O2': 1, 'N2':0}, T=T, m=5)
    >>> inc3.set_attr(fluid={'O2': 0, 'N2': 1}, T=T)
    >>> outg.set_attr(fluid={'N2': 0.4})
    >>> nw.solve('design')
    >>> round(inc3.m.val_SI, 2)
    0.25
    >>> abs(round((outg.T.val_SI - T) / T, 5)) < 0.01
    True
    >>> T = 173.15
    >>> inc1.set_attr(T=T)
    >>> inc2.set_attr(T=T)
    >>> inc3.set_attr(T=T)
    >>> nw.solve('design')
    >>> abs(round((outg.T.val_SI - T) / T, 5)) < 0.01
    True
    """

    @staticmethod
    def component():
        return 'T-junction merge'

    def get_variables(self):
        return {
            "pr1": dc_cp(
                min_val=1e-4,
                max_val=1,
                num_eq=1,
                deriv=self.pr_deriv,
                func=self.pr_func,
                func_params={"pr": "pr"},
                latex=self.pr_func_doc,
            ),
            "pr2": dc_cp(
                min_val=1e-4,
                max_val=1,
                num_eq=1,
                deriv=self.pr_deriv,
                func=self.pr_func,
                func_params={"pr": "pr"},
                latex=self.pr_func_doc,
            ),
            "zeta1": dc_cp(
                min_val=0,
                max_val=1e15,
                num_eq=1,
                latex=self.zeta_func_doc,
                deriv=self.zeta_deriv,
                func=self.zeta_func,
                func_params={"zeta": "zeta1", "inconn": 0, "outconn": 0},
            ),
            "zeta2": dc_cp(
                min_val=0,
                max_val=1e15,
                num_eq=1,
                latex=self.zeta_func_doc,
                deriv=self.zeta_deriv,
                func=self.zeta_func,
                func_params={"zeta": "zeta2", "inconn": 1, "outconn": 0},
            ),
            "num_in": dc_simple(),
        }

    def get_mandatory_constraints(self):
        return {
            'mass_flow_constraints': {
                'func': self.mass_flow_func, 'deriv': self.mass_flow_deriv,
                'constant_deriv': True, 'latex': self.mass_flow_func_doc,
                'num_eq': 1},
            'fluid_constraints': {
                'func': self.fluid_func, 'deriv': self.fluid_deriv,
                'constant_deriv': False, 'latex': self.fluid_func_doc,
                'num_eq': self.num_nw_fluids},
            'energy_balance_constraints': {
                'func': self.energy_balance_func,
                'deriv': self.energy_balance_deriv,
                'constant_deriv': False, 'latex': self.energy_balance_func_doc,
                'num_eq': 1},
            # 'pressure_constraints': {
            #     'func': self.pressure_equality_func,
            #     'deriv': self.pressure_equality_deriv,
            #     'constant_deriv': True,
            #     'latex': self.pressure_equality_func_doc,
            #     'num_eq': self.num_i + self.num_o - 1}
        }

    def inlets(self):
        if self.num_in.is_set:
            return ['in' + str(i + 1) for i in range(self.num_in.val)]
        else:
            self.set_attr(num_in=2)
            return self.inlets()

    @staticmethod
    def outlets():
        return ['out1']

    def zeta_func(self, zeta="", inconn=0, outconn=0):
        r"""
        Calculate residual value of :math:`\zeta`-function.

        Parameters
        ----------
        zeta : str
            Component parameter to evaluate the zeta_func on, e.g.
            :code:`zeta1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.

        Returns
        -------
        residual : float
            Residual value of function.

            .. math::

                0 = \begin{cases}
                p_{in} - p_{out} & |\dot{m}| < \epsilon \\
                \frac{\zeta}{D^4} - \frac{(p_{in} - p_{out}) \cdot \pi^2}
                {8 \cdot \dot{m}_{in} \cdot |\dot{m}_{in}| \cdot \frac{v_{in} +
                v_{out}}{2}} &
                |\dot{m}| > \epsilon
                \end{cases}

        Note
        ----
        The zeta value is caluclated on the basis of a given pressure loss at
        a given flow rate in the design case. As the cross sectional area A
        will not change, it is possible to handle the equation in this way:

        .. math::

            \frac{\zeta}{D^4} = \frac{\Delta p \cdot \pi^2}
            {8 \cdot \dot{m}^2 \cdot v}
        """
        data = self.get_attr(zeta)
        i = self.inl[inconn].get_flow()
        o = self.outl[outconn].get_flow()

        v_i = v_mix_ph(i, T0=self.inl[inconn].T.val_SI)
        v_o = v_mix_ph(o, T0=self.outl[outconn].T.val_SI)

        return ((i[1] - o[1]) - data.val * abs(i[0]) * i[0] * ((v_i + v_o) / 2) /
                2)

    def zeta_func_doc(self, label, zeta="", inconn=0, outconn=0):
        r"""
        Calculate residual value of :math:`\zeta`-function.

        Parameters
        ----------
        zeta : str
            Component parameter to evaluate the zeta_func on, e.g.
            :code:`zeta1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.

        Returns
        -------
        residual : float
            Residual value of function.
        """
        inl = r"_\mathrm{in," + str(inconn + 1) + r"}"
        outl = r"_\mathrm{out," + str(outconn + 1) + r"}"
        latex = (
            r"0 = \begin{cases}"
            + "\n"
            + r"p"
            + inl
            + r"- p"
            + outl
            + r" & |\dot{m}"
            + inl
            + r"| < \unitfrac[0.0001]{kg}{s} \\"
            + "\n"
            + r"\frac{\zeta}{D^4}-\frac{(p"
            + inl
            + r"-p"
            + outl
            + r")"
            r"\cdot\pi^2}{8\cdot\dot{m}"
            + inl
            + r"\cdot|\dot{m}"
            + inl
            + r"|\cdot\frac{v"
            + inl
            + r" + v"
            + outl
            + r"}{2}}"
            + r"& |\dot{m}"
            + inl
            + r"| \geq \unitfrac[0.0001]{kg}{s}"
            + "\n"
            r"\end{cases}"
        )
        return generate_latex_eq(self, latex, label)

    def zeta_deriv(self, increment_filter, k, zeta="", inconn=0, outconn=0):
        r"""
        Calculate partial derivatives of zeta function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.

        zeta : str
            Component parameter to evaluate the zeta_func on, e.g.
            :code:`zeta1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.
        """
        data = self.get_attr(zeta)
        f = self.zeta_func
        outpos = self.num_i + outconn
        if not increment_filter[inconn, 0]:
            self.jacobian[k, inconn, 0] = self.numeric_deriv(
                f, "m", inconn, zeta=zeta, inconn=inconn, outconn=outconn
            )
        if not increment_filter[inconn, 1]:
            self.jacobian[k, inconn, 1] = self.numeric_deriv(
                f, "p", inconn, zeta=zeta, inconn=inconn, outconn=outconn
            )
        if not increment_filter[inconn, 2]:
            self.jacobian[k, inconn, 2] = self.numeric_deriv(
                f, "h", inconn, zeta=zeta, inconn=inconn, outconn=outconn
            )
        if not increment_filter[outpos, 0]:
            self.jacobian[k, outpos, 0] = self.numeric_deriv(
                f, "m", inconn, zeta=zeta, inconn=inconn, outconn=outconn
            )

        if not increment_filter[outpos, 0]:
            self.jacobian[k, outpos, 0] = self.numeric_deriv(
                f, "m", outpos, zeta=zeta, inconn=inconn, outconn=outconn
            )
        if not increment_filter[outpos, 1]:
            self.jacobian[k, outpos, 1] = self.numeric_deriv(
                f, "p", outpos, zeta=zeta, inconn=inconn, outconn=outconn
            )
        if not increment_filter[outpos, 2]:
            self.jacobian[k, outpos, 2] = self.numeric_deriv(
                f, "h", outpos, zeta=zeta, inconn=inconn, outconn=outconn
            )
        # custom variable zeta
        if data.is_var:
            pos = self.num_i + self.num_o + data.var_pos
            self.jacobian[k, pos, 0] = self.numeric_deriv(
                f, zeta, 2, zeta=zeta, inconn=inconn, outconn=outconn
            )

    def fluid_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.

        Returns
        -------
        residual : list
            Vector of residual values for component's fluid balance.

            .. math::

                0 = \sum_i \dot{m}_{in,i} \cdot x_{fl,in,i} -
                \dot {m}_{out} \cdot x_{fl,out}\\
                \forall fl \in \text{network fluids},
                \; \forall i \in \text{inlets}
        """
        residual = []
        for fluid, x in self.outl[0].fluid.val.items():
            res = -x * self.outl[0].m.val_SI
            for i in self.inl:
                res += i.fluid.val[fluid] * i.m.val_SI
            residual += [res]
        return residual

    def fluid_func_doc(self, label):
        r"""
        Calculate the vector of residual values for fluid balance equations.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = (
            r'0=\sum_i \dot{m}_{\mathrm{in,}i} \cdot x_{fl\mathrm{,in,}i}'
            r'- \dot {m}_\mathrm{out} \cdot x_{fl\mathrm{,out}}'
            r'\; \forall fl \in \text{network fluids,} \; \forall i \in'
            r'\text{inlets}'
        )
        return generate_latex_eq(self, latex, label)

    def fluid_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = 0
        for fluid, x in self.outl[0].fluid.val.items():
            j = 0
            for inl in self.inl:
                self.jacobian[k, j, 0] = inl.fluid.val[fluid]
                self.jacobian[k, j, i + 3] = inl.m.val_SI
                j += 1
            self.jacobian[k, j, 0] = -x
            self.jacobian[k, j, i + 3] = -self.outl[0].m.val_SI
            i += 1
            k += 1

    def energy_balance_func(self):
        r"""
        Calculate energy balance.

        Returns
        -------
        residual : float
            Residual value of energy balance.

            .. math::

                0 = \sum_i \left(\dot{m}_{in,i} \cdot h_{in,i} \right) -
                \dot{m}_{out} \cdot h_{out}\\
                \forall i \in \text{inlets}
        """
        res = -self.outl[0].m.val_SI * self.outl[0].h.val_SI
        for i in self.inl:
            res += i.m.val_SI * i.h.val_SI
        return res

    def energy_balance_func_doc(self, label):
        r"""
        Calculate energy balance.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = (
            r'0=\sum_i\left(\dot{m}_{\mathrm{in,}i}\cdot h_{\mathrm{in,}i}'
            r'\right) - \dot{m}_\mathrm{out} \cdot h_\mathrm{out} '
            r'\; \forall i \in \text{inlets}'
        )
        return generate_latex_eq(self, latex, label)

    def energy_balance_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        self.jacobian[k, self.num_i, 0] = -self.outl[0].h.val_SI
        self.jacobian[k, self.num_i, 2] = -self.outl[0].m.val_SI
        j = 0
        for i in self.inl:
            self.jacobian[k, j, 0] = i.h.val_SI
            self.jacobian[k, j, 2] = i.m.val_SI
            j += 1

    def initialise_fluids(self):
        """Fluid initialisation for fluid mixture at outlet of the node."""
        num_fl = {}
        for o in self.outl:
            num_fl[o] = num_fluids(o.fluid.val)

        for i in self.inl:
            num_fl[i] = num_fluids(i.fluid.val)

        ls = []
        if any(num_fl.values()) and not all(num_fl.values()):
            for conn, num in num_fl.items():
                if num >= 1:
                    ls += [conn]

            for c in ls:
                for fluid in self.nw_fluids:
                    for o in self.outl:
                        if not o.fluid.val_set[fluid]:
                            o.fluid.val[fluid] = c.fluid.val[fluid]
                    for i in self.inl:
                        if not i.fluid.val_set[fluid]:
                            i.fluid.val[fluid] = c.fluid.val[fluid]
            for o in self.outl:
                o.target.propagate_fluid_to_target(o, o.target)

    def propagate_fluid_to_target(self, inconn, start):
        r"""
        Fluid propagation stops here.

        Parameters
        ----------
        inconn : tespy.connections.connection.Connection
            Connection to initialise.

        start : tespy.components.component.Component
            This component is the fluid propagation starting point.
            The starting component is saved to prevent infinite looping.
        """
        return

    def propagate_fluid_to_source(self, outconn, start):
        r"""
        Propagate the fluids towards connection's source in recursion.

        Parameters
        ----------
        outconn : tespy.connections.connection.Connection
            Connection to initialise.

        start : tespy.components.component.Component
            This component is the fluid propagation starting point.
            The starting component is saved to prevent infinite looping.
        """
        for inconn in self.inl:
            for fluid, x in outconn.fluid.val.items():
                if (not inconn.fluid.val_set[fluid] and
                        not inconn.good_starting_values):
                    inconn.fluid.val[fluid] = x

            inconn.source.propagate_fluid_to_source(inconn, start)

    def entropy_balance(self):
        r"""
        Calculate entropy balance of a merge.

        Note
        ----
        A definition of reference points is included for compensation of
        differences in zero point definitions of different fluid compositions.

        - Reference temperature: 298.15 K.
        - Reference pressure: 1 bar.

        .. math::

            \dot{S}_\mathrm{irr}= \dot{m}_\mathrm{out} \cdot
            \left( s_\mathrm{out} - s_\mathrm{out,ref} \right)
            - \sum_{i} \dot{m}_{\mathrm{in,}i} \cdot
            \left( s_{\mathrm{in,}i} - s_{\mathrm{in,ref,}i} \right)\\
        """
        T_ref = 298.15
        p_ref = 1e5
        self.S_irr = self.outl[0].m.val_SI * (
            self.outl[0].s.val_SI -
            s_mix_pT([0, p_ref, 0, self.outl[0].fluid.val], T_ref))
        for i in self.inl:
            self.S_irr -= i.m.val_SI * (
                i.s.val_SI -
                s_mix_pT([0, p_ref, 0, i.fluid.val], T_ref))

    def exergy_balance(self, T0):
        r"""
        Calculate exergy balance of a merge.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        Please note, that the exergy balance accounts for physical exergy only.

        .. math ::

            \dot{E}_\mathrm{P} =
            \begin{cases}
            \begin{cases}
            \sum_i \dot{m}_i \cdot \left(e_\mathrm{out}^\mathrm{PH} -
            e_{\mathrm{in,}i}^\mathrm{PH}\right)
            & T_{\mathrm{in,}i} < T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} \geq T_0 \\
            \sum_i \dot{m}_i \cdot e_\mathrm{out}^\mathrm{PH}
            & T_{\mathrm{in,}i} < T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} < T_0 \\
            \end{cases} & T_\mathrm{out} > T_0\\

            \text{not defined (nan)} & T_\mathrm{out} = T_0\\

            \begin{cases}
            \sum_i \dot{m}_i \cdot e_\mathrm{out}^\mathrm{PH}
            & T_{\mathrm{in,}i} > T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} \geq T_0 \\
            \sum_i \dot{m}_i \cdot \left(e_\mathrm{out}^\mathrm{PH} -
            e_{\mathrm{in,}i}^\mathrm{PH}\right)
            & T_{\mathrm{in,}i} > T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} < T_0 \\
            \end{cases} & T_\mathrm{out} < T_0\\
            \end{cases}

            \dot{E}_\mathrm{F} =
            \begin{cases}
            \begin{cases}
            \sum_i \dot{m}_i \cdot \left(e_{\mathrm{in,}i}^\mathrm{PH} -
            e_\mathrm{out}^\mathrm{PH}\right)
            & T_{\mathrm{in,}i} > T_\mathrm{out} \\
            \sum_i \dot{E}_{\mathrm{in,}i}^\mathrm{PH}
            & T_{\mathrm{in,}i} < T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} < T_0 \\
            \end{cases} & T_\mathrm{out} > T_0\\

            \sum_i \dot{E}_{\mathrm{in,}i}^\mathrm{PH} & T_\mathrm{out} = T_0\\

            \begin{cases}
            \sum_i \dot{E}_{\mathrm{in,}i}^\mathrm{PH}
            & T_{\mathrm{in,}i} > T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} \geq T_0 \\
            \sum_i \dot{m}_i \cdot \left(e_{\mathrm{in,}i}^\mathrm{PH} -
            e_\mathrm{out}^\mathrm{PH}\right)
            & T_{\mathrm{in,}i} < T_\mathrm{out} \\
            \end{cases} & T_\mathrm{out} < T_0\\
            \end{cases}

            \forall i \in \text{merge inlets}

            \dot{E}_\mathrm{bus} = \text{not defined (nan)}
        """
        self.E_P = 0
        self.E_F = 0
        if self.outl[0].T.val_SI > T0:
            for i in self.inl:
                if i.T.val_SI < self.outl[0].T.val_SI:
                    if i.T.val_SI >= T0:
                        self.E_P += i.m.val_SI * (
                            self.outl[0].ex_physical - i.ex_physical)
                    else:
                        self.E_P += i.m.val_SI * self.outl[0].ex_physical
                        self.E_F += i.Ex_physical
                else:
                    self.E_F += i.m.val_SI * (
                        i.ex_physical - self.outl[0].ex_physical)
        elif self.outl[0].T.val_SI == T0:
            self.E_P = np.nan
            for i in self.inl:
                self.E_F += i.Ex_physical
        else:
            for i in self.inl:
                if i.T.val_SI > self.outl[0].T.val_SI:
                    if i.T.val_SI >= T0:
                        self.E_P += i.m.val_SI * self.outl[0].ex_physical
                        self.E_F += i.Ex_physical
                    else:
                        self.E_P += i.m.val_SI * (
                            self.outl[0].ex_physical - i.ex_physical)
                else:
                    self.E_F += i.m.val_SI * (
                        i.ex_physical - self.outl[0].ex_physical)

        self.E_bus = np.nan
        self.E_D = self.E_F - self.E_P
        self.epsilon = self.E_P / self.E_F

    def get_plotting_data(self):
        """Generate a dictionary containing FluProDia plotting information.

        Returns
        -------
        data : dict
            A nested dictionary containing the keywords required by the
            :code:`calc_individual_isoline` method of the
            :code:`FluidPropertyDiagram` class. First level keys are the
            connection index ('in1' -> 'out1', therefore :code:`1` etc.).
        """
        return {
            i + 1: {
                'isoline_property': 'p',
                'isoline_value': self.inl[i].p.val,
                'isoline_value_end': self.outl[0].p.val,
                'starting_point_property': 'v',
                'starting_point_value': self.inl[i].vol.val,
                'ending_point_property': 'v',
                'ending_point_value': self.outl[0].vol.val
            } for i in range(self.num_i)}
