# -*- coding: utf-8

"""Module of class Pipe.


This file is part of project TESPy (github.com/oemof/tespy). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location tespy/components/piping/pipe.py

SPDX-License-Identifier: MIT
"""



import logging

from scipy.misc import derivative
import numpy as np

from tespy.components.heat_exchangers.simple import HeatExchangerSimple
from tespy.tools.data_containers import ComponentCharacteristics as dc_cc
from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.data_containers import DataContainerSimple as dc_simple
from tespy.tools.data_containers import GroupedComponentProperties as dc_gcp
from tespy.tools.document_models import generate_latex_eq
from tespy.tools.fluid_properties import T_mix_ph
from tespy.tools.fluid_properties import s_mix_ph
from tespy.tools.fluid_properties import v_mix_ph
from tespy.tools.fluid_properties import visc_mix_ph
from tespy.tools.helpers import convert_to_SI
from tespy.tools.helpers import darcy_friction_factor as dff




class RectangularPipe(HeatExchangerSimple):
    r"""
    The RectangularPipe is a subclass of a HeatExchangerSimple.

    **Mandatory Equations**

    - :py:meth:`tespy.components.component.Component.fluid_func`
    - :py:meth:`tespy.components.component.Component.mass_flow_func`

    **Optional Equations**

    - :py:meth:`tespy.components.component.Component.pr_func`
    - :py:meth:`tespy.components.component.Component.zeta_func`
    - :py:meth:`tespy.components.heat_exchangers.simple.HeatExchangerSimple.energy_balance_func`
    - :py:meth:`tespy.components.heat_exchangers.simple.HeatExchangerSimple.hydro_group_func`
    - :py:meth:`tespy.components.heat_exchangers.simple.HeatExchangerSimple.kA_group_func`
    - :py:meth:`tespy.components.heat_exchangers.simple.HeatExchangerSimple.kA_char_group_func`

    Inlets/Outlets

    - in1
    - out1

    Image

    .. image:: _images/Pipe.svg
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

    Q : float, dict, :code:`"var"`
        Heat transfer, :math:`Q/\text{W}`.

    pr : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio, :math:`pr/1`.

    zeta : float, dict, :code:`"var"`
        Geometry independent friction coefficient,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    H : float, dict, :code:`"var"`
        Height of the pipes, :math:`H/\text{m}`.

    W : float, dict, :code:`"var"`
        Width of the pipes, :math:`H/\text{m}`.

    L : float, dict, :code:`"var"`
        Length of the pipes, :math:`L/\text{m}`.

    ks : float, dict, :code:`"var"`
        Pipe's roughness, :math:`ks/\text{m}` for darcy friction,
        :math:`ks/\text{1}` for hazen-williams equation.

    hydro_group : str, dict
        Parametergroup for pressure drop calculation based on pipes dimensions.
        Choose 'HW' for hazen-williams equation, else darcy friction factor is
        used.

    kA : float, dict, :code:`"var"`
        Area independent heat transfer coefficient,
        :math:`kA/\frac{\text{W}}{\text{K}}`.

    kA_char : tespy.tools.characteristics.CharLine, dict
        Characteristic line for heat transfer coefficient.

    Tamb : float, dict
        Ambient temperature, provide parameter in network's temperature
        unit.

    kA_group : str, dict
        Parametergroup for heat transfer calculation from ambient temperature
        and area independent heat transfer coefficient kA.

    Example
    -------
    A mass flow of 10 kg/s ethanol is transported in a pipeline. The pipe is
    considered adiabatic and has a length of 500 meters. We can calculate the
    diameter required at a given pressure loss of 2.5 %. After we determined
    the required diameter, we can predict pressure loss at a different mass
    flow through the pipeline.

    >>> from tespy.components import Sink, Source, Pipe
    >>> from tespy.connections import Connection
    >>> from tespy.networks import Network
    >>> import shutil
    >>> fluid_list = ['ethanol']
    >>> nw = Network(fluids=fluid_list)
    >>> nw.set_attr(p_unit='bar', T_unit='C', h_unit='kJ / kg', iterinfo=False)
    >>> so = Source('source 1')
    >>> si = Sink('sink 1')
    >>> pi = Pipe('pipeline')
    >>> pi.component()
    'pipe'
    >>> pi.set_attr(pr=0.975, Q=0, design=['pr'], L=100, D='var', ks=5e-5)
    >>> inc = Connection(so, 'out1', pi, 'in1')
    >>> outg = Connection(pi, 'out1', si, 'in1')
    >>> nw.add_conns(inc, outg)
    >>> inc.set_attr(fluid={'ethanol': 1}, m=10, T=30, p=3)
    >>> nw.solve('design')
    >>> nw.save('tmp')
    >>> round(pi.D.val, 3)
    0.119
    >>> outg.p.val / inc.p.val == pi.pr.val
    True
    >>> inc.set_attr(m=15)
    >>> pi.set_attr(D=pi.D.val)
    >>> nw.solve('offdesign', design_path='tmp')
    >>> round(pi.pr.val, 2)
    0.94
    >>> shutil.rmtree('./tmp', ignore_errors=True)
    """

    @staticmethod
    def component():
        return 'rectangular pipe'

    def get_variables(self):
        return {
            'Q': dc_cp(
                deriv=self.energy_balance_deriv,
                latex=self.energy_balance_func_doc, num_eq=1,
                func=self.energy_balance_func),
            'pr': dc_cp(
                min_val=1e-4, max_val=1, num_eq=1,
                deriv=self.pr_deriv, latex=self.pr_func_doc,
                func=self.pr_func, func_params={'pr': 'pr'}),
            'zeta': dc_cp(
                min_val=0, max_val=1e15, num_eq=1,
                deriv=self.zeta_deriv, func=self.zeta_func,
                latex=self.zeta_func_doc,
                func_params={'zeta': 'zeta'}),
            'H': dc_cp(min_val=1e-3, max_val=2, d=1e-4),
            'W': dc_cp(min_val=1e-3, max_val=2, d=1e-4),
            'L': dc_cp(min_val=1e-2, d=1e-3),
            'ks': dc_cp(val=1e-4, min_val=1e-7, max_val=1e-3, d=1e-8),
            'kA': dc_cp(min_val=0, d=1),
            'kA_char': dc_cc(param='m'), 'Tamb': dc_cp(),
            'dissipative': dc_simple(val=True),
            'hydro_group': dc_gcp(
                elements=['L', 'ks', 'H', 'W'], num_eq=1,
                latex=self.hydro_group_func_doc,
                func=self.hydro_group_func, deriv=self.hydro_group_deriv),
            'kA_group': dc_gcp(
                elements=['kA', 'Tamb'], num_eq=1,
                latex=self.kA_group_func_doc,
                func=self.kA_group_func, deriv=self.kA_group_deriv),
            'kA_char_group': dc_gcp(
                elements=['kA_char', 'Tamb'], num_eq=1,
                latex=self.kA_char_group_func_doc,
                func=self.kA_char_group_func, deriv=self.kA_char_group_deriv)
        }

    def darcy_func(self):
        r"""
        Equation for pressure drop calculation from darcy friction factor.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = p_{in} - p_{out} - \frac{8 \cdot |\dot{m}_{in}| \cdot
                \dot{m}_{in} \cdot \frac{v_{in}+v_{out}}{2} \cdot L \cdot
                \lambda\left(Re, ks, D\right)}{\pi^2 \cdot D^5}\\

                Re = \frac{4 \cdot |\dot{m}_{in}|}{\pi \cdot D \cdot
                \frac{\eta_{in}+\eta_{out}}{2}}\\
                \eta: \text{dynamic viscosity}\\
                v: \text{specific volume}\\
                \lambda: \text{darcy friction factor}
        """
        i, o = self.inl[0].get_flow(), self.outl[0].get_flow()

        if abs(i[0]) < 1e-4:
            return i[1] - o[1]

        visc_i = visc_mix_ph(i, T0=self.inl[0].T.val_SI)
        visc_o = visc_mix_ph(o, T0=self.outl[0].T.val_SI)
        v_i = v_mix_ph(i, T0=self.inl[0].T.val_SI)
        v_o = v_mix_ph(o, T0=self.outl[0].T.val_SI)

        cross_sectional_area = self.H.val * self.W.val
        perimeter = 2 * self.H.val + 2 * self.W.val
        hydraulic_diameter = 4 * cross_sectional_area / perimeter

        Re = abs(i[0]) * hydraulic_diameter / (cross_sectional_area * (visc_i + visc_o) / 2)

        return ((i[1] - o[1]) - abs(i[0]) * i[0] * (v_i + v_o) / 2 *
                self.L.val * dff(Re, self.ks.val, hydraulic_diameter) /
                (2 * hydraulic_diameter * cross_sectional_area**2))

    def darcy_func_doc(self, label):
        r"""
        Equation for pressure drop calculation from darcy friction factor.

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
            r'\begin{split}' + '\n'
            r'0 = &p_\mathrm{in}-p_\mathrm{out}-'
            r'\frac{8\cdot|\dot{m}_\mathrm{in}| \cdot\dot{m}_\mathrm{in}'
            r'\cdot \frac{v_\mathrm{in}+v_\mathrm{out}}{2} \cdot L \cdot'
            r'\lambda\left(Re, ks, D\right)}{\pi^2 \cdot D^5}\\' + '\n'
            r'Re =&\frac{4 \cdot |\dot{m}_\mathrm{in}|}{\pi \cdot D \cdot'
            r'\frac{\eta_\mathrm{in}+\eta_\mathrm{out}}{2}}\\' + '\n'
            r'\end{split}'
        )
        return generate_latex_eq(self, latex, label)

    def hazen_williams_func(self):
        r"""
        Equation for pressure drop calculation from Hazen-Williams equation.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = \left(p_{in} - p_{out} \right) \cdot \left(-1\right)^i -
                \frac{10.67 \cdot |\dot{m}_{in}| ^ {1.852}
                \cdot L}{ks^{1.852} \cdot D^{4.871}} \cdot g \cdot
                \left(\frac{v_{in} + v_{out}}{2}\right)^{0.852}

                i = \begin{cases}
                0 & \dot{m}_{in} \geq 0\\
                1 & \dot{m}_{in} < 0
                \end{cases}

        Note
        ----
        Gravity :math:`g` is set to :math:`9.81 \frac{m}{s^2}`
        """
        i, o = self.inl[0].get_flow(), self.outl[0].get_flow()

        if abs(i[0]) < 1e-4:
            return i[1] - o[1]

        v_i = v_mix_ph(i, T0=self.inl[0].T.val_SI)
        v_o = v_mix_ph(o, T0=self.outl[0].T.val_SI)

        cross_sectional_area = self.H * self.W
        perimeter = 2 * self.H + 2 * self.W
        hydraulic_diameter = 4 * cross_sectional_area / perimeter

        return ((i[1] - o[1]) * np.sign(i[0]) -
                (10.67 * abs(i[0]) ** 1.852 * self.L.val /
                 (self.ks.val ** 1.852 * hydraulic_diameter ** 4.871)) *
                (9.81 * ((v_i + v_o) / 2) ** 0.852))

    def hazen_williams_func_doc(self, label):
        r"""
        Equation for pressure drop calculation from Hazen-Williams equation.

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
            r'0 = \left(p_\mathrm{in} - p_\mathrm{out} \right) -'
            r'\frac{10.67 \cdot |\dot{m}_\mathrm{in}| ^ {1.852}'
            r'\cdot L}{ks^{1.852} \cdot D^{4.871}} \cdot g \cdot'
            r'\left(\frac{v_\mathrm{in}+ v_\mathrm{out}}{2}\right)^{0.852}'
        )
        return generate_latex_eq(self, latex, label)

    def hydro_group_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of hydro group (pressure drop).

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """

        i, o = self.inl[0].get_flow(), self.outl[0].get_flow()

        visc_i = visc_mix_ph(i, T0=self.inl[0].T.val_SI)
        visc_o = visc_mix_ph(o, T0=self.outl[0].T.val_SI)
        v_i = v_mix_ph(i, T0=self.inl[0].T.val_SI)
        v_o = v_mix_ph(o, T0=self.outl[0].T.val_SI)

        cross_sectional_area = self.H.val * self.W.val
        perimeter = 2 * self.H.val + 2 * self.W.val
        hydraulic_diameter = 4 * cross_sectional_area / perimeter

        Re = abs(i[0]) * hydraulic_diameter / (cross_sectional_area * (visc_i + visc_o) / 2)

        # hazen williams equation
        if self.hydro_group.method == 'HW':
            func = self.hazen_williams_func
        # darcy friction factor
        else:
            func = self.darcy_func
        if not increment_filter[0, 0]:
            self.jacobian[k, 0, 0] = self.numeric_deriv(func, 'm', 0)
            # self.jacobian[k, 0, 0] = (- abs(i[0]) * (v_i + v_o) / 2 *
            #     self.L.val * dff(Re, self.ks.val, hydraulic_diameter) /
            #     (hydraulic_diameter * cross_sectional_area**2)) +\
            #       (- abs(i[0]) * i[0] * (v_i + v_o) / 2 *
            #       self.L.val * derivative(self.ff, i[0], dx=abs(i[0]) * np.finfo(float).eps ** 0.5) /
            #       (2 * hydraulic_diameter * cross_sectional_area ** 2))
            #print(Re)
            #print(derivative(self.ff, i[0], dx=abs(i[0]) * np.finfo(float).eps ** 0.5))
            # print((- abs(i[0]) * (v_i + v_o) / 2 *
            #     self.L.val * dff(Re, self.ks.val, hydraulic_diameter) /
            #     (hydraulic_diameter * cross_sectional_area**2)) +\
            #       (- abs(i[0]) * i[0] * (v_i + v_o) / 2 *
            #       self.L.val * derivative(self.ff, i[0], dx=abs(i[0]) * np.finfo(float).eps ** 0.5) /
            #       (2 * hydraulic_diameter * cross_sectional_area ** 2))
            #       )
            # print(self.numeric_deriv(func, 'm', 0))
        if not increment_filter[0, 1]:
            self.jacobian[k, 0, 1] = 1
        if not increment_filter[0, 2]:
            self.jacobian[k, 0, 2] = 0
        if not increment_filter[1, 1]:
            self.jacobian[k, 1, 1] = -1
        if not increment_filter[1, 2]:
            self.jacobian[k, 1, 2] = 0
        # custom variables of hydro group
        for var in self.hydro_group.elements:
            var = self.get_attr(var)
            if var.is_var:
                self.jacobian[k, 2 + var.var_pos, 0] = (
                    self.numeric_deriv(func, self.vars[var], 2))

    def ff(self, mass_flow):
        i, o = self.inl[0].get_flow(), self.outl[0].get_flow()

        visc_i = visc_mix_ph(i, T0=self.inl[0].T.val_SI)
        visc_o = visc_mix_ph(o, T0=self.outl[0].T.val_SI)

        cross_sectional_area = self.H.val * self.W.val
        perimeter = 2 * self.H.val + 2 * self.W.val
        hydraulic_diameter = 4 * cross_sectional_area / perimeter
        Re = abs(mass_flow) * hydraulic_diameter / (cross_sectional_area * (visc_i + visc_o) / 2)
        return dff(Re, self.ks.val, hydraulic_diameter)
