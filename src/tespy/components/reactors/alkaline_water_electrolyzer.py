import numpy as np
from tespy.components.component import Component
from tespy.tools import ComponentProperties as dc_cp

from scipy.constants import e, Avogadro, R
from tespy.tools.global_vars import molar_masses


class AlkalineWaterElectrolyzer(Component):
    """
    This is a component for the alkaline electrolysis of water.

    From this part, it should be clear for the user, which parameters are available, which mandatory equations
    are applied and which optional equations can be applied using the
    component parameters.
    """

    @staticmethod
    def component(self):
        return "alkaline water electrolyzer"

    def get_variables(self):
        return {
            "pr": dc_cp(
                min_val=1e-4,
                max_val=1,
                num_eq=2,
                deriv=self.pr_deriv,
                func=self.pr_func,
                func_params={"pr": "pr"},
                latex=self.pr_func_doc,
            )
        }

    def get_mandatory_constraints(self):
        return {
            "mass_flow_constraints": {
                "func": self.mass_flow_func,
                "deriv": self.mass_flow_deriv,
                "constant_deriv": True,
                "latex": self.mass_flow_func_doc,
                "num_eq": 2,
            },
            "fluid_constraints": {
                "func": self.fluid_func,
                "deriv": self.fluid_deriv,
                "constant_deriv": True,
                "latex": self.fluid_func_doc,
                "num_eq": 2*self.num_nw_fluids,
            },
            "enthalpy_equality_constraints": {
                "func": self.enthalpy_equality_func,
                "deriv": self.enthalpy_equality_deriv,
                "constant_deriv": True,
                "latex": self.enthalpy_equality_func_doc,
                "num_eq": 2,
            },
        }

    def inlets(self):
        return ["in1", "in2"]

    def outlets(self):
        return ["out1", "out2"]

    def enthalpy_equality_func(self):
        r"""
        Equation for enthalpy equality.

        Returns
        -------
        residual : list
            Residual values of equations.

            .. math::

                0 = h_{in,i} - h_{out,i} \;\forall i\in\text{inlets}
        """
        #flow = self.outl[0].get_flow()
        #T = fp.T_mix_ph(flow, T0=273.15 + 25)

        fluid_mass = self.alkaline_electrolysis()
        residual = []

        # equation 1
        residual += [self.inl[0].h.val_SI - self.outl[0].h.val_SI]
        # equation 2
        residual += [self.inl[1].h.val_SI - self.outl[1].h.val_SI]
        #residual += [fp.h_mix_pT(flow, T) - self.outl[1].h.val_SI]

        return residual

    def enthalpy_equality_func_doc(self, label):
        r"""
        Equation for enthalpy equality.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        indices = list(range(1, self.num_i + 1))
        if len(indices) > 1:
            indices = ', '.join(str(idx) for idx in indices)
        else:
            indices = str(indices[0])
        latex = (
            r'0=h_{\mathrm{in,}i}-h_{\mathrm{out,}i}'
            r'\; \forall i \in [' + indices + r']')
        return generate_latex_eq(self, latex, label)

    def enthalpy_equality_deriv(self):
        r"""
        Calculate partial derivatives for all mass flow balance equations.

        Returns
        -------
        deriv : ndarray
            Matrix with partial derivatives for the mass flow balance
            equations.
        """
        deriv = np.zeros((
            self.num_o,
            self.num_i + self.num_o + self.num_vars,
            self.num_nw_vars))

        # equation 1
        deriv[0, 0, 2] = 1
        deriv[0, 1, 2] = 0
        deriv[0, 2, 2] = -1
        deriv[0, 3, 2] = 0
        # equation 2
        deriv[1, 1, 2] = 0
        deriv[1, 1, 2] = 1
        deriv[1, 2, 2] = 0
        deriv[1, 3, 2] = -1

        return deriv

    def fluid_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.

        Returns
        -------
        residual : list
            Vector of residual values for component's fluid balance.

            .. math::

                0 = x_{fl,in,i} - x_{fl,out,i} \; \forall fl \in
                \text{network fluids,} \; \forall i \in \text{inlets}
        """

        fluid_mass = self.alkaline_electrolysis()

        residual = []

        # equation 1
        residual += [(fluid_mass["synthesized_H2_mass_cathode"]
                      / self.outl[0].m.val_SI) - self.outl[0].fluid.val['H2']]
        # equation 2
        residual += [self.inl[1].fluid.val['H2'] - self.outl[1].fluid.val['H2']]

        # equation 3
        residual += [((self.inl[0].fluid.val['H2O'] * self.inl[0].m.val_SI - fluid_mass["electrolyzed_H2O_mass_cathode"])
                       / self.outl[0].m.val_SI) - self.outl[0].fluid.val['H2O']]
        # equation 4
        residual += [((self.inl[1].fluid.val['H2O'] * self.inl[1].m.val_SI + fluid_mass["synthesized_H2O_mass_anode"])
                      / self.outl[1].m.val_SI) - self.outl[1].fluid.val['H2O']]

        # equation 5
        residual += [((self.inl[0].fluid.val['KOH'] * self.inl[0].m.val_SI)
                     / self.outl[0].m.val_SI) - self.outl[0].fluid.val['KOH']]
        # equation 6
        residual += [((self.inl[1].fluid.val['KOH'] * self.inl[1].m.val_SI)
                     / self.outl[1].m.val_SI) - self.outl[1].fluid.val['KOH']]

        # equation 7
        residual += [self.inl[0].fluid.val['O2'] - self.outl[0].fluid.val['O2']]
        # equation 8
        residual += [(fluid_mass["synthesized_O2_mass_anode"]
                      / self.outl[1].m.val_SI) - self.outl[1].fluid.val['O2']]

        return residual

    def fluid_func_doc(self, label):
        r"""
        Get fluid balance equations in LaTeX format.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        indices = list(range(1, self.num_i + 1))
        if len(indices) > 1:
            indices = ", ".join(str(idx) for idx in indices)
        else:
            indices = str(indices[0])
        latex = (
            r"0=x_{fl\mathrm{,in,}i}-x_{fl\mathrm{,out,}i}\;"
            r"\forall fl \in\text{network fluids,}"
            r"\; \forall i \in [" + indices + r"]"
        )
        return generate_latex_eq(self, latex, label)

    def fluid_deriv(self):
        r"""
        Calculate partial derivatives for all fluid balance equations.
        Returns
        -------
        deriv : ndarray
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros(
            (
                self.fluid_constraints["num_eq"],
                self.num_i + self.num_o,
                self.num_nw_vars,
            )
        )

        # equation 1
        deriv[0, 0, 3] = 0
        deriv[0, 2, 3] = -1
        # equation 2
        deriv[1, 1, 3] = 1
        deriv[1, 3, 3] = -1

        # equation 3
        deriv[2, 0, 4] = 1
        deriv[2, 2, 4] = -1
        # equation 4
        deriv[3, 1, 4] = 1
        deriv[3, 3, 4] = -1

        # equation 5
        deriv[4, 0, 5] = 1
        deriv[4, 2, 5] = -1
        # equation 6
        deriv[5, 1, 5] = 1
        deriv[5, 3, 5] = -1

        # equation 7
        deriv[6, 0, 6] = 1
        deriv[6, 2, 6] = -1
        # equation 8
        deriv[7, 1, 6] = 0
        deriv[7, 3, 6] = -1

        return deriv

    def mass_flow_func(self):
        r"""
        Calculate the residual value for mass flow balance equation.

        Returns
        -------
        residual : list
            Vector with residual value for component's mass flow balance.

            .. math::

                0 = \dot{m}_{in,i} -\dot{m}_{out,i} \;\forall i\in\text{inlets}
        """

        fluid_mass = self.alkaline_electrolysis()

        residual = []

        # equation 1
        residual += [self.inl[0].m.val_SI - self.outl[0].m.val_SI - fluid_mass['transferred_OH_mass']]

        # equation 2
        # residual += [fluid_mass['transferred_OH_mass'] - self.outl[1].m.val_SI]
        residual += [self.inl[0].m.val_SI + self.inl[1].m.val_SI - self.outl[0].m.val_SI - self.outl[1].m.val_SI]

        return residual

    def mass_flow_func_doc(self, label):
        r"""
        Get mass flow equations in LaTeX format.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        indices = list(range(1, self.num_i + 1))
        if len(indices) > 1:
            indices = ', '.join(str(idx) for idx in indices)
        else:
            indices = str(indices[0])
        latex = (
            r'0=\dot{m}_{\mathrm{in,}i}-\dot{m}_{\mathrm{out,}i}'
            r'\; \forall i \in [' + indices + r']')
        return generate_latex_eq(self, latex, label)

    def mass_flow_deriv(self):
        r"""
        Calculate partial derivatives for all mass flow balance equations.

        Returns
        -------
        deriv : ndarray
            Matrix with partial derivatives for the mass flow balance
            equations.
        """
        deriv = np.zeros((
            self.num_o,
            self.num_i + self.num_o + self.num_vars,
            self.num_nw_vars))

        # mass equation 1
        deriv[0, 0, 0] = 1
        deriv[0, 2, 0] = -1

        # mass equation 2
        deriv[1, 0, 0] = 1
        deriv[1, 1, 0] = 1
        deriv[1, 2, 0] = -1
        deriv[1, 3, 0] = -1

        return deriv

    def pr_func(self, pr='', inconn=0, outconn=0):
        r"""
        Calculate residual value of pressure ratio function.

        Parameters
        ----------
        pr : str
            Component parameter to evaluate the pr_func on, e.g.
            :code:`pr1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.

        Returns
        -------
        residual : float
            Residual value of function.

            .. math::

                0 = p_{in} \cdot pr - p_{out}
        """
        pr = self.get_attr(pr)

        residual = []
        # equation 1
        residual += [self.inl[0].p.val_SI * pr.val - self.outl[0].p.val_SI]

        # equation 2
        residual += [self.inl[1].p.val_SI * pr.val - self.outl[1].p.val_SI]

        return residual

    def pr_func_doc(self, label, pr='', inconn=0, outconn=0):
        r"""
        Calculate residual value of pressure ratio function.

        Parameters
        ----------
        pr : str
            Component parameter to evaluate the pr_func on, e.g.
            :code:`pr1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.

        Returns
        -------
        residual : float
            Residual value of function.
        """
        latex = (
            r'0=p_\mathrm{in,' + str(inconn + 1) + r'}\cdot ' + pr +
            r' - p_\mathrm{out,' + str(outconn + 1) + r'}'
        )
        return generate_latex_eq(self, latex, label)

    def pr_deriv(self, increment_filter, k, pr='', inconn=0, outconn=0):
        r"""
        Calculate residual value of pressure ratio function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.

        pr : str
            Component parameter to evaluate the pr_func on, e.g.
            :code:`pr1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.
        """
        pr = self.get_attr(pr)

        # equation 1
        self.jacobian[k, 0, 1] = pr.val
        self.jacobian[k, 2, 1] = -1

        # equation 2
        self.jacobian[k+1, 1, 1] = pr.val
        self.jacobian[k+1, 3, 1] = -1


        if pr.is_var:
            pos = self.num_i + self.num_o + pr.var_pos
            self.jacobian[k, pos, 0] = self.inl[inconn].p.val_SI

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

        return  # added

        conn_idx = self.outl.index(outconn)
        inconn = self.inl[conn_idx]

        for fluid, x in outconn.fluid.val.items():
            if (inconn.fluid.val_set[fluid] is False and
                    inconn.good_starting_values is False):
                inconn.fluid.val[fluid] = x

        inconn.source.propagate_fluid_to_source(inconn, start)

    def alkaline_electrolysis(self):
        """constants"""
        faraday_constant = e * Avogadro  # [C/mol]
        universal_gas_constant = R  # [J/(mol*K)]
        standard_pressure = 101325  # [Pa]
        standard_temperature = 273.15  # [K]

        """initial system state"""
        number_of_cells = 1
        temperature = 65  # [degC]
        pressure = 16  # [bar]
        mass_concentration_KOH = 0.3
        volume_flow_cathode = 0.3  # 3.0229952 * (number_of_cells / 80)  # [m^3/hr]
        volume_flow_anode = 0.3  # 3.2755530 * (number_of_cells / 80)  # [m^3/hr]

        stack_voltage = number_of_cells * 1.8  # [V]
        stack_current_density = 0  # [A/cm^2]
        active_cell_surface = 2710  # [cm^2]
        faraday_efficiency = 1  # 0.953
        stack_current = (
            stack_current_density
            * active_cell_surface
            * number_of_cells
            * faraday_efficiency
        )  # [A]

        """electrochemistry"""
        # molar mass in [kg/mol]
        molar_mass_H2O = molar_masses["H2O"]
        molar_mass_H2 = molar_masses["H2"]
        molar_mass_O2 = molar_masses["O2"]
        molar_mass_OH = (molar_mass_H2 + molar_mass_O2) / 2

        # current in [A]
        # current_density = 0.5  # 0.6  # [A/cm^2]
        # active_cell_surface = 2700  # [cm^2]
        # faraday_efficiency = 1  # 0.953
        # current = (
        #     current_density * active_cell_surface * number_of_cells * faraday_efficiency
        # )

        # hour in seconds (hence flow rate per hour)
        electrolysis_time = 1

        # valency of an atom is the measure of its combining capacity with other atoms
        valency_H = 1
        valency_O = 2

        # stoichiometric reaction equation
        # cathode:   4*H2O +4e- --> 2*H2 + 4*OH-
        # anode:     4*OH- --> O2 + 2*H2O + 4*e-
        # total:     2*H2O --> 2*H2 + O2

        # faraday's law
        #  n = I * t / (z * F)

        synthesized_H2_mol_cathode = (
            stack_current * electrolysis_time / (2 * valency_H * faraday_constant)
        )

        synthesized_H2_mass_cathode = molar_mass_H2 * synthesized_H2_mol_cathode

        electrolyzed_H2O_mass_cathode = molar_mass_H2O * 2 * synthesized_H2_mol_cathode

        transferred_OH_mass = molar_mass_OH * 2 * synthesized_H2_mol_cathode

        synthesized_O2_mass_anode = molar_mass_O2 * 0.5 * synthesized_H2_mol_cathode

        synthesized_H2O_mass_anode = molar_mass_H2O * synthesized_H2_mol_cathode

        """energy balance"""
        power_input = stack_current * stack_voltage  # [W]
        effective_power_input = (power_input * faraday_efficiency) / 1000  # [kW]
        effective_energy_input = effective_power_input * electrolysis_time  # [kJ]
        enthalpy_of_reaction = 285.900  # [kJ/mol]
        electrolysis_energy = enthalpy_of_reaction * synthesized_H2_mol_cathode  # [kJ]

        energy_losses = (effective_energy_input - electrolysis_energy)*1000 #[J]

        return {
            "synthesized_H2_mol_cathode": synthesized_H2_mol_cathode,
            "synthesized_H2_mass_cathode": synthesized_H2_mass_cathode,
            "electrolyzed_H2O_mass_cathode": electrolyzed_H2O_mass_cathode,
            "transferred_OH_mass": transferred_OH_mass,
            "synthesized_O2_mass_anode": synthesized_O2_mass_anode,
            "synthesized_H2O_mass_anode": synthesized_H2O_mass_anode,
            "energy_losses": energy_losses,

        }
