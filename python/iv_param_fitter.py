import numpy as np
import time
import json
from pathlib import Path
from .constants import PhysicsConstants
from .ceb_numeric_model import CEBNumericModel
from .utils import Utils
from .minimization import MinimizationAlgorithms
from .golubev import I as golubev_current

try:
    from .ceb_bindings import compute_ceb_properties_threaded
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

class IVParamFitter:
    def __init__(self, config_file: str = None, use_cpp_backend: bool = True, display: bool = False, output_dir: str = "."):
        self.model = CEBNumericModel()
        self.constants = PhysicsConstants()
        self.Iexp = None
        self.Vexp = None
        self.Inum = None
        self.Vnum = None
        self.Irex = None
        self.Vrex = None
        self.Te_num = None
        self.Igol = None
        self.par = {}
        self.to_fit = {}
        self.mins = {}
        self.maxs = {}
        self.init_brute = {}
        self._num_iterations = None
        
        self.amp_constants = self.constants.get_amplifier_constants('AD745')
        self.use_cpp_backend = use_cpp_backend and HAS_CPP_BACKEND
        self.display = display
        self.set_output_dir(output_dir)
        
        # Initialize display system
        self.fig = None
        self.ax = None
        self.ax_lin = None
        self.ax_res_num = None
        self.ax_res_gol = None
        self.line_exp = None
        self.line_num = None
        self.line_gol = None
        self.line_exp_lin = None
        self.line_num_lin = None
        self.line_gol_lin = None
        self.line_res_num = None
        self.line_res_gol = None
        self.eval_count = 0
        
        if self.display:
            self._setup_display()
        
        if self.use_cpp_backend:
            print("Using C++ threaded backend for CEB computation")
        elif use_cpp_backend and not HAS_CPP_BACKEND:
            print("Warning: C++ backend requested but not available, using pure Python")
        
        if config_file:
            self.load_config(config_file)
        else:
            self.load_default_parameters()
    
    def set_output_dir(self, output_dir: str) -> None:        
        self.output_dir = Path(output_dir)

    def _setup_display(self) -> None:
        """Setup matplotlib display for fitting visualization."""
        try:
            import matplotlib.pyplot as plt
            
            # Check if we're in a suitable environment for display
            import os
            if os.environ.get('DISPLAY') is None and os.name != 'nt':
                print("No display available, display disabled")
                self.display = False
                self.fig = None
                self.ax = None
                self.ax_lin = None
                return
            
            self.fig, ((self.ax, self.ax_lin), (self.ax_res_num, self.ax_res_gol)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Setup log scale subplot (top-left)
            self.line_exp, = self.ax.plot([], [], 'bo-', label='Experimental', markersize=4, alpha=0.7)
            self.line_num, = self.ax.plot([], [], 'r-', label='Numerical Fit', linewidth=2)
            self.line_gol, = self.ax.plot([], [], 'g--', label='Golubev Fit', linewidth=2)
            self.ax.set_xlabel('Voltage (V)', fontsize=12)
            self.ax.set_ylabel('Current (A)', fontsize=12)
            self.ax.set_yscale('log')
            self.ax.set_title('IV Curve Fitting Progress (Log Scale)', fontsize=12, fontweight='bold')
            self.ax.legend(fontsize=10)
            self.ax.grid(True, alpha=0.3)
            self.ax.tick_params(labelsize=10)
            
            # Setup linear scale subplot (top-right)
            self.line_exp_lin, = self.ax_lin.plot([], [], 'bo-', label='Experimental', markersize=4, alpha=0.7)
            self.line_num_lin, = self.ax_lin.plot([], [], 'r-', label='Numerical Fit', linewidth=2)
            self.line_gol_lin, = self.ax_lin.plot([], [], 'g--', label='Golubev Fit', linewidth=2)
            self.ax_lin.set_xlabel('Voltage (V)', fontsize=12)
            self.ax_lin.set_ylabel('Current (A)', fontsize=12)
            self.ax_lin.set_title('IV Curve Fitting Progress (Linear Scale)', fontsize=12, fontweight='bold')
            self.ax_lin.legend(fontsize=10)
            self.ax_lin.grid(True, alpha=0.3)
            self.ax_lin.tick_params(labelsize=10)
            
            # Setup numerical residual subplot (bottom-left)
            self.line_res_num, = self.ax_res_num.plot([], [], 'r-', label='Numerical Residual', linewidth=2)
            self.ax_res_num.axhline(0.0, color='k', linestyle=':', linewidth=1)
            self.ax_res_num.set_xlabel('Voltage (V)', fontsize=12)
            self.ax_res_num.set_ylabel('(I_num - I_exp) / I_exp', fontsize=12)
            self.ax_res_num.set_title('Numerical Fit Residual', fontsize=12, fontweight='bold')
            self.ax_res_num.legend(fontsize=10)
            self.ax_res_num.grid(True, alpha=0.3)
            self.ax_res_num.tick_params(labelsize=10)
            
            # Setup Golubev residual subplot (bottom-right)
            self.line_res_gol, = self.ax_res_gol.plot([], [], 'g--', label='Golubev Residual', linewidth=2)
            self.ax_res_gol.axhline(0.0, color='k', linestyle=':', linewidth=1)
            self.ax_res_gol.set_xlabel('Voltage (V)', fontsize=12)
            self.ax_res_gol.set_ylabel('(I_gol - I_exp) / I_exp', fontsize=12)
            self.ax_res_gol.set_title('Golubev Fit Residual', fontsize=12, fontweight='bold')
            self.ax_res_gol.legend(fontsize=10)
            self.ax_res_gol.grid(True, alpha=0.3)
            self.ax_res_gol.tick_params(labelsize=10)
            plt.ion()  # Turn on interactive mode
            plt.tight_layout()
            
            # Store reference to pyplot for later use
            self.plt = plt
            
        except ImportError as e:
            print(f"matplotlib not available ({e}), display disabled")
            self.display = False
            self.fig = None
            self.ax = None
            self.ax_lin = None
            self.plt = None
        except Exception as e:
            print(f"Error setting up display ({e}), display disabled")
            self.display = False
            self.fig = None
            self.ax = None
            self.ax_lin = None
            self.plt = None
    
    def _update_display(self, Irex, Vrex) -> None:
        """Update the display with current IV curves."""
        if not self.display or self.fig is None:
            return
        
        if Irex is None or Vrex is None or self.Inum is None or self.Vnum is None:
            return
        
        try:
            # Compute Golubev current for display (best effort; None -> empty)
            self.compute_golubev_current()
            Igol = self.Igol if self.Igol is not None else np.zeros_like(self.Inum)
            
            # Calculate chi-squared for display
            chi_sq = Utils.chi_sq(self.Inum, Irex)
            chi_sq_gol = Utils.chi_sq_golubev(Igol, Irex) if self.Igol is not None else float('nan')
            
            # Update data - log scale (top-left subplot)
            self.line_exp.set_data(Vrex, Irex)
            self.line_num.set_data(self.Vnum, self.Inum)
            self.line_gol.set_data(self.Vnum, Igol)
            
            # Update data - linear scale (top-right subplot)
            self.line_exp_lin.set_data(Vrex, Irex)
            self.line_num_lin.set_data(self.Vnum, self.Inum)
            self.line_gol_lin.set_data(self.Vnum, Igol)
            
            # Update data - residual subplots
            with np.errstate(divide='ignore', invalid='ignore'):
                res_num = (self.Inum - Irex) / Irex
                res_gol = (Igol - Irex) / Irex
            self.line_res_num.set_data(self.Vnum, res_num)
            self.line_res_gol.set_data(self.Vnum, res_gol)
            
            # Update axis limits - log scale
            all_v = np.concatenate([Vrex, self.Vnum])
            all_i = np.concatenate([Irex, self.Inum, Igol])
            
            self.ax.set_xlim(np.min(all_v) * 0.95, np.max(all_v) * 1.05)
            self.ax.set_ylim(np.min(all_i) * 0.95, np.max(all_i) * 1.05)
            
            # Update axis limits - linear scale
            self.ax_lin.set_xlim(np.min(all_v) * 0.95, np.max(all_v) * 1.05)
            self.ax_lin.set_ylim(np.min(all_i) * 0.95, np.max(all_i) * 1.05)
            
            # Update axis limits - residual subplots
            fin_v = np.isfinite(res_num) & np.isfinite(res_gol)
            if np.any(fin_v):
                res_min = np.min(np.concatenate([res_num[fin_v], res_gol[fin_v]]))
                res_max = np.max(np.concatenate([res_num[fin_v], res_gol[fin_v]]))
            else:
                res_min, res_max = -1.0, 1.0
            pad = (res_max - res_min) * 0.05
            self.ax_res_num.set_xlim(np.min(all_v) * 0.95, np.max(all_v) * 1.05)
            self.ax_res_num.set_ylim(res_min - pad, res_max + pad)
            self.ax_res_gol.set_xlim(np.min(all_v) * 0.95, np.max(all_v) * 1.05)
            self.ax_res_gol.set_ylim(res_min - pad, res_max + pad)
            
            # Update plot with chi-squared info - all subplots
            title = f'Evals: {self.eval_count} | χ²(num): {chi_sq:.3e} | χ²(gol): {chi_sq_gol:.3e}'
            self.ax.set_title(f'IV Curve Fitting Progress (Log Scale)\n{title}', fontsize=12, fontweight='bold')
            self.ax_lin.set_title(f'IV Curve Fitting Progress (Linear Scale)\n{title}', fontsize=12, fontweight='bold')
            self.plt.pause(0.001)  # Small pause to allow GUI update
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def _close_display(self) -> None:
        """Close the display window."""
        if not self.display or self.fig is None:
            return
        
        try:
            if self.plt is not None:
                self.plt.ioff()  # Turn off interactive mode
                self.plt.close(self.fig)
                self.plt = None
            self.fig = None
            self.ax = None
            self.ax_lin = None
        except Exception as e:
            print(f"Error closing display: {e}")
    
    def load_config(self, config_file: str) -> None:
        config = Utils.load_json_config(config_file)
        
        if 'data_file' in config:
            self.data_file = config['data_file']
        else:
            self.data_file = "SPC-CEB_300mK_Triton11-2026.txt"
        
        if 'amp_type' in config:
            self.amp_constants = self.constants.get_amplifier_constants(config['amp_type'])
        
        if 'parameters' in config:
            for param_name, param_data in config['parameters'].items():
                self.par[param_name] = param_data['value']
                self.to_fit[param_name] = param_data.get('vary', False)
                self.mins[param_name] = param_data.get('min', 0.0)
                self.maxs[param_name] = param_data.get('max', 1.0)
                self.init_brute[param_name] = param_data.get('init_brute', False)
            
            print("Parameters loaded from config:")
            for param_name, value in self.par.items():
                print(f"{param_name} = {value:.6f}, to fit = {self.to_fit[param_name]}")
    
    def save_config(self, config_file: str) -> None:
        config = {
            'data_file': getattr(self, 'data_file', "SPC-CEB_300mK_Triton11-2026.txt"),
            'amp_type': 'AD745',  # Default, could be stored
            'parameters': {}
        }
        
        for param_name, value in self.par.items():
            config['parameters'][param_name] = {
                'value': float(value),
                'vary': self.to_fit.get(param_name, False)
            }
        
        Utils.save_json_config(config_file, config)
    
    def load_default_parameters(self) -> None:
        default_params = {
            'Pbg': 0.0,
            'beta': 0.111,
            'TephPOW': 5.0,
            'Vol': 0.02,
            'Z': 0.5,
            'Tc': 1.18,
            'Rn': 11500.0,
            'Rleak': 40000000.0,
            'Wt': 0.0001,
            'tm': 1.0,
            'ii': 0.0,
            'Ra': 200.0,
            'M': 2,
            'MP': 1,
            'Tp': 0.19,
            'F': 14.2,
            'dF': 0.1,
            'dVFinVg': 1.1,
            'dVStartVg': 0.0,
            'dV': 2e-6
        }
        
        self.par = default_params.copy()
        self.to_fit = {name: False for name in default_params.keys()}
        
        # Mark default fit parameters
        self.to_fit['beta'] = True
        self.to_fit['Z'] = True
        self.to_fit['Tp'] = True
        
        print("Default parameters loaded:")
        for param_name, value in self.par.items():
            print(f"{param_name} = {value:.6f}, to fit = {self.to_fit[param_name]}")
    
    def load_experiment_data(self, filename: str, remove_offset: bool = False) -> int:
        self.Iexp, self.Vexp = Utils.load_experimental_data(filename, remove_offset)
        return len(self.Iexp)

    
    def _compute_ceb_properties_python(self) -> int:
        start_time = time.time()
        
        # Physical parameters
        M = float(self.par['M'])  # number of bolometers in series
        MP = float(self.par['MP'])  # number of bolometers in parallel
        total_bolometers = M * MP
        
        Pbg = self.par['Pbg']  # incoming power [pW]
        beta = self.par['beta']  # returning power ratio
        TephPOW = self.par['TephPOW']  # exponent for Te-ph
        Vol = self.par['Vol']  # volume of absorber [um³]
        Sigma = self.par['Z']  # heat exchange in normal metal
        Tc = self.par['Tc']  # critical temperature [K]
        Rn = self.par['Rn'] * MP / M  # normal resistance per bolometer [Ohm]
        Rleak = self.par['Rleak'] * MP / M  # leakage resistance per bolometer [Ohm]
        Wt = self.par['Wt']  # transparency of barrier
        tm = self.par['tm']  # depairing energy
        ii = self.par['ii']  # coefficient for Andreev current
        Ra = self.par['Ra']  # normal resistance of 1 absorber [Ohm]
        Tph = self.par['Tp']  # phonon temperature [K]
        F = self.par['F']  # main frequency [GHz]
        dF = self.par['dF']  # bandwidth [GHz]
        dVFinVg = self.par['dVFinVg']  # voltage range end [Vg units]
        dVStartVg = self.par['dVStartVg']  # voltage range start [Vg units]
        dV = self.par['dV']  # voltage step [V]
        
        Te = Tph  # electron temperature to be found [K]
        Tsin = Tph  # electron temperature in superconductor [K]
        
        DeltaT = np.sqrt(1.0 - np.power(Tsin / Tc, 3.2))
        dPbg = Pbg
        Delta = self.constants.BCS_INTEGRAL * Tc  # [K]
        
        # Normalized constants
        Rsin = (Rn - Ra) / self.constants.NUMBER_OF_SINS_IN_CEB
        I0 = 1e9 * (Delta / Rsin * self.constants.K)  # [nA]
        Vg = Delta * self.constants.K  # [eV]
        tauSin = Tsin / Delta
        tauE = Te / Delta
        
        # Calculation parameters
        Vstr = dVStartVg * Vg
        Vfin = dVFinVg * Vg
        
        voltage_steps = int(np.round((Vfin - Vstr) / dV))
        if voltage_steps == 0:
            raise ValueError("No voltage steps to do")
        
        V = np.linspace(Vstr, Vfin, voltage_steps + 1)
        
        I = np.zeros(voltage_steps + 1)
        I_A = np.zeros(voltage_steps + 1)
        
        self.Inum = np.zeros(voltage_steps - 1)
        self.Vnum = np.zeros(voltage_steps - 1)
        self.Te_num = np.zeros(voltage_steps - 1)
        
        # Open output files in the output directory
        file_noise = open(self.output_dir / 'Noise.txt', 'w')
        print(f"VSHAMPOR: opened {file_noise} to write")
        file_Te = open(self.output_dir / 'Te.txt', 'w')
        file_NEP = open(self.output_dir / 'NEP.txt', 'w')
        file_G = open(self.output_dir / 'G.txt', 'w')
        
        # Write headers
        file_noise.write(f"Voltage\tNOISEep\tNOISEs\tNOISEa\tNOISE\tNOISEph\tNOISE^2-NOISEph^2\n")
        file_Te.write(f"Voltage\tCurrent\tIqp\tIand\tV/Rleak\tTe\tTs\tDeltaT\tPeph\tPand\tPleak\tPabs\tPcool\n")
        file_NEP.write(f"Voltage\tCurrent\tNEPeph\tNEPs\tNEPa\tNEP\tNEPph\tSv\tNEP^2-NEPph^2\n")
        file_G.write(f"Voltage\tGe\tGnis\n")
        
        for voltage_step in range(1, voltage_steps):
            dT = 0.005
            
            tauELower = 0.0
            tauEUpper = 3.0 / self.constants.BCS_INTEGRAL
            
            # Find tauE so that Pheat == NUMBER_OF_SINS_IN_CEB * Pcool
            for _ in range(15):
                tauE = (tauELower + tauEUpper) / 2.0
                
                I[voltage_step] = self.model.current_integral(DeltaT, V[voltage_step] / Vg, tauSin, tauE) * I0 + 1e9 * (V[voltage_step] / Rleak)  # [nA]
                I_A[voltage_step] = 0.0
                if ii != 0.0:
                    I_A[voltage_step] = ii * self.model.and_current(DeltaT, V[voltage_step] / Vg, tauE, Wt, tm) * I0  # [nA]
                
                Pe_ph = Sigma * Vol * (np.power(Tph, TephPOW) - np.power(tauE * Delta, TephPOW)) * 1e3  # [pW]
                Pabs = np.power(I[voltage_step], 2) * Ra * 1e-6  # [pW]
                Pleak = self.constants.NUMBER_OF_SINS_IN_CEB * np.power(V[voltage_step], 2) / Rleak * 1e12  # [pW]
                Pand = np.power(I_A[voltage_step] * 1e-3, 2) * Ra + 2.0 * (I_A[voltage_step] * 1e3) * V[voltage_step]  # [pW]
                
                Pcool, Ps = self.model.power_cool_integral(DeltaT, V[voltage_step] / Vg, tauSin, tauE)
                Pcool *= np.power(Vg, 2) / Rsin * 1e12  # [pW]
                Ps *= np.power(Vg, 2) / Rsin * 1e12  # [pW]
                
                Pheat = Pe_ph + Pabs + Pand + dPbg + 2.0 * beta * Ps + Pleak
                
                if Pheat < self.constants.NUMBER_OF_SINS_IN_CEB * Pcool:
                    tauEUpper = tauE
                else:
                    tauELower = tauE
            
            Te = tauE * Delta
            
            self.Te_num[voltage_step - 1] = Te

            self.Inum[voltage_step - 1] = 1e-9 * (I[voltage_step] + I_A[voltage_step]) * MP
            self.Vnum[voltage_step - 1] = (self.constants.NUMBER_OF_SINS_IN_CEB * V[voltage_step] + 1e-9 * (I[voltage_step] + I_A[voltage_step]) * Ra) * M
            
            # Write Te file
            file_Te.write(f"{self.Vnum[voltage_step - 1]:.6e}\t{self.Inum[voltage_step - 1]:.6e}\t")
            file_Te.write(f"{1e-9 * I[voltage_step] * MP:.6e}\t{1e-9 * I_A[voltage_step] * MP:.6e}\t")
            file_Te.write(f"{1e9 * (V[voltage_step] / Rleak) * MP:.6e}\t{Te:.6e}\t{Tsin:.6e}\t")
            file_Te.write(f"{DeltaT:.6e}\t{Pe_ph:.6e}\t{Pand:.6e}\t{Pleak:.6e}\t{Pabs:.6e}\t{Pcool:.6e}\n")
            
            # Calculate NEP and noise parameters
            dPT = self.model.power_cool_integral(DeltaT, V[voltage_step] / Vg, tauSin, tauE + dT / Delta)[0] - \
                   self.model.power_cool_integral(DeltaT, V[voltage_step] / Vg, tauSin, tauE - dT / Delta)[0]
            
            dPdT = 1e12 * (np.power(Vg, 2) / Rsin) * dPT / (2.0 * dT)  # [pW/K]
            
            dIdT = I0 * (self.model.current_integral(DeltaT, V[voltage_step] / Vg, tauSin, tauE + dT / Delta) -
                       self.model.current_integral(DeltaT, V[voltage_step] / Vg, tauSin, tauE - dT / Delta)) / (2.0 * dT)  # [nA/K]
            
            dIdV = I0 * (self.model.current_integral(DeltaT, V[voltage_step + 1] / Vg, tauSin, tauE) +
                        ii * self.model.and_current(DeltaT, V[voltage_step + 1] / Vg, tauE, Wt, tm) -
                        self.model.current_integral(DeltaT, V[voltage_step - 1] / Vg, tauSin, tauE) -
                        ii * self.model.and_current(DeltaT, V[voltage_step - 1] / Vg, tauE, Wt, tm)) / (2.0 * dV)  # [nA/V]
            
            dPdV = np.power(Vg, 2) / Rsin * 1e12 * (self.model.power_cool_integral(DeltaT, V[voltage_step + 1] / Vg, tauSin, tauE)[0] -
                                                    self.model.power_cool_integral(DeltaT, V[voltage_step - 1] / Vg, tauSin, tauE)[0]) / (2.0 * dV)  # [pW/V]
            
            G_NIS = dPdT
            G_e = 5.0 * Sigma * Vol * np.power(Te, 4) * 1e3  # [pW/K]
            G = G_e + self.constants.NUMBER_OF_SINS_IN_CEB * (G_NIS - dIdT / dIdV * dPdV)  # [pW/K]
            
            Sv = -2.0 * dIdT / dIdV / G / MP  # [V/pW], for 1 bolo
            
            NEPe_ph2 = 10.0 * self.constants.E * self.constants.K * Sigma * Vol * (np.power(Tph, TephPOW) + np.power(Te, TephPOW)) * 1e3 * 1e12  # [pW²/Hz]
            
            NoiA = np.power(self.amp_constants['voltage_noise'] * np.sqrt(2), 2) + np.power(
                self.amp_constants['current_noise'] / np.sqrt(2) * (2.0 * 1e9 / dIdV + Ra) * M / MP, 2)  # [V²/Hz]
            
            NEPa = NoiA / np.power(Sv, 2)  # [pW²/Hz]
            
            # NEP SIN approximation
            dI = 1e9 * (2.0 * self.constants.E * np.abs(I[voltage_step]) / np.power(dIdV * Sv, 2))  # [pW²/Hz]
            dPdI = 1e9 * (2.0 * 2.0 * self.constants.E * Pcool / (dIdV * Sv))  # [pW²/Hz]
            mm = np.log(np.sqrt(2.0 * np.pi * self.constants.K * Te * Vg) / (2.0 * np.abs(I[voltage_step]) * Rsin * 1e-9))
            dP = (0.5 + np.power(mm, 2)) * np.power(self.constants.K * Te, 2) * np.abs(I[voltage_step]) * self.constants.E * 1e-9 * 1e24  # [pW²/Hz]
            
            NEPs = self.constants.NUMBER_OF_SINS_IN_CEB * (dI - 2.0 * dPdI + dP)  # [pW²/Hz]
            
            NEPph = 1e12 * np.sqrt(total_bolometers * 2.0 * (F * 1e9) * (dPbg * 1e-12) * self.constants.H +
                     np.power((dPbg * 1e-12) * total_bolometers, 2) / (dF * 1e9))  # [pW/sqrt(Hz)]
            
            NEP = np.sqrt((NEPe_ph2 + NEPs) * total_bolometers + NEPa + np.power(NEPph, 2))
            
            # Write Noise file
            file_noise.write(f"{(2.0 * V[voltage_step] + 1e-9 * I[voltage_step] * Ra) * M:.6e}\t")
            file_noise.write(f"{1e9 * np.sqrt(NEPe_ph2 * total_bolometers) * np.abs(Sv):.6e}\t")
            file_noise.write(f"{1e9 * np.sqrt(NEPs * total_bolometers) * np.abs(Sv):.6e}\t{1e9 * np.sqrt(NoiA):.6e}\t")
            file_noise.write(f"{1e9 * NEP * np.abs(Sv):.6e}\t{1e9 * NEPph * np.abs(Sv):.6e}\t")
            file_noise.write(f"{1e9 * np.abs(Sv) * np.sqrt(np.power(NEP, 2) - np.power(NEPph, 2)):.6e}\n")
            
            # Write NEP file
            file_NEP.write(f"{(2.0 * V[voltage_step] + 1e-9 * I[voltage_step] * Ra) * M:.6e}\t")
            file_NEP.write(f"{1e-9 * I[voltage_step] * MP:.6e}\t{1e-12 * np.sqrt(NEPe_ph2 * total_bolometers):.6e}\t")
            file_NEP.write(f"{1e-12 * np.sqrt(NEPs * total_bolometers):.6e}\t{1e-12 * np.sqrt(NEPa):.6e}\t")
            file_NEP.write(f"{1e-12 * NEP:.6e}\t{1e-12 * NEPph:.6e}\t{1e12 * np.abs(Sv):.6e}\t")
            file_NEP.write(f"{1e-12 * np.sqrt(np.power(NEP, 2) - np.power(NEPph, 2)):.6e}\n")
            
            # Write G file
            file_G.write(f"{(self.constants.NUMBER_OF_SINS_IN_CEB * V[voltage_step] + 1e-9 * I[voltage_step] * Ra) * M:.6e}\t")
            file_G.write(f"{G_e:.6e}\t{G_NIS:.6e}\n")
            
            print(f"{voltage_step:3d}/{voltage_steps - 1:3d}: V:{self.Vnum[voltage_step - 1]:10.6e}\t"
                  f"I:{self.Inum[voltage_step - 1]:10.6e}\tSv:{1e12 * np.abs(Sv):10.6e}\t"
                  f"Te:{Te:10.6e}\tNEPs:{1e-12 * np.sqrt(NEPs * total_bolometers):10.6e}\t"
                  f"NEPt:{1e-12 * NEP:10.6e}")
        
        file_noise.close()
        file_Te.close()
        file_NEP.close()
        file_G.close()
        
        print(f"Time spent: {time.time() - start_time:.2f} seconds")
        return voltage_steps - 1
    
    def compute_ceb_properties(self) -> int:
        """Compute CEB properties using the fastest available backend."""
        if self.use_cpp_backend and HAS_CPP_BACKEND:
            return self._compute_ceb_properties_cpp()
        else:
            return self._compute_ceb_properties_python()
    
    def _compute_ceb_properties_cpp(self) -> int:
        """Use C++ threaded backend for computation and write output files."""
        import time
        
        start_time = time.time()
        
        # Prepare parameters dictionary for C++ library
        params = self.par.copy()
        
        # Get amplifier noise parameters
        amp_noise = {
            'voltage_noise': self.amp_constants['voltage_noise'],
            'current_noise': self.amp_constants['current_noise']
        }
        
        # Call C++ threaded function
        result = compute_ceb_properties_threaded(params, amp_noise=amp_noise)
        
        # Store results in the IVParamFitter instance
        self.Inum = result['Inum']
        self.Vnum = result['Vnum']
        self.Te_num = result.get('Te')
        
        # Write output files from result data
        self._write_output_files(result, params)
        
        print(f"Time spent: {result['time_spent']:.2f} seconds")
        return len(result['Inum']) + 1
    
    def _write_output_files(self, result, params) -> None:
        """Write CEB output files from computed result data."""
        import os
        import datetime
        import numpy as np
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Physical parameters for file writing
        M = float(params['M'])
        MP = float(params['MP'])
        total_bolometers = M * MP
        
        # Check if detailed data is available
        detailed_data = result.get('Te') is not None
        
        if detailed_data:
            # Open output files in the output directory
            file_noise = open(self.output_dir / 'Noise.txt', 'w')
            file_Te = open(self.output_dir / 'Te.txt', 'w')
            file_NEP = open(self.output_dir / 'NEP.txt', 'w')
            file_G = open(self.output_dir / 'G.txt', 'w')
            
            # Write headers
            file_noise.write(f"Voltage\tNOISEep\tNOISEs\tNOISEa\tNOISE\tNOISEph\tNOISE^2-NOISEph^2\n")
            file_Te.write(f"Voltage\tCurrent\tIqp\tIand\tV/Rleak\tTe\tTs\tDeltaT\tPeph\tPand\tPleak\tPabs\tPcool\n")
            file_NEP.write(f"Voltage\tCurrent\tNEPeph\tNEPs\tNEPa\tNEP\tNEPph\tSv\tNEP^2-NEPph^2\n")
            file_G.write(f"Voltage\tGe\tGnis\n")
            
            # Write data rows
            for i in range(len(result['I'])):
                # Write Te file
                file_Te.write(f"{self.Vnum[i]:.6e}\t{self.Inum[i]:.6e}\t")
                file_Te.write(f"{1e-9 * result['I'][i] * MP:.6e}\t{1e-9 * result['I_A'][i] * MP:.6e}\t")
                file_Te.write(f"{1e9 * (result['I'][i] / params['Rleak']) * MP:.6e}\t{result['Te'][i]:.6e}\t{result['Tsin'][i]:.6e}\t")
                file_Te.write(f"{result['DeltaT'][i]:.6e}\t{result['Pe_ph'][i]:.6e}\t{result['Pand'][i]:.6e}\t")
                file_Te.write(f"{result['Pleak'][i]:.6e}\t{result['Pabs'][i]:.6e}\t{result['Pcool'][i]:.6e}\n")
                
                # Write Noise file
                file_noise.write(f"{(2.0 * result['I'][i] + 1e-9 * result['I'][i] * params['Ra']) * M:.6e}\t")
                file_noise.write(f"{1e9 * np.sqrt(result['NEPe_ph2'][i] * total_bolometers) * np.abs(result['Sv'][i]):.6e}\t")
                file_noise.write(f"{1e9 * np.sqrt(result['NEPs'][i] * total_bolometers) * np.abs(result['Sv'][i]):.6e}\t")
                file_noise.write(f"{1e9 * np.sqrt(result['NoiA'][i]):.6e}\t")
                file_noise.write(f"{1e9 * result['NEP'][i] * np.abs(result['Sv'][i]):.6e}\t")
                file_noise.write(f"{1e9 * result['NEPph'][i] * np.abs(result['Sv'][i]):.6e}\t")
                file_noise.write(f"{1e9 * np.abs(result['Sv'][i]) * np.sqrt(np.power(result['NEP'][i], 2) - np.power(result['NEPph'][i], 2)):.6e}\n")
                
                # Write NEP file
                file_NEP.write(f"{(2.0 * result['I'][i] + 1e-9 * result['I'][i] * params['Ra']) * M:.6e}\t")
                file_NEP.write(f"{1e-9 * result['I'][i] * MP:.6e}\t{1e-12 * np.sqrt(result['NEPe_ph2'][i] * total_bolometers):.6e}\t")
                file_NEP.write(f"{1e-12 * np.sqrt(result['NEPs'][i] * total_bolometers):.6e}\t{1e-12 * np.sqrt(result['NoiA'][i]):.6e}\t")
                file_NEP.write(f"{1e-12 * result['NEP'][i]:.6e}\t{1e-12 * result['NEPph'][i]:.6e}\t{1e12 * np.abs(result['Sv'][i]):.6e}\t")
                file_NEP.write(f"{1e-12 * np.sqrt(np.power(result['NEP'][i], 2) - np.power(result['NEPph'][i], 2)):.6e}\n")
                
                # Write G file
                file_G.write(f"{(2.0 * result['I'][i] + 1e-9 * result['I'][i] * params['Ra']) * M:.6e}\t")
                file_G.write(f"{result['G_e'][i]:.6e}\t{result['G_NIS'][i]:.6e}\n")
            
            file_noise.close()
            file_Te.close()
            file_NEP.close()
            file_G.close()
        else:
            # Create minimal output files if detailed data not available
            print("Warning: Detailed result data not available, writing minimal output files")
            
            file_Te = open(self.output_dir / 'Te.txt', 'w')
            file_Te.write(f"Voltage\tCurrent\n")
            for i in range(len(self.Vnum)):
                file_Te.write(f"{self.Vnum[i]:.6e}\t{self.Inum[i]:.6e}\n")
            file_Te.close()
            
            file_noise = open(self.output_dir / 'Noise.txt', 'w')
            file_noise.write(f"Voltage\tNOISE\n")
            for i in range(len(self.Vnum)):
                file_noise.write(f"{self.Vnum[i]:.6e}\t0.0\n")
            file_noise.close()
            
            file_NEP = open(self.output_dir / 'NEP.txt', 'w')
            file_NEP.write(f"Voltage\tNEP\n")
            for i in range(len(self.Vnum)):
                file_NEP.write(f"{self.Vnum[i]:.6e}\t0.0\n")
            file_NEP.close()
            
            file_G = open(self.output_dir / 'G.txt', 'w')
            file_G.write(f"Voltage\tG\n")
            for i in range(len(self.Vnum)):
                file_G.write(f"{self.Vnum[i]:.6e}\t0.0\n")
            file_G.close()
    
    def resample(self) -> tuple:
        if self.Vnum is None:
            raise RuntimeError("Vnum not calculated yet, run computeCEBProperties first")
        self.Irex, self.Vrex = Utils.resample(self.Iexp, self.Vexp, self.Vnum)
        return self.Irex, self.Vrex
    
    def compute_golubev_current(self) -> np.ndarray:
        """Compute the analytical Golubev SINIS current for the last computed state.

        Each SIN junction is modelled with the gap/current scale derived from the main
        parameters (par['Rn'], par['Ra'], par['Tc'], par['M'], par['MP']) so that the
        resulting current is directly comparable to the resampled experimental current
        on the same voltage grid. Returns None (and skips the extra term) when not
        computable, e.g. the detailed electron temperature is unavailable.
        """
        if self.Vnum is None or self.Inum is None or self.Te_num is None:
            self.Igol = None
            return None
        
        M = float(self.par['M'])
        MP = float(self.par['MP'])
        Ra = self.par['Ra']
        
        # Normal resistance of a single SIN junction (per-bolometer minus absorber,
        # split over the SINs of one bolometer), same decomposition as the numeric model.
        Rn_bolo = self.par['Rn'] * MP / M
        Rsin = (Rn_bolo - Ra) / self.constants.NUMBER_OF_SINS_IN_CEB
        
        # Superconducting gap energy in Joules (the model computes it in Kelvin).
        Delta_j = self.constants.BCS_INTEGRAL * self.par['Tc'] * self.constants.K * self.constants.E
        
        # Voltage across a single SIN junction: per-bolometer voltage minus the small
        # absorber drop, divided over the SINs of one bolometer.
        Vsin = (self.Vnum / M - (self.Inum / MP) * Ra) / self.constants.NUMBER_OF_SINS_IN_CEB
        
        self.Igol = MP * golubev_current(Vsin, self.Te_num, Rsin, Delta_j)
        return self.Igol
    
    def sequential_fit(self, run_count: int = 3) -> None:
        """Perform sequential fitting using golden section method"""
        import random
        
        # Reset evaluation counter for display
        self.eval_count = 0
        par_seq = [name for name, fit in self.to_fit.items() if fit]
        random.shuffle(par_seq)
        
        for run in range(run_count):
            print(f"SeqFit run {run}")
            
            fmin = float('nan')
            for param_name in par_seq:
                current_value = self.par[param_name]
                
                def objective(param_value: float) -> float:
                    return self._sequential_fit_objective(param_value, param_name)
                
                lower_bound = 0.5 * current_value
                upper_bound = 2.0 * current_value
                
                optimal_value, fmin = MinimizationAlgorithms.golden_section_minimization(
                    objective, lower_bound, upper_bound, tolerance=1e-3
                )
                
                self.par[param_name] = optimal_value
                print(f"  {param_name}: {current_value:.6e} -> {optimal_value:.6e}, fmin = {fmin:.6e}")
                   
            # Save results
            self._save_fit_results(fmin)

    def _init_brute_params(self) -> None:
        from lmfit import Parameters, minimize as lmfit_minimize
        params = Parameters()
        par_seq = []
        for name, to_init_brute in self.init_brute.items():
            if to_init_brute:
                params.add(name, value=self.par[name], vary=True, 
                            min=self.mins[name],
                            max=self.maxs[name])
                par_seq.append(name)
        if not par_seq:
            return
        
        print(f"Brute-initializing params: {par_seq}")
        num_brute_iterations = 100
        self._num_iterations = num_brute_iterations ** len(par_seq)
        self.eval_count = 0
        result = lmfit_minimize(self._lmfit_objective, params, method='brute', args=(par_seq,),
        Ns=100
        )
        for name in par_seq:
            param = result.params[name]
            print(f"Brute-initialized {name} to: {param.value}")
            self.par[name] = param.value

        
        fmin = result.chisqr
        self._save_fit_results(fmin)
    
    def _lmfit_objective(self, params, names_to_fit) -> np.ndarray:
        for name in names_to_fit:
            self.par[name] = params[name].value
        
        self.compute_ceb_properties()
        
        # Update display if enabled
        self.eval_count += 1
        for name in names_to_fit:
            if self._num_iterations is not None:
                print(f"Iteration {self.eval_count}/{self._num_iterations}: fmin {Utils.chi_sq(self.Inum, self.Irex)}")
            print(f"{name}: {params[name].value}")
        print(f"Current chi-sq: {Utils.chi_sq(self.Inum, self.Irex)}")
        print()

        self._update_display(self.Irex, self.Vrex)
        
        residual = (self.Inum - self.Irex) / (self.Irex * len(self.Irex))
        if self.compute_golubev_current() is not None:
            residual = np.concatenate([
                residual,
                (self.Igol - self.Irex) / (self.Irex * len(self.Irex))
            ])
        return residual

    def _sequential_fit_objective(self, param_value: float, param_name: str) -> float:
        old_value = self.par[param_name]
        self.par[param_name] = param_value

        self.compute_ceb_properties()
        Irex, Vrex = Utils.resample(self.Iexp, self.Vexp, self.Inum, self.Vnum)

        result = Utils.chi_sq_der(self.Vnum, self.Inum, Irex)

        # Update display if enabled
        self.eval_count += 1
        self._update_display(Irex, Vrex)

        if self.compute_golubev_current() is not None:
            result = result + Utils.chi_sq_golubev(self.Igol, Irex)

        self.par[param_name] = old_value
        return result

    def lmfit_sequential_fit(self, run_count: int = 3, method: str = 'leastsq') -> None:
        """Perform sequential fitting using lmfit"""
        self._init_brute_params()
        import random
        from lmfit import Parameters, minimize as lmfit_minimize
        
        # Reset evaluation counter for display
        self.eval_count = 0
        
        
        par_seq = [name for name, fit in self.to_fit.items() if fit]
        random.shuffle(par_seq)
        
        for run in range(run_count):
            print(f"LMFIT run {run}")
            
            params = Parameters()
            for name in par_seq:
                params.add(name, value=self.par[name], vary=True, 
                          min=0.5 * self.par[name] if run != 0 else self.mins[name],
                          max=2.0 * self.par[name] if run != 0 else self.maxs[name])
            
            self._num_iterations = None
            result = lmfit_minimize(self._lmfit_objective, params, method=method, args=(par_seq,))
            
            fmin = result.chisqr
            nfev = result.nfev
            
            print(f"  Final chi-square: {fmin:.6e}")
            print(f"  Number of function evaluations: {nfev}")
            print(f"  Fitted parameters with uncertainties:")
            
            for name in par_seq:
                param = result.params[name]
                value = param.value
                stderr = param.stderr
                
                self.par[name] = value
                
                if stderr is not None:
                    print(f"    {name}: {value:.6e} +/- {stderr:.6e}")
                else:
                    print(f"    {name}: {value:.6e} (no uncertainty available)")
            
            # Save results
            self._save_fit_results(fmin)
    
    def _compute_current_chi_sq(self) -> float:
        if self.Inum is None or self.Vnum is None:
            self.compute_ceb_properties()
            self.resample()
        result = Utils.chi_sq_der(self.Vnum, self.Inum, self.Irex)
        if self.compute_golubev_current() is not None:
            result = result + Utils.chi_sq_golubev(self.Igol, self.Irex)
        return result
    
    def _save_fit_results(self, fmin: float) -> None:
        fitparams_path = self.output_dir / 'fitparameters_new.txt'
        append_newline = fitparams_path.exists() and fitparams_path.stat().st_size > 0
        
        with open(fitparams_path, 'a') as params:
            if append_newline:
                params.write('\n')
            params.write(f"time = {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for param_name, param_value in self.par.items():
                fit_status = "fit" if self.to_fit.get(param_name, False) else "skip"
                params.write(f"{param_name} = {param_value} ({fit_status})\n")
            params.write(f"fmin = {fmin}\n")

