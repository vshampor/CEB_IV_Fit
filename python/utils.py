import numpy as np
import json
from pathlib import Path
from typing import Tuple, Any, Dict, Callable, Union

class Utils:
    @staticmethod
    def load_experimental_data(filename: Union[str, Path], remove_offset: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        try:
            data = np.loadtxt(filename, delimiter=' ')
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            Vexp = data[:, 0]
            Iexp = data[:, 1]
        except Exception as e:
            raise RuntimeError(f"Unable to read experimental data from \"{filename}\": {e}")
        
        print(f"Fitting for \"{filename}\" started!")
        
        with open("fitparameters_new.txt", 'a') as params:
            params.write(f"data source = \"{filename}\"\n")
        
        if remove_offset:
            offset = Utils.eliminate_offset(Iexp, Vexp)
            Vexp -= offset
            Utils.write_iv("IV offset.txt", Iexp, Vexp)
        
        return Iexp, Vexp
    
    @staticmethod
    def resample(Iexp: np.ndarray, Vexp: np.ndarray, Vnum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(Iexp) != len(Vexp):
            raise ValueError("Experimental I and V must be of the same size")
        
        countnum = len(Vnum)
        countexp = len(Vexp)
        Irex = np.zeros(countnum)
        Vrex = Vnum.copy()
        
        for i in range(countnum):
            dCurVal = Vnum[i]
            ddCurVal = np.abs(Vnum[i] - Vexp)
            lCurPos = np.argmin(ddCurVal)
            
            if (lCurPos < countexp - 1) and ((Vexp[lCurPos] <= dCurVal) == (dCurVal <= Vexp[lCurPos + 1])):
                lLeft = lCurPos
                lRight = lCurPos + 1
            else:
                lLeft = lCurPos - 1
                lRight = lCurPos
            
            if lRight == 0:
                Irex[i] = Iexp[0]
            elif lRight > countexp - 1:
                Irex[i] = Iexp[countexp - 1]
            else:
                Irex[i] = Iexp[lLeft] + (dCurVal - Vexp[lLeft]) * (Iexp[lRight] - Iexp[lLeft]) / (Vexp[lRight] - Vexp[lLeft])
        
        return Irex, Vrex
    
    @staticmethod
    def chi_sq(Inum: np.ndarray, Irex: np.ndarray) -> float:
        if len(Inum) != len(Irex):
            raise ValueError("Numeric I and Recalculated I must be of the same size")
        return np.sum(np.power((Inum - Irex) / Irex, 2)) / len(Inum)
    
    @staticmethod
    def chi_sq_hi(Inum: np.ndarray, Irex: np.ndarray) -> float:
        if len(Inum) != len(Irex):
            raise ValueError("Numeric I and Recalculated I must be of the same size")
        
        dI2 = np.power(Inum - Irex, 2)
        countnum = len(Inum)
        indices = np.arange(countnum)
        sum_val = np.sum(dI2 * indices)
        return 1e8 * sum_val / countnum
    
    @staticmethod
    def chi_sq_golubev(Igol: np.ndarray, Irex: np.ndarray) -> float:
        if len(Igol) != len(Irex):
            raise ValueError("Golubev I and Recalculated I must be of the same size")
        return np.sum(np.power((Igol - Irex) / Irex, 2)) / len(Igol)
    
    @staticmethod
    def chi_sq_der(Vnum: np.ndarray, Inum: np.ndarray, Irex: np.ndarray, return_array: bool = False) -> float:
        if len(Inum) != len(Vnum):
            raise ValueError("Numeric I and V must be of the same size")
        if len(Inum) != len(Irex):
            raise ValueError("Numeric I and Recalculated I must be of the same size")
        
        countnum = len(Vnum)
        dVnum = Vnum[1:] - Vnum[:-1]
        dInum = Inum[1:] - Inum[:-1]
        dIrex = Irex[1:] - Irex[:-1]
        
        chi2 = Utils.chi_sq(Inum, Irex)
        chi2_der = np.sum(np.power((dInum - dIrex) / dVnum, 2) / np.power(dIrex / dVnum, 2)) / (countnum - 1)
        
        return chi2 + chi2_der
    
    @staticmethod
    def write_iv(filename: Union[str, Path], I: np.ndarray, V: np.ndarray) -> None:
        if len(I) != len(V):
            raise ValueError("I and V must be of the same size")
        
        data = np.column_stack((V, I))
        np.savetxt(filename, data, delimiter='\t')
    
    @staticmethod
    def eliminate_offset(Iexp: np.ndarray, Vexp: np.ndarray) -> float:
        dOffset = 0.0
        dLBound = -0.0005
        dRBound = 0.0005
        
        def objective(offset):
            Voffset = Vexp - offset
            return Utils.symmetrize_measure(Iexp, Voffset)
        
        dOffset = Utils.golden_minimize(objective, dLBound, dRBound, dOffset)
        return dOffset
    
    @staticmethod
    def symmetrize_measure(Iexp: np.ndarray, Vexp: np.ndarray) -> float:
        if not np.all(Iexp[:-1] <= Iexp[1:]):
            raise RuntimeError("I doesn't rise monotonously")
        
        Vofx = Vexp
        Iofx = Iexp
        
        absIofx = np.abs(Iofx)
        indexOfTheSmallestAbsIofx = np.argmin(np.where(absIofx >= 0, absIofx, np.inf))
        
        count = len(Iexp)
        lLengthPos = count - indexOfTheSmallestAbsIofx
        lLengthNeg = indexOfTheSmallestAbsIofx
        
        if lLengthPos > lLengthNeg:
            lLengthHigh = lLengthPos
            lLengthLow = lLengthNeg
            Vhigh = Vofx[indexOfTheSmallestAbsIofx:]
            Ihigh = Iofx[indexOfTheSmallestAbsIofx:]
            Vlow = Vofx[indexOfTheSmallestAbsIofx - 1::-1]
            Ilow = Iofx[indexOfTheSmallestAbsIofx - 1::-1]
        else:
            lLengthHigh = lLengthNeg
            lLengthLow = lLengthPos
            Vhigh = Vofx[indexOfTheSmallestAbsIofx - 1::-1]
            Ihigh = Iofx[indexOfTheSmallestAbsIofx - 1::-1]
            Vlow = Vofx[indexOfTheSmallestAbsIofx:]
            Ilow = Iofx[indexOfTheSmallestAbsIofx:]
        
        Utils.write_iv("IV low.txt", Ilow, Vlow)
        Utils.write_iv("IV high.txt", Ihigh, Vhigh)
        
        Irex, Vrex = Utils.resample(Ihigh, Vhigh, Vlow)
        Utils.write_iv("IV rex.txt", Irex, Vrex)
        
        return Utils.chi_sq_hi(Ilow, Irex)
    
    @staticmethod
    def golden_minimize(f: Callable[[float], float], a: float, b: float, x_initial: float, tolerance: float = 1e-8) -> float:
        R = (np.sqrt(5.0) - 1.0) / 2.0
        C = 1.0 - R
        
        if abs(b - x_initial) > abs(x_initial - a):
            x1 = x_initial
            x2 = x_initial + C * (b - x_initial)
        else:
            x2 = x_initial
            x1 = x_initial - C * (x_initial - a)
        
        f1 = f(x1)
        f2 = f(x2)
        
        while abs(b - a) > tolerance * (abs(x1) + abs(x2)):
            if f2 < f1:
                a = x1
                x1 = x2
                f1 = f2
                x2 = R * x2 + C * b
                f2 = f(x2)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = R * x1 + C * a
                f1 = f(x1)
        
        return x1 if f1 < f2 else x2
    
    @staticmethod
    def load_json_config(filename: Union[str, Path]) -> Dict[str, Any]:
        with open(filename, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json_config(filename: Union[str, Path], config: Dict[str, Any]) -> None:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
    
    @staticmethod
    def create_sample_config() -> Dict[str, Any]:
        sample_config: Dict[str, Any] = {
            "data_file": "SPC-CEB_300mK_Triton11-2026.txt",
            "amp_type": "AD745",
            "sep": "\t",
            "threads": 56,
            "parameters": {
                "Pbg": {"value": 0.0, "vary": False, "min": 0.0, "max": 1.0},
                "beta": {"value": 0.111, "vary": True, "min": 0.0, "max": 1.0},
                "TephPOW": {"value": 5.0, "vary": False, "min": 1.0, "max": 7.0},
                "Vol": {"value": 0.02, "vary": False, "min": 0.001, "max": 0.1},
                "Z": {"value": 0.5, "vary": True, "min": 0.1, "max": 2.0},
                "Tc": {"value": 1.18, "vary": False, "min": 1.0, "max": 1.5},
                "Rn": {"value": 11500.0, "vary": False, "min": 5000.0, "max": 20000.0},
                "Rleak": {"value": 40000000.0, "vary": False, "min": 1e6, "max": 1e8},
                "Wt": {"value": 0.0001, "vary": False, "min": 0.0, "max": 0.001},
                "tm": {"value": 1.0, "vary": False, "min": 0.5, "max": 2.0},
                "ii": {"value": 0.0, "vary": False, "min": 0.0, "max": 1.0},
                "Ra": {"value": 200.0, "vary": False, "min": 50.0, "max": 1000.0},
                "M": {"value": 2, "vary": False, "min": 1, "max": 10},
                "MP": {"value": 1, "vary": False, "min": 1, "max": 10},
                "Tp": {"value": 0.19, "vary": True, "min": 0.1, "max": 0.4},
                "F": {"value": 14.2, "vary": False, "min": 1.0, "max": 100.0},
                "dF": {"value": 0.1, "vary": False, "min": 0.01, "max": 1.0},
                "dVFinVg": {"value": 1.1, "vary": False, "min": 0.5, "max": 2.0},
                "dVStartVg": {"value": 0.0, "vary": False, "min": 0.0, "max": 0.5},
                "dV": {"value": 2e-6, "vary": False, "min": 1e-6, "max": 10e-6}
            }
        }
        return sample_config