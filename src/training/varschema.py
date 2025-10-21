# src/training/varschema.py
from dataclasses import dataclass, field
from typing import List, Dict
import json, hashlib

@dataclass(frozen=True)
class VarSchema:
    daily_forcing:   List[str]
    monthly_forcing: List[str]
    monthly_states:  List[str]
    annual_forcing:  List[str]
    annual_states:   List[str]
    monthly_fluxes:  List[str]
    month_lengths:   List[int] = field(default_factory=lambda: [31,28,31,30,31,30,31,31,30,31,30,31])

    # Orders
    def input_order(self) -> List[str]:
        return (self.daily_forcing
              + self.monthly_forcing
              + self.monthly_states
              + self.annual_forcing
              + self.annual_states)

    def output_order(self) -> List[str]:
        return (self.monthly_fluxes
              + self.monthly_states
              + self.annual_states)

    # Head-local names
    def out_monthly_names(self) -> List[str]:
        # monthly head is fluxes + states, in global output order
        mf, ms = set(self.monthly_fluxes), set(self.monthly_states)
        return [n for n in self.output_order() if (n in mf or n in ms)]

    def out_annual_names(self) -> List[str]:
        a = set(self.annual_states)
        return [n for n in self.output_order() if n in a]

    # Index maps
    def map_input(self) -> Dict[str,int]:
        return {n:i for i,n in enumerate(self.input_order())}

    def map_output_global(self) -> Dict[str,int]:
        return {n:i for i,n in enumerate(self.output_order())}

    def map_out_monthly_local(self) -> Dict[str,int]:
        return {n:i for i,n in enumerate(self.out_monthly_names())}

    def map_out_annual_local(self) -> Dict[str,int]:
        return {n:i for i,n in enumerate(self.out_annual_names())}

    # Convenience index lists
    def in_monthly_state_idx(self) -> List[int]:
        m = self.map_input()
        return [m[n] for n in self.monthly_states]

    def in_annual_state_idx(self) -> List[int]:
        m = self.map_input()
        return [m[n] for n in self.annual_states]

    def out_monthly_state_idx_local(self) -> List[int]:
        m = self.map_out_monthly_local()
        return [m[n] for n in self.monthly_states]

    def out_annual_state_idx_local(self) -> List[int]:
        m = self.map_out_annual_local()
        return [m[n] for n in self.annual_states]

    # Dims
    def dims(self):
        nm = len(self.out_monthly_names())
        na = len(self.out_annual_names())
        return dict(
            input_dim=len(self.input_order()),
            output_dim=len(self.output_order()),
            nm=nm, na=na,
        )

    # Repro signature
    def signature(self) -> str:
        payload = dict(
            daily_forcing=self.daily_forcing,
            monthly_forcing=self.monthly_forcing,
            monthly_states=self.monthly_states,
            annual_forcing=self.annual_forcing,
            annual_states=self.annual_states,
            monthly_fluxes=self.monthly_fluxes,
            month_lengths=self.month_lengths,
        )
        s = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]