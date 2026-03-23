from .backbone import SharedTransformerBlock, Embedding, OutputHead
from .modulator import ModulatorNetwork
from .halting import (
    AttractorHalt, LearnedHalt, ModulatorHalt, EnergyBudgetHalt,
    InhibitoryDamping, SynapticDepression, HaltCombiner
)
from .oscillator import OscillatoryGating
