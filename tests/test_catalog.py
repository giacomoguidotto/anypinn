"""Tests for anypinn.catalog — ODE functions and DataModules."""

import pytest
import torch
from torch import Tensor

from anypinn.core.nn import Argument


class TestSIRODE:
    def test_sir_output_shape(self):
        from anypinn.catalog.sir import SIR

        x = torch.tensor(0.0)
        y = torch.stack([torch.tensor([100.0]), torch.tensor([1.0])])
        args = {
            "beta": Argument(0.3),
            "delta": Argument(0.1),
            "N": Argument(1000.0),
        }
        result = SIR(x, y, args)
        assert result.shape == (2, 1)

    def test_rsir_output_shape(self):
        from anypinn.catalog.sir import rSIR

        x = torch.tensor(0.0)
        y = torch.tensor([10.0])
        args = {
            "delta": Argument(0.1),
            "Rt": Argument(2.5),
        }
        result = rSIR(x, y, args)
        assert result.shape == (1,)

    def test_sir_conservation(self):
        """dS + dI should have a specific relationship with dR (= delta * I)."""
        from anypinn.catalog.sir import SIR

        x = torch.tensor(0.0)
        S_val, I_val = 990.0, 10.0
        y = torch.stack([torch.tensor([S_val]), torch.tensor([I_val])])
        args = {
            "beta": Argument(0.3),
            "delta": Argument(0.1),
            "N": Argument(1000.0),
        }
        result = SIR(x, y, args)
        dS, dI = result[0], result[1]
        assert (dS + dI).item() == pytest.approx(-0.1 * 10.0, rel=1e-4)


class TestSIRInvDataModule:
    def test_setup_produces_correct_coll_shape(self):
        from anypinn.catalog.sir import SIRInvDataModule
        from anypinn.core.config import GenerationConfig, MLPConfig, ScalarConfig
        from anypinn.core.nn import Argument
        from anypinn.problems.ode import ODEHyperparameters, ODEProperties

        hp = ODEHyperparameters(
            lr=1e-3,
            training_data=GenerationConfig(
                batch_size=16,
                data_ratio=0.5,
                collocations=200,
                x=torch.linspace(0, 50, 100),
                noise_level=0.0,
                args_to_train={},
            ),
            fields_config=MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh"),
            params_config=ScalarConfig(init_value=0.5),
        )

        def sir_ode(x: Tensor, y: Tensor, args: dict) -> Tensor:
            return -y

        props = ODEProperties(
            ode=sir_ode,
            args={"beta": Argument(0.3), "delta": Argument(0.1), "N": Argument(1000.0)},
            y0=torch.tensor([990.0, 10.0]),
        )
        dm = SIRInvDataModule(hp=hp, gen_props=props)
        dm.setup()

        assert dm.coll.shape == (200, 1)
        assert dm.coll.min().item() >= 0.0


class TestDampedOscillatorDataModule:
    def test_setup_produces_correct_coll_shape(self):
        from anypinn.catalog.damped_oscillator import DampedOscillatorDataModule
        from anypinn.core.config import GenerationConfig, MLPConfig, ScalarConfig
        from anypinn.problems.ode import ODEHyperparameters, ODEProperties

        hp = ODEHyperparameters(
            lr=1e-3,
            training_data=GenerationConfig(
                batch_size=16,
                data_ratio=0.5,
                collocations=100,
                x=torch.linspace(0, 10, 50),
                noise_level=0.0,
                args_to_train={},
            ),
            fields_config=MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh"),
            params_config=ScalarConfig(init_value=0.5),
        )

        def osc_ode(x: Tensor, y: Tensor, args: dict) -> Tensor:
            return -y

        props = ODEProperties(ode=osc_ode, args={}, y0=torch.tensor([1.0, 0.0]))
        dm = DampedOscillatorDataModule(hp=hp, gen_props=props)
        dm.setup()

        assert dm.coll.shape == (100, 1)
        assert dm.coll.min().item() >= 0.0
        assert dm.coll.max().item() <= 10.0


class TestLotkaVolterraDataModule:
    def test_setup_produces_correct_coll_shape(self):
        from anypinn.catalog.lotka_volterra import LotkaVolterraDataModule
        from anypinn.core.config import GenerationConfig, MLPConfig, ScalarConfig
        from anypinn.problems.ode import ODEHyperparameters, ODEProperties

        hp = ODEHyperparameters(
            lr=1e-3,
            training_data=GenerationConfig(
                batch_size=16,
                data_ratio=0.5,
                collocations=100,
                x=torch.linspace(0, 10, 50),
                noise_level=0.0,
                args_to_train={},
            ),
            fields_config=MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh"),
            params_config=ScalarConfig(init_value=0.5),
        )

        def lv_ode(x: Tensor, y: Tensor, args: dict) -> Tensor:
            return -y

        props = ODEProperties(ode=lv_ode, args={}, y0=torch.tensor([10.0, 5.0]))
        dm = LotkaVolterraDataModule(hp=hp, gen_props=props)
        dm.setup()

        assert dm.coll.shape == (100, 1)


class TestSEIRDataModule:
    def test_setup_produces_correct_coll_shape(self):
        from anypinn.catalog.seir import SEIRDataModule
        from anypinn.core.config import GenerationConfig, MLPConfig, ScalarConfig
        from anypinn.problems.ode import ODEHyperparameters, ODEProperties

        hp = ODEHyperparameters(
            lr=1e-3,
            training_data=GenerationConfig(
                batch_size=16,
                data_ratio=0.5,
                collocations=100,
                x=torch.linspace(0, 10, 50),
                noise_level=0.0,
                args_to_train={},
            ),
            fields_config=MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh"),
            params_config=ScalarConfig(init_value=0.5),
        )

        def seir_ode(x: Tensor, y: Tensor, args: dict) -> Tensor:
            return -y

        props = ODEProperties(ode=seir_ode, args={}, y0=torch.tensor([990.0, 0.0, 10.0]))
        dm = SEIRDataModule(hp=hp, gen_props=props)
        dm.setup()

        assert dm.coll.shape == (100, 1)
