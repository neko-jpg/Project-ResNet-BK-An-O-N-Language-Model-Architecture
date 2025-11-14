"""
Tests for Physics-Informed Learning Components
"""

import torch
import torch.nn as nn
import pytest
import sys
sys.path.insert(0, 'src')

from models.physics_informed_layer import PhysicsInformedBKLayer
from training.physics_informed_trainer import PhysicsInformedTrainer
from training.symplectic_optimizer import SymplecticSGD, SymplecticAdam
from training.equilibrium_propagation import EquilibriumPropagationTrainer


class TestPhysicsInformedBKLayer:
    """Test PhysicsInformedBKLayer functionality."""
    
    def test_layer_creation(self):
        """Test that layer can be created."""
        layer = PhysicsInformedBKLayer(
            d_model=64,
            n_seq=128,
            num_experts=4,
            dropout_p=0.1
        )
        assert layer is not None
        assert layer.d_model == 64
        assert layer.n_seq == 128
    
    def test_forward_pass(self):
        """Test forward pass through layer."""
        layer = PhysicsInformedBKLayer(d_model=64, n_seq=128, num_experts=4)
        x = torch.randn(2, 128, 64)  # (B, N, D)
        
        output = layer(x)
        
        assert output.shape == (2, 128, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_energy_computation(self):
        """Test energy computation."""
        layer = PhysicsInformedBKLayer(d_model=64, n_seq=128, num_experts=4)
        x = torch.randn(2, 128, 64)
        x_prev = torch.randn(2, 128, 64)
        
        E_total, T_total, V_total = layer.compute_energy(x, x_prev)
        
        assert E_total.shape == (2,)
        assert T_total.shape == (2,)
        assert V_total.shape == (2,)
        assert not torch.isnan(E_total).any()
    
    def test_energy_conservation_loss(self):
        """Test energy conservation loss computation."""
        layer = PhysicsInformedBKLayer(d_model=64, n_seq=128, num_experts=4)
        E_current = torch.randn(2)
        E_prev = torch.randn(2)
        
        loss = layer.energy_conservation_loss(E_current, E_prev)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss).any()
    
    def test_hamiltonian_loss(self):
        """Test Hamiltonian loss computation."""
        layer = PhysicsInformedBKLayer(d_model=64, n_seq=128, num_experts=4)
        x = torch.randn(2, 128, 64)
        x_prev = torch.randn(2, 128, 64)
        
        loss, loss_dict = layer.hamiltonian_loss(x, x_prev)
        
        assert loss.item() >= 0
        assert 'loss_conservation' in loss_dict
        assert 'E_current' in loss_dict
        assert 'lambda_energy' in loss_dict


class TestSymplecticOptimizers:
    """Test symplectic optimizers."""
    
    def test_symplectic_sgd_creation(self):
        """Test SymplecticSGD creation."""
        model = nn.Linear(10, 10)
        optimizer = SymplecticSGD(model.parameters(), lr=0.01)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
    
    def test_symplectic_sgd_step(self):
        """Test SymplecticSGD optimization step."""
        model = nn.Linear(10, 10)
        optimizer = SymplecticSGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Forward and backward
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Check velocity states exist
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    assert 'velocity' in optimizer.state[p]
    
    def test_symplectic_adam_creation(self):
        """Test SymplecticAdam creation."""
        model = nn.Linear(10, 10)
        optimizer = SymplecticAdam(model.parameters(), lr=0.001, symplectic=True)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
    
    def test_symplectic_adam_step(self):
        """Test SymplecticAdam optimization step."""
        model = nn.Linear(10, 10)
        optimizer = SymplecticAdam(model.parameters(), lr=0.001, symplectic=True)
        
        # Forward and backward
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Check velocity states exist
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    assert 'velocity' in optimizer.state[p]
                    assert 'exp_avg' in optimizer.state[p]
                    assert 'exp_avg_sq' in optimizer.state[p]


class TestPhysicsInformedTrainer:
    """Test PhysicsInformedTrainer."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        from models.resnet_bk import LanguageModel
        
        model = LanguageModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_seq=128,
            num_experts=4
        )
        
        # Replace with physics-informed layers
        for block in model.blocks:
            block.bk_layer = PhysicsInformedBKLayer(
                d_model=64,
                n_seq=128,
                num_experts=4
            )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        trainer = PhysicsInformedTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_energy_init=0.1
        )
        
        assert trainer is not None
        assert trainer.lambda_energy_lr == 0.01
    
    def test_train_step(self):
        """Test single training step."""
        from models.resnet_bk import LanguageModel
        
        model = LanguageModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_seq=128,
            num_experts=4
        )
        
        # Replace with physics-informed layers
        for block in model.blocks:
            block.bk_layer = PhysicsInformedBKLayer(
                d_model=64,
                n_seq=128,
                num_experts=4
            )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        trainer = PhysicsInformedTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )
        
        # Create dummy batch
        x_batch = torch.randint(0, 1000, (2, 128))
        y_batch = torch.randint(0, 1000, (2 * 128,))
        x_prev_batch = torch.randint(0, 1000, (2, 128))
        
        # Training step
        metrics = trainer.train_step(x_batch, y_batch, x_prev_batch)
        
        assert 'loss_total' in metrics
        assert 'loss_lm' in metrics
        assert 'loss_energy' in metrics
        assert metrics['loss_total'] >= 0


class TestEquilibriumPropagation:
    """Test EquilibriumPropagationTrainer."""
    
    def test_trainer_creation(self):
        """Test EP trainer creation."""
        from models.resnet_bk import LanguageModel
        
        model = LanguageModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_seq=128,
            num_experts=4
        )
        
        # Replace with physics-informed layers
        for block in model.blocks:
            block.bk_layer = PhysicsInformedBKLayer(
                d_model=64,
                n_seq=128,
                num_experts=4
            )
        
        trainer = EquilibriumPropagationTrainer(
            model=model,
            beta=0.5,
            n_relax_steps=5,
            lr=0.01
        )
        
        assert trainer is not None
        assert trainer.beta == 0.5
        assert trainer.n_relax_steps == 5
    
    def test_energy_computation(self):
        """Test total energy computation."""
        from models.resnet_bk import LanguageModel
        
        model = LanguageModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_seq=128,
            num_experts=4
        )
        
        # Replace with physics-informed layers
        for block in model.blocks:
            block.bk_layer = PhysicsInformedBKLayer(
                d_model=64,
                n_seq=128,
                num_experts=4
            )
        
        trainer = EquilibriumPropagationTrainer(model=model)
        
        hidden_states = torch.randn(2, 128, 64)
        energy = trainer.compute_total_energy(hidden_states)
        
        assert energy.shape == (2,)
        assert not torch.isnan(energy).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
