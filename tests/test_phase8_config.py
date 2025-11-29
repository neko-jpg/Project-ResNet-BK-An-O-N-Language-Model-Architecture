"""
Tests for Phase 8 Configuration (Task 37.2)

Validates:
- Phase8Config correctly inherits from Phase7Config
- All Phase 7 parameters are accessible
- Phase 8 specific parameters are properly defined
- Default values are reasonable
- Configuration serialization works
"""
import pytest
import torch
from dataclasses import asdict

from src.models.phase8.config import Phase8Config, Phase8Diagnostics


class TestPhase8ConfigInheritance:
    """Test Phase8Config inheritance from Phase7Config."""
    
    def test_inherits_from_phase7config(self):
        """Phase8Config should inherit from Phase7Config."""
        from src.models.phase7.integrated_model import Phase7Config
        
        # Check that Phase8Config is a subclass of Phase7Config
        assert issubclass(Phase8Config, Phase7Config), "Phase8Config must inherit from Phase7Config"
        
        # Verify instance has Phase7Config in its MRO
        config = Phase8Config()
        assert Phase7Config in type(config).__mro__, "Phase7Config should be in Phase8Config's MRO"
    
    def test_has_phase7_parameters(self):
        """Phase8Config should have all Phase7 parameters."""
        config = Phase8Config()
        
        # Check key Phase 7 parameters
        assert hasattr(config, 'vocab_size'), "Missing vocab_size from Phase7Config"
        assert hasattr(config, 'd_model'), "Missing d_model from Phase7Config"
        assert hasattr(config, 'n_layers'), "Missing n_layers from Phase7Config"
        assert hasattr(config, 'htt_rank'), "Missing htt_rank from Phase7Config"
        assert hasattr(config, 'use_hybrid_attention'), "Missing use_hybrid_attention from Phase7Config"
        assert hasattr(config, 'hyperbolic_window_size'), "Missing hyperbolic_window_size from Phase7Config"
        assert hasattr(config, 'num_heads'), "Missing num_heads from Phase7Config"
        assert hasattr(config, 'use_triton_kernel'), "Missing use_triton_kernel from Phase7Config"
        assert hasattr(config, 'use_gradient_checkpointing'), "Missing use_gradient_checkpointing from Phase7Config"
        assert hasattr(config, 'use_mixed_precision'), "Missing use_mixed_precision from Phase7Config"
    
    def test_has_resnetbk_parameters(self):
        """Phase8Config should have ResNetBK parameters (via Phase7)."""
        config = Phase8Config()
        
        # Check key ResNetBK parameters
        assert hasattr(config, 'num_experts'), "Missing num_experts from ResNetBKConfig"
        assert hasattr(config, 'top_k'), "Missing top_k from ResNetBKConfig"
        assert hasattr(config, 'use_scattering_router'), "Missing use_scattering_router"
        assert hasattr(config, 'use_birman_schwinger'), "Missing use_birman_schwinger"
        assert hasattr(config, 'use_mourre'), "Missing use_mourre"
        assert hasattr(config, 'use_lap'), "Missing use_lap"
        assert hasattr(config, 'ar_ssm_max_rank'), "Missing ar_ssm_max_rank"
        assert hasattr(config, 'ar_ssm_min_rank'), "Missing ar_ssm_min_rank"


class TestPhase8SpecificParameters:
    """Test Phase 8 specific parameters."""
    
    def test_has_bk_hyperbolic_parameters(self):
        """Phase8Config should have BK-Core hyperbolic integration parameters."""
        config = Phase8Config()
        
        assert hasattr(config, 'use_bk_hyperbolic'), "Missing use_bk_hyperbolic"
        assert hasattr(config, 'bk_hyperbolic_gate_scale'), "Missing bk_hyperbolic_gate_scale"
        assert hasattr(config, 'bk_hyperbolic_resonance_threshold'), "Missing bk_hyperbolic_resonance_threshold"
        
        # Check defaults
        assert config.use_bk_hyperbolic == True, "BK hyperbolic should be enabled by default"
        assert config.bk_hyperbolic_gate_scale == 1.0, "Default gate scale should be 1.0"
    
    def test_has_ar_ssm_fusion_parameters(self):
        """Phase8Config should have AR-SSM fusion parameters."""
        config = Phase8Config()
        
        assert hasattr(config, 'use_ar_ssm_fusion'), "Missing use_ar_ssm_fusion"
        assert hasattr(config, 'ar_ssm_hyperbolic_rank_threshold'), "Missing ar_ssm_hyperbolic_rank_threshold"
        assert hasattr(config, 'ar_ssm_curvature_adaptation_rate'), "Missing ar_ssm_curvature_adaptation_rate"
        
        # Check defaults
        assert config.use_ar_ssm_fusion == True, "AR-SSM fusion should be enabled by default"
    
    def test_has_optional_component_flags(self):
        """Phase8Config should have flags for optional components."""
        config = Phase8Config()
        
        # Optional components (should be False by default for stability)
        assert hasattr(config, 'enable_entailment_cones'), "Missing enable_entailment_cones"
        assert hasattr(config, 'enable_persistent_homology'), "Missing enable_persistent_homology"
        assert hasattr(config, 'enable_sheaf_attention'), "Missing enable_sheaf_attention"
        assert hasattr(config, 'enable_adaptive_computation'), "Missing enable_adaptive_computation"
        assert hasattr(config, 'enable_koopman_bridge'), "Missing enable_koopman_bridge"
        assert hasattr(config, 'enable_sparse_attention'), "Missing enable_sparse_attention"
        assert hasattr(config, 'enable_kv_compression'), "Missing enable_kv_compression"
        assert hasattr(config, 'enable_curvature_adaptation'), "Missing enable_curvature_adaptation"
        
        # Numerical guards should be enabled by default
        assert hasattr(config, 'enable_numerical_guards'), "Missing enable_numerical_guards"
        assert config.enable_numerical_guards == True, "Numerical guards should be enabled by default"
    
    def test_component_specific_parameters(self):
        """Phase8Config should have parameters for each component."""
        config = Phase8Config()
        
        # Entailment
        assert hasattr(config, 'entailment_aperture'), "Missing entailment_aperture"
        assert hasattr(config, 'entailment_margin'), "Missing entailment_margin"
        
        # Topology
        assert hasattr(config, 'topology_persistence_threshold'), "Missing topology_persistence_threshold"
        assert hasattr(config, 'topology_max_dimension'), "Missing topology_max_dimension"
        assert hasattr(config, 'topology_betti_threshold'), "Missing topology_betti_threshold"
        
        # Sheaf
        assert hasattr(config, 'sheaf_agreement_threshold'), "Missing sheaf_agreement_threshold"
        assert hasattr(config, 'sheaf_num_sections'), "Missing sheaf_num_sections"
        
        # Adaptive
        assert hasattr(config, 'adaptive_exit_threshold'), "Missing adaptive_exit_threshold"
        assert hasattr(config, 'adaptive_min_layers'), "Missing adaptive_min_layers"
        
        # Sparse
        assert hasattr(config, 'sparse_top_k'), "Missing sparse_top_k"
        assert hasattr(config, 'sparse_block_size'), "Missing sparse_block_size"
        
        # KV Cache
        assert hasattr(config, 'kv_cache_dim'), "Missing kv_cache_dim"
        assert hasattr(config, 'kv_eviction_threshold'), "Missing kv_eviction_threshold"
        
        # Curvature
        assert hasattr(config, 'curvature_initial'), "Missing curvature_initial"
        assert hasattr(config, 'curvature_min'), "Missing curvature_min"
        assert hasattr(config, 'curvature_max'), "Missing curvature_max"
        
        # Safety
        assert hasattr(config, 'max_norm'), "Missing max_norm"
        assert hasattr(config, 'grad_clip'), "Missing grad_clip"


class TestPhase8ConfigDefaults:
    """Test default values are reasonable."""
    
    def test_phase7_defaults_preserved(self):
        """Phase 7 defaults should be preserved or reasonably overridden."""
        config = Phase8Config()
        
        # Phase 7 defaults that should be preserved
        assert config.htt_rank == 16, "HTT rank should default to 16"
        assert config.use_hybrid_attention == True, "Hybrid attention should be enabled"
        assert config.hyperbolic_window_size == 64, "Hyperbolic window should be 64"
        assert config.num_heads == 8, "Should have 8 attention heads"
    
    def test_phase8_optimization_defaults(self):
        """Phase 8 should enable optimizations by default."""
        config = Phase8Config()
        
        assert config.use_gradient_checkpointing == True, "Gradient checkpointing should be enabled"
        assert config.use_mixed_precision == True, "Mixed precision should be enabled"
        assert config.use_triton_kernel == True, "Triton kernels should be enabled"
        assert config.triton_kernel_version == 'fast', "Should use fast kernel by default"
    
    def test_reasonable_numerical_bounds(self):
        """Numerical parameters should have reasonable bounds."""
        config = Phase8Config()
        
        # Curvature bounds
        assert 0 < config.curvature_min < config.curvature_initial < config.curvature_max
        assert config.curvature_min == 0.1
        assert config.curvature_max == 5.0
        
        # Norm bounds
        assert 0 < config.max_norm < 1.0, "Max norm should be less than 1.0 for hyperbolic space"
        assert config.max_norm == 0.99
        
        # Gradient clipping
        assert config.grad_clip > 0, "Gradient clipping should be positive"


class TestPhase8ConfigModification:
    """Test configuration can be modified."""
    
    def test_can_modify_phase7_parameters(self):
        """Should be able to modify inherited Phase 7 parameters."""
        config = Phase8Config(
            d_model=128,
            n_layers=8,
            htt_rank=32,
            num_heads=16
        )
        
        assert config.d_model == 128
        assert config.n_layers == 8
        assert config.htt_rank == 32
        assert config.num_heads == 16
    
    def test_can_modify_phase8_parameters(self):
        """Should be able to modify Phase 8 specific parameters."""
        config = Phase8Config(
            use_bk_hyperbolic=False,
            enable_entailment_cones=True,
            enable_persistent_homology=True,
            curvature_initial=2.0
        )
        
        assert config.use_bk_hyperbolic == False
        assert config.enable_entailment_cones == True
        assert config.enable_persistent_homology == True
        assert config.curvature_initial == 2.0
    
    def test_can_enable_all_components(self):
        """Should be able to enable all optional components."""
        config = Phase8Config(
            enable_entailment_cones=True,
            enable_persistent_homology=True,
            enable_sheaf_attention=True,
            enable_adaptive_computation=True,
            enable_koopman_bridge=True,
            enable_sparse_attention=True,
            enable_kv_compression=True,
            enable_curvature_adaptation=True
        )
        
        assert config.enable_entailment_cones == True
        assert config.enable_persistent_homology == True
        assert config.enable_sheaf_attention == True
        assert config.enable_adaptive_computation == True
        assert config.enable_koopman_bridge == True
        assert config.enable_sparse_attention == True
        assert config.enable_kv_compression == True
        assert config.enable_curvature_adaptation == True


class TestPhase8Diagnostics:
    """Test Phase8Diagnostics dataclass."""
    
    def test_diagnostics_creation(self):
        """Should be able to create diagnostics object."""
        diag = Phase8Diagnostics()
        assert diag is not None
    
    def test_has_bk_hyperbolic_metrics(self):
        """Diagnostics should have BK-Core hyperbolic metrics."""
        diag = Phase8Diagnostics()
        
        assert hasattr(diag, 'bk_hyperbolic_gate_mean')
        assert hasattr(diag, 'bk_hyperbolic_gate_std')
        assert hasattr(diag, 'bk_resonance_detected')
        assert hasattr(diag, 'bk_resonance_strength')
    
    def test_has_ar_ssm_metrics(self):
        """Diagnostics should have AR-SSM fusion metrics."""
        diag = Phase8Diagnostics()
        
        assert hasattr(diag, 'ar_ssm_rank_mean')
        assert hasattr(diag, 'ar_ssm_hyperbolic_distance_mean')
        assert hasattr(diag, 'ar_ssm_curvature_adjusted')
    
    def test_has_component_metrics(self):
        """Diagnostics should have metrics for all components."""
        diag = Phase8Diagnostics()
        
        # Entailment
        assert hasattr(diag, 'entailment_violation_rate')
        assert hasattr(diag, 'avg_aperture')
        
        # Topology
        assert hasattr(diag, 'betti_numbers')
        assert hasattr(diag, 'persistent_entropy')
        assert hasattr(diag, 'circular_reasoning_detected')
        
        # Sheaf
        assert hasattr(diag, 'sheaf_agreement_mean')
        assert hasattr(diag, 'sheaf_consensus_rate')
        
        # Adaptive
        assert hasattr(diag, 'avg_layers_executed')
        assert hasattr(diag, 'early_exit_rate')
        assert hasattr(diag, 'adaptive_compute_savings')
        
        # Hyperbolic
        assert hasattr(diag, 'avg_hyperbolic_norm')
        assert hasattr(diag, 'curvature_value')
        assert hasattr(diag, 'boundary_collapse_warnings')
    
    def test_has_performance_metrics(self):
        """Diagnostics should have performance metrics."""
        diag = Phase8Diagnostics()
        
        assert hasattr(diag, 'peak_memory_mb')
        assert hasattr(diag, 'forward_time_ms')
        assert hasattr(diag, 'tokens_per_second')
    
    def test_has_safety_metrics(self):
        """Diagnostics should have numerical safety metrics."""
        diag = Phase8Diagnostics()
        
        assert hasattr(diag, 'gradient_overflow_count')
        assert hasattr(diag, 'nan_detected')
        assert hasattr(diag, 'inf_detected')
        assert hasattr(diag, 'precision_upcast_count')
    
    def test_diagnostics_defaults(self):
        """Diagnostics should have reasonable default values."""
        diag = Phase8Diagnostics()
        
        assert diag.bk_hyperbolic_gate_mean == 0.0
        assert diag.bk_resonance_detected == False
        assert diag.nan_detected == False
        assert diag.inf_detected == False
        assert diag.gradient_overflow_count == 0
        assert diag.boundary_collapse_warnings == 0


class TestPhase8ConfigSerialization:
    """Test configuration serialization."""
    
    def test_config_to_dict(self):
        """Should be able to convert config to dict."""
        config = Phase8Config(d_model=128, use_bk_hyperbolic=True)
        config_dict = asdict(config)
        
        assert isinstance(config_dict, dict)
        assert 'd_model' in config_dict
        assert config_dict['d_model'] == 128
        assert 'use_bk_hyperbolic' in config_dict
        assert config_dict['use_bk_hyperbolic'] == True
    
    def test_config_from_dict(self):
        """Should be able to create config from dict."""
        config_dict = {
            'd_model': 256,
            'n_layers': 12,
            'use_bk_hyperbolic': True,
            'enable_entailment_cones': True
        }
        
        config = Phase8Config(**config_dict)
        
        assert config.d_model == 256
        assert config.n_layers == 12
        assert config.use_bk_hyperbolic == True
        assert config.enable_entailment_cones == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
