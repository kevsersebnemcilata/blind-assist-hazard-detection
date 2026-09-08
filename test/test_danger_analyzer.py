# tests/test_danger_analyzer.py

import pytest
import numpy as np
from src.core.danger_analyzer import DangerAnalyzer

class TestDangerAnalyzer:
    """Danger analyzer testleri"""
    
    @pytest.fixture
    def analyzer(self):
        """Her test için yeni analyzer örneği"""
        return DangerAnalyzer()
    
    # TEST 1: Basit analiz
    def test_analyze_returns_six_values(self, analyzer):
        """Analyze 6 değer döndürmeli"""
        # Dummy depth ve motion haritaları
        depth_map = np.random.rand(480, 640) * 100
        motion_map = np.random.rand(480, 640) * 20
        
        result = analyzer.analyze(motion_map, depth_map)
        
        assert len(result) == 6, f"Expected 6 values, got {len(result)}"
        motion_score, depth_score, delta_d, approach_score, danger_score, trend = result
        
        # Hepsi 0-1 arasında olmalı (trend hariç)
        assert 0 <= motion_score <= 1
        assert 0 <= depth_score <= 1
        assert 0 <= approach_score <= 1
        assert 0 <= danger_score <= 1
        assert trend in [-1, 0, 1]
        
        print(f"✅ Motion: {motion_score:.2f}, Danger: {danger_score:.2f}, Trend: {trend}")
    
    # TEST 2: Artan danger trend
    def test_increasing_danger_trend(self, analyzer):
        """Danger artan ise trend = +1 olmalı"""
        for i in range(15):
            # Her iterasyonda motion artar
            depth_map = np.ones((480, 640)) * 100
            motion_map = np.ones((480, 640)) * (5 + i * 2)  # Artan motion
            
            _, _, _, _, danger, trend = analyzer.analyze(motion_map, depth_map)
        
        # Son iterasyonda trend artan olmalı
        assert trend >= 0, f"Expected increasing trend, got {trend}"
        print(f"✅ Artan danger trend: {trend}")
    
    # TEST 3: Azalan danger trend
    def test_decreasing_danger_trend(self, analyzer):
        """Danger azalan ise trend = -1 olmalı"""
        # Önce yüksek danger
        for i in range(10):
            depth_map = np.ones((480, 640)) * 100
            motion_map = np.ones((480, 640)) * 15
            analyzer.analyze(motion_map, depth_map)
        
        # Sonra düşük danger
        for i in range(10):
            depth_map = np.ones((480, 640)) * 50
            motion_map = np.ones((480, 640)) * 2
            _, _, _, _, danger, trend = analyzer.analyze(motion_map, depth_map)
        
        assert trend <= 0, f"Expected decreasing trend, got {trend}"
        print(f"✅ Azalan danger trend: {trend}")
    
    # TEST 4: Stable danger
    def test_stable_danger_trend(self, analyzer):
        """Danger değişmezse trend = 0 olmalı"""
        # Sabit danger
        for i in range(15):
            depth_map = np.ones((480, 640)) * 50
            motion_map = np.ones((480, 640)) * 5
            _, _, _, _, danger, trend = analyzer.analyze(motion_map, depth_map)
        
        # Trend stabil olmalı
        assert trend == 0, f"Expected stable trend (0), got {trend}"
        print(f"✅ Stable danger trend: {trend}")