# tests/test_approach_tracker.py

import pytest
from src.utils.approach_tracker import ApproachTracker

class TestApproachTracker:
    """Yaklaşma takibi testleri"""
    
    @pytest.fixture
    def tracker(self):
        return ApproachTracker(history_len=8, growth_threshold=0.05)
    
    # TEST 1: Yaklaşan nesne
    def test_approaching_object_high_score(self, tracker):
        """Büyüyen bbox = yaklaşıyor"""
        detections = [
            {"track_id": 1, "bbox": (100, 100, 150, 150)},  # Alan: 2500
        ]
        
        # Aynı nesne gittikçe büyüyor
        for size in [100, 120, 140, 160, 180, 200, 220, 240]:
            detections = [
                {"track_id": 1, "bbox": (100, 100, 100+size, 100+size)}
            ]
            tracker.update(detections)
        
        score = tracker.score(1)
        assert score > 0.5, f"Yaklaşan nesne yüksek skor olmalı, aldı: {score}"
        print(f"✅ Yaklaşan nesne skoru: {score:.2f}")
    
    # TEST 2: Uzaklaşan nesne
    def test_receding_object_zero_score(self, tracker):
        """Küçülen bbox = uzaklaşıyor"""
        # Gittikçe küçülen nesne
        for size in [240, 220, 200, 180, 160, 140, 120, 100]:
            detections = [
                {"track_id": 1, "bbox": (100, 100, 100+size, 100+size)}
            ]
            tracker.update(detections)
        
        score = tracker.score(1)
        assert score <= 0.2, f"Uzaklaşan nesne düşük skor olmalı, aldı: {score}"
        print(f"✅ Uzaklaşan nesne skoru: {score:.2f}")
    
    # TEST 3: Sabit alan
    def test_constant_size_zero_score(self, tracker):
        """Boyutu değişmeyen nesne = 0 skor"""
        for _ in range(8):
            detections = [
                {"track_id": 1, "bbox": (100, 100, 150, 150)}  # Sabit
            ]
            tracker.update(detections)
        
        score = tracker.score(1)
        assert score == 0.0, f"Sabit nesne skor 0 olmalı, aldı: {score}"
        print(f"✅ Sabit nesne skoru: {score:.2f}")
    
    # TEST 4: Görülmeyen nesne
    def test_unseen_object_cleanup(self, tracker):
        """Görülmeyen nesne silinmeli"""
        # Önce tespit et
        detections = [{"track_id": 1, "bbox": (100, 100, 150, 150)}]
        tracker.update(detections)
        
        # Sonra update etme (görülmedi)
        tracker.update([])
        
        # Skor 0 olmalı (nesne silindi)
        score = tracker.score(1)
        assert score == 0.0, "Silinmiş nesne skor 0 olmalı"
        print(f"✅ Silinmiş nesne temizlendi")