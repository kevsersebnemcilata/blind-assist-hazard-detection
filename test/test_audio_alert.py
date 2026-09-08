# tests/test_audio_alert.py

import pytest
from src.utils.audio_alert import _build_message, _direction

class TestAudioAlert:
    """Audio alert message testleri"""
    
    # TEST 1: Yön belirleme
    def test_direction_left(self):
        """Sol taraf tespiti"""
        assert _direction(0.2) == "solda"
    
    def test_direction_center(self):
        """Orta taraf tespiti"""
        assert _direction(0.5) == "önde"
    
    def test_direction_right(self):
        """Sağ taraf tespiti"""
        assert _direction(0.8) == "sağda"
    
    # TEST 2: Mesaj oluşturma
    def test_build_message_single_object(self):
        """Tek nesne için mesaj"""
        detections = [
            {
                "class_name": "car",
                "risk_score": 0.8,
                "cx_norm": 0.2,
                "approach_score": 0.9
            }
        ]
        
        message = _build_message(detections)
        assert "araç" in message.lower()
        assert "solda" in message.lower()
        assert "yaklaşıyor" in message.lower()
        print(f"✅ Tek nesne mesajı: {message}")
    
    # TEST 3: Çoklu nesne
    def test_build_message_multiple_objects(self):
        """Çoklu nesne mesajı"""
        detections = [
            {"class_name": "car", "risk_score": 0.9, "cx_norm": 0.5, "approach_score": 0.1},
            {"class_name": "person", "risk_score": 0.7, "cx_norm": 0.5, "approach_score": 0.1},
        ]
        
        message = _build_message(detections)
        assert "araç" in message.lower()
        print(f"✅ Çoklu nesne mesajı: {message}")
    
    # TEST 4: Bilinmeyen sınıf
    def test_build_message_unknown_class(self):
        """Bilinmeyen sınıf fallback"""
        detections = [
            {"class_name": "unknown_thing", "risk_score": 0.8, "cx_norm": 0.5}
        ]
        
        message = _build_message(detections)
        assert message == "Dikkat!"  # Default fallback
        print(f"✅ Bilinmeyen sınıf fallback: {message}")