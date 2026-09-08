# tests/test_risk_score.py

import pytest
from src.utils.risk import compute_risk_score

class TestComputeRiskScore:
    """Risk skoru hesaplama testleri"""
    
    # TEST 1: Araç (high risk) testi
    def test_vehicle_high_risk(self):
        """Araba + yakın mesafe + hareketli = yüksek risk"""
        score = compute_risk_score(
            class_id=2,              # Car (araç)
            depth_norm=0.9,          # Çok yakın
            conf=0.95,               # Yüksek güven
            motion_score=0.8,        # Hareketli
            approach_score=0.7,      # Yaklaşıyor
            bbox_area_norm=0.3       # Geniş
        )
        
        # Kontroller
        assert score > 0.5, f"Beklenen: >0.5, Aldı: {score}"
        assert score <= 1.0, "Risk skoru 1.0'dan büyük olamaz"
        print(f"✅ Araç risk skoru: {score:.2f}")
    
    # TEST 2: Trafik ışığı (low risk) testi
    def test_traffic_light_low_risk(self):
        """Trafik ışığı = düşük risk"""
        score = compute_risk_score(
            class_id=9,              # Traffic light
            depth_norm=0.5,          # Orta mesafe
            conf=0.8,
            motion_score=0.0,        # Hareketsiz
            approach_score=0.0
        )
        
        assert score < 0.3, f"Beklenen: <0.3, Aldı: {score}"
        print(f"✅ Trafik ışığı risk skoru: {score:.2f}")
    
    # TEST 3: Bilinmeyen sınıf testi
    def test_unknown_class_returns_zero(self):
        """Bilinmeyen sınıf = 0 risk"""
        score = compute_risk_score(
            class_id=999,            # Bilinmeyen
            depth_norm=0.9,
            conf=0.95
        )
        
        assert score == 0.0, "Bilinmeyen sınıf 0 olmalı"
        print(f"✅ Bilinmeyen sınıf: {score}")
    
    # TEST 4: Kişi (medium risk) testi
    def test_person_medium_risk(self):
        """Kişi + yakın + yaklaşıyor = medium-high risk"""
        score = compute_risk_score(
            class_id=0,              # Person (kişi)
            depth_norm=0.85,
            conf=0.9,
            motion_score=0.6,
            approach_score=0.8
        )
        
        assert 0.3 < score < 0.8, f"Expected medium risk, got {score}"
        print(f"✅ Kişi risk skoru: {score:.2f}")
    
    # TEST 5: Tüm parametreler sıfır (minimum)
    def test_zero_parameters_minimum_score(self):
        """Tüm sıfır = minimum risk"""
        score = compute_risk_score(
            class_id=0,
            depth_norm=0.0,
            conf=0.0,
            motion_score=0.0,
            approach_score=0.0,
            bbox_area_norm=0.0
        )
        
        assert score >= 0.0, "Risk negatif olamaz"
        print(f"✅ Minimum risk: {score:.2f}")