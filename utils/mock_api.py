# Mock API for simulating backend responses
# Contains fake data for /predict, /recommend, and /impact endpoints

import random
from datetime import datetime, timedelta

# Set seed for deterministic results
random.seed(42)

# Mock history data with 5 examples
MOCK_HISTORY = [
    {
        "timestamp": "2025-10-15 08:30:00",
        "crop": "Wheat",
        "disease": "Leaf Rust",
        "confidence": 89.5,
        "pesticide_saved_l": 12.5,
        "water_saved_l": 450.0
    },
    {
        "timestamp": "2025-10-18 14:15:00",
        "crop": "Rice",
        "disease": "Blast Disease",
        "confidence": 93.2,
        "pesticide_saved_l": 15.0,
        "water_saved_l": 520.0
    },
    {
        "timestamp": "2025-10-22 10:45:00",
        "crop": "Tomato",
        "disease": "Early Blight",
        "confidence": 87.8,
        "pesticide_saved_l": 8.5,
        "water_saved_l": 380.0
    },
    {
        "timestamp": "2025-10-25 16:20:00",
        "crop": "Potato",
        "disease": "Late Blight",
        "confidence": 91.0,
        "pesticide_saved_l": 10.0,
        "water_saved_l": 410.0
    },
    {
        "timestamp": "2025-10-28 09:00:00",
        "crop": "Maize",
        "disease": "Northern Corn Leaf Blight",
        "confidence": 85.6,
        "pesticide_saved_l": 11.2,
        "water_saved_l": 395.0
    }
]


# Disease database with detailed information
DISEASE_DATABASE = {
    "Wheat": [
        {
            "name": "Leaf Rust",
            "confidence": 91.4,
            "symptoms": "Orange-brown pustules on leaves, reduced photosynthesis"
        },
        {
            "name": "Powdery Mildew",
            "confidence": 87.2,
            "symptoms": "White powdery coating on leaves and stems"
        }
    ],
    "Rice": [
        {
            "name": "Blast Disease",
            "confidence": 93.2,
            "symptoms": "Diamond-shaped lesions with gray centers on leaves"
        },
        {
            "name": "Brown Spot",
            "confidence": 88.5,
            "symptoms": "Circular brown spots with yellow halos"
        }
    ],
    "Potato": [
        {
            "name": "Late Blight",
            "confidence": 91.0,
            "symptoms": "Dark water-soaked lesions on leaves, white mold underneath"
        },
        {
            "name": "Early Blight",
            "confidence": 86.3,
            "symptoms": "Concentric ring patterns on older leaves"
        }
    ],
    "Tomato": [
        {
            "name": "Early Blight",
            "confidence": 87.8,
            "symptoms": "Brown spots with concentric rings on lower leaves"
        },
        {
            "name": "Leaf Mold",
            "confidence": 89.1,
            "symptoms": "Pale green to yellow spots on upper leaf surface"
        }
    ],
    "Maize": [
        {
            "name": "Northern Corn Leaf Blight",
            "confidence": 85.6,
            "symptoms": "Long grayish-green to tan cigar-shaped lesions"
        },
        {
            "name": "Common Rust",
            "confidence": 90.3,
            "symptoms": "Small circular to elongate reddish-brown pustules"
        }
    ]
}


# Treatment recommendations database
TREATMENT_DATABASE = {
    "Leaf Rust": {
        "chemical_treatment": "Apply Propiconazole 25% EC @ 0.1% or Tebuconazole 25% EC @ 0.1%",
        "organic_treatment": "Neem oil spray (5ml/liter water), Garlic extract spray, Sulfur dust application",
        "dosage": "2-3 sprays at 10-day intervals. First spray at disease appearance",
        "prevention": "Use resistant varieties, crop rotation, remove volunteer plants, maintain proper spacing",
        "yield_estimate": "88%"
    },
    "Powdery Mildew": {
        "chemical_treatment": "Sulfur 80% WP @ 2.5g/liter or Carbendazim 50% WP @ 1g/liter",
        "organic_treatment": "Baking soda solution (1 tbsp/gallon), Milk spray (40% milk, 60% water)",
        "dosage": "Weekly sprays starting from tillering stage, 3-4 applications",
        "prevention": "Balanced fertilization, avoid excess nitrogen, ensure good air circulation",
        "yield_estimate": "92%"
    },
    "Blast Disease": {
        "chemical_treatment": "Tricyclazole 75% WP @ 0.6g/liter or Isoprothiolane 40% EC @ 1.5ml/liter",
        "organic_treatment": "Pseudomonas fluorescens application, Neem cake in soil",
        "dosage": "2-3 sprays at 7-10 day intervals during critical stages",
        "prevention": "Use certified seeds, silicon application, avoid excessive nitrogen",
        "yield_estimate": "85%"
    },
    "Brown Spot": {
        "chemical_treatment": "Mancozeb 75% WP @ 2g/liter or Copper oxychloride 50% WP @ 2.5g/liter",
        "organic_treatment": "Trichoderma seed treatment, vermicompost application",
        "dosage": "2 sprays at 15-day intervals starting from symptom appearance",
        "prevention": "Balanced NPK fertilization, proper water management, seed treatment",
        "yield_estimate": "90%"
    },
    "Late Blight": {
        "chemical_treatment": "Metalaxyl 8% + Mancozeb 64% WP @ 2.5g/liter or Cymoxanil 8% + Mancozeb 64% @ 2g/liter",
        "organic_treatment": "Bordeaux mixture (1%), Copper sulfate spray, destroy infected plants",
        "dosage": "Preventive sprays every 7-10 days during favorable conditions",
        "prevention": "Use certified disease-free seeds, hill up plants, destroy cull piles",
        "yield_estimate": "78%"
    },
    "Early Blight": {
        "chemical_treatment": "Chlorothalonil 75% WP @ 2g/liter or Mancozeb 75% WP @ 2.5g/liter",
        "organic_treatment": "Neem oil (5ml/liter), Bacillus subtilis application, compost tea",
        "dosage": "Spray every 7-14 days starting from first symptoms",
        "prevention": "Crop rotation, mulching, prune lower branches, drip irrigation",
        "yield_estimate": "87%"
    },
    "Leaf Mold": {
        "chemical_treatment": "Chlorothalonil 50% WP @ 2g/liter or Difenoconazole 25% EC @ 0.5ml/liter",
        "organic_treatment": "Baking soda spray, increase ventilation, reduce humidity",
        "dosage": "Apply at first sign, repeat every 7-10 days for 3-4 applications",
        "prevention": "Ensure proper ventilation, avoid overhead watering, resistant varieties",
        "yield_estimate": "91%"
    },
    "Northern Corn Leaf Blight": {
        "chemical_treatment": "Propiconazole 25% EC @ 1ml/liter or Azoxystrobin 23% SC @ 1ml/liter",
        "organic_treatment": "Trichoderma application, crop residue management",
        "dosage": "2 sprays: first at tasseling, second 15 days later",
        "prevention": "Resistant hybrids, crop rotation, bury crop residues",
        "yield_estimate": "86%"
    },
    "Common Rust": {
        "chemical_treatment": "Mancozeb 75% WP @ 2.5g/liter or Propiconazole 25% EC @ 1ml/liter",
        "organic_treatment": "Sulfur dust, neem-based products",
        "dosage": "Apply at first appearance, repeat after 15 days if needed",
        "prevention": "Early planting, resistant varieties, balanced nutrition",
        "yield_estimate": "93%"
    },
    "Healthy": {
        "chemical_treatment": "No treatment required",
        "organic_treatment": "Continue preventive care: neem oil spray monthly, compost application",
        "dosage": "Maintain regular monitoring and preventive schedule",
        "prevention": "Continue good agricultural practices, monitor regularly",
        "yield_estimate": "100%"
    }
}


def predict(image_bytes, crop=None, weather=None):
    """
    Simulate /predict endpoint
    
    Args:
        image_bytes: Image data (can be None for mock)
        crop: Crop type (Wheat, Rice, Potato, Tomato, Maize)
        weather: Optional weather data dict
        
    Returns:
        dict with crop, diseases list, and gradcam placeholder
    """
    # Default to Wheat if no crop specified
    crop = crop or "Wheat"
    
    # Get diseases for the crop
    diseases = DISEASE_DATABASE.get(crop, DISEASE_DATABASE["Wheat"])
    
    # Adjust confidence based on weather if provided
    adjusted_diseases = []
    for disease in diseases:
        disease_copy = disease.copy()
        
        # Weather adjustments (simulate realistic variations)
        if weather:
            if weather.get("humidity", 0) > 80:
                disease_copy["confidence"] += 3.5
            if weather.get("temperature", 25) > 30:
                disease_copy["confidence"] -= 2.0
            if weather.get("rainfall", 0) > 50:
                disease_copy["confidence"] += 2.5
                
        # Cap confidence between 0-100
        disease_copy["confidence"] = max(0, min(100, disease_copy["confidence"]))
        adjusted_diseases.append(disease_copy)
    
    return {
        "crop": crop,
        "diseases": adjusted_diseases,
        "gradcam": None  # Placeholder for heatmap visualization
    }


def recommend(disease_name, crop, weather=None):
    """
    Simulate /recommend endpoint
    
    Args:
        disease_name: Name of the detected disease
        crop: Crop type
        weather: Optional weather data for climate-adjusted advice
        
    Returns:
        dict with treatment recommendations
    """
    # Get base treatment data
    treatment = TREATMENT_DATABASE.get(
        disease_name, 
        TREATMENT_DATABASE["Healthy"]
    ).copy()
    
    treatment["disease"] = disease_name
    treatment["crop"] = crop
    
    # Add climate-adjusted advice
    climate_advice = []
    if weather:
        if weather.get("rainfall", 0) > 50:
            climate_advice.append("⚠️ High rainfall detected: Avoid spraying for 24-48 hours after rain")
        if weather.get("temperature", 25) > 35:
            climate_advice.append("🌡️ High temperature: Apply treatments early morning or late evening")
        if weather.get("humidity", 50) > 80:
            climate_advice.append("💧 High humidity: Disease spread risk is elevated, monitor closely")
        if weather.get("humidity", 50) < 40:
            climate_advice.append("☀️ Low humidity: Ensure adequate irrigation for treatment effectiveness")
    
    treatment["climate_advice"] = climate_advice if climate_advice else ["✅ Current weather conditions are suitable for treatment application"]
    
    return treatment


def impact(history=None):
    """
    Simulate /impact endpoint
    
    Args:
        history: List of historical treatment records (uses MOCK_HISTORY if None)
        
    Returns:
        dict with aggregated environmental impact metrics
    """
    # Use provided history or default mock history
    data = history if history else MOCK_HISTORY
    
    if not data:
        return {
            "pesticide_reduction_pct": 0.0,
            "water_saved_liters": 0.0,
            "yield_preservation_pct": 0.0,
            "timeseries": []
        }
    
    # Calculate aggregated metrics
    total_pesticide = sum(entry.get("pesticide_saved_l", 0) for entry in data)
    total_water = sum(entry.get("water_saved_l", 0) for entry in data)
    avg_confidence = sum(entry.get("confidence", 0) for entry in data) / len(data)
    
    # Pesticide reduction percentage (baseline: 100L per season)
    baseline_pesticide = 100.0
    pesticide_reduction_pct = (total_pesticide / baseline_pesticide) * 100
    
    # Yield preservation based on early detection confidence
    yield_preservation_pct = min(95.0, 70.0 + (avg_confidence * 0.25))
    
    # Generate timeseries data
    timeseries = []
    cumulative_pesticide = 0
    cumulative_water = 0
    
    for entry in data:
        cumulative_pesticide += entry.get("pesticide_saved_l", 0)
        cumulative_water += entry.get("water_saved_l", 0)
        
        timeseries.append({
            "date": entry.get("timestamp", "2025-01-01"),
            "pesticide_reduction": round(cumulative_pesticide, 2),
            "water_saved": round(cumulative_water, 2)
        })
    
    return {
        "pesticide_reduction_pct": round(pesticide_reduction_pct, 1),
        "water_saved_liters": round(total_water, 1),
        "yield_preservation_pct": round(yield_preservation_pct, 1),
        "timeseries": timeseries
    }
