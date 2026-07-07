import os
import io
import sys
import asyncio
import json
from PIL import Image
from fastapi import UploadFile

# Add current folder to search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import predict, health

def safe_print(data):
    print(json.dumps(data, ensure_ascii=True, indent=2))

async def test_backend():
    print("==================================================")
    print("RUNNING FARMER LEAF DOCTOR BACKEND UNIT TESTS")
    print("==================================================")
    
    # Test 1: Health check
    print("\n[Test 1] Querying health check...")
    health_res = await health()
    print("Response:")
    safe_print(health_res)
    assert health_res["status"] == "healthy"
    print("=> Health check test passed!")
    
    # Test 2: Successful mock prediction
    print("\n[Test 2] Testing prediction with valid image (Tomato Early Blight)...")
    
    # Generate 160x160 red RGB image in memory
    img = Image.new('RGB', (160, 160), color='green')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Instantiate FastAPI UploadFile
    upload_file = UploadFile(
        file=img_byte_arr,
        filename="test_tomato_early_blight.jpg",
        headers={"content-type": "image/jpeg"}
    )
    
    # Run predict route
    result = await predict(upload_file)
    print("Response:")
    safe_print(result)
    
    # Assert correctness
    assert result["success"] is True
    assert result["crop"] == "Tomato"
    assert result["disease"] == "Early Blight"
    assert result["confidence"] >= 0.60
    assert len(result["remedy"]) > 0
    assert len(result["remedy_ta"]) > 0
    print("=> Valid image prediction test passed!")

    # Test 3: Low-confidence detection
    print("\n[Test 3] Testing prediction under low-confidence simulation...")
    
    # Generate mock image with 'bad_quality' in filename
    img_byte_arr.seek(0)
    upload_file_low = UploadFile(
        file=img_byte_arr,
        filename="bad_quality_leaf.jpg",
        headers={"content-type": "image/jpeg"}
    )
    
    result_low = await predict(upload_file_low)
    print("Response structure:")
    safe_print(result_low)
    assert result_low["success"] is False
    assert result_low["low_confidence"] is True
    assert "Unable to confidently identify" in result_low["message"]
    print("=> Low-confidence detection test passed!")

    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(test_backend())

