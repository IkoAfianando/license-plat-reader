"""
Test script for Roboflow License Plate Detection and OCR
This enhanced script performs two steps:
1. Detects license plate bounding boxes using Roboflow.
2. Extracts text from those boxes using EasyOCR.
"""

import os
from dotenv import load_dotenv
import json
import time
import argparse
from pathlib import Path
import numpy as np # Diperlukan untuk konversi gambar ke format yang bisa dibaca OCR

# --- BLOK BARU: Import untuk OCR dan Gambar ---
try:
    from PIL import Image
    print("✅ Pillow (PIL) library imported successfully")
except ImportError:
    print("❌ Pillow library not found. Install with: pip install Pillow")
    exit(1)

try:
    import easyocr
    print("✅ EasyOCR library imported successfully")
except ImportError:
    print("❌ EasyOCR library not found. Install with: pip install easyocr")
    exit(1)
# --- AKHIR BLOK BARU ---


# Load environment variables from .env if present
load_dotenv()

class RoboflowTester:
    def __init__(self, api_key=None, workspace=None, project_slug=None, version=None):
        """Initialize Roboflow API connection and OCR reader"""
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY')
        if not self.api_key:
            print("❌ API key required. Set ROBOFLOW_API_KEY environment variable")
            exit(1)
        
        # --- BARIS BARU: Inisialisasi OCR Reader ---
        # 'en' untuk bahasa Inggris. Untuk plat Indonesia, kombinasi ini seringkali cukup baik.
        # Anda bisa menambahkan 'id' jika perlu: ['id', 'en']
        print("\n⏳ Initializing OCR engine (EasyOCR)... This may take a moment.")
        self.ocr_reader = easyocr.Reader(['en']) 
        print("✅ OCR engine initialized.")
        # --- AKHIR BARIS BARU ---
            
        try:
            from roboflow import Roboflow
            self.rf = Roboflow(api_key=self.api_key)
            resolved_workspace = workspace or os.getenv('ROBOFLOW_WORKSPACE', 'roboflow-universe-projects')
            resolved_project = project_slug or os.getenv('ROBOFLOW_PROJECT', 'license-plate-recognition-rxg4e')
            resolved_version = version or int(os.getenv('ROBOFLOW_VERSION', '4'))

            self.project = self.rf.workspace(resolved_workspace).project(resolved_project)
            self.model = self.project.version(resolved_version).model
            print("✅ Connected to Roboflow API")
            print(f"🔗 Model: workspace='{resolved_workspace}', project='{resolved_project}', version={resolved_version}")
        except Exception as e:
            print(f"❌ Failed to connect to Roboflow: {e}")
            exit(1)

    def test_single_image(self, image_path, output_dir=None, confidence=40, overlap=30):
        """Test detection on single image and perform OCR on results"""
        print(f"\n🔍 Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
            
        try:
            # --- Langkah 1: Deteksi Objek dengan Roboflow ---
            start_time = time.time()
            result = self.model.predict(image_path, confidence=confidence, overlap=overlap)
            inference_time = time.time() - start_time
            
            detections = result.json()
            
            print(f"⏱️  Roboflow inference time: {inference_time:.3f} seconds")
            print(f"📊 Detections found: {len(detections.get('predictions', []))}")
            
            # --- Langkah 2: Proses OCR pada setiap deteksi ---
            if detections.get('predictions'):
                try:
                    # Buka gambar asli untuk di-crop
                    original_image = Image.open(image_path).convert("RGB")
                    
                    for i, detection in enumerate(detections['predictions']):
                        # Roboflow memberikan koordinat pusat (x, y). Kita perlu mengubahnya menjadi
                        # koordinat pojok (kiri, atas, kanan, bawah) untuk cropping.
                        x_center = detection['x']
                        y_center = detection['y']
                        width = detection['width']
                        height = detection['height']

                        left = x_center - (width / 2)
                        top = y_center - (height / 2)
                        right = x_center + (width / 2)
                        bottom = y_center + (height / 2)
                        
                        # Crop gambar plat nomor
                        cropped_plate_img = original_image.crop((left, top, right, bottom))
                        
                        # Konversi gambar PIL ke array NumPy yang bisa dibaca EasyOCR
                        cropped_plate_np = np.array(cropped_plate_img)

                        # Jalankan OCR pada gambar yang sudah di-crop
                        ocr_start_time = time.time()
                        ocr_results = self.ocr_reader.readtext(cropped_plate_np)
                        ocr_inference_time = time.time() - ocr_start_time
                        
                        # Gabungkan hasil teks OCR (jika ada) menjadi satu string
                        recognized_text = " ".join([res[1] for res in ocr_results]).strip()
                        
                        # Tambahkan hasil OCR ke dalam dictionary deteksi
                        detection['ocr_text'] = recognized_text
                        detection['ocr_confidence'] = [res[2] for res in ocr_results] # simpan confidence per kata

                        print(f"  [Detection {i+1}]")
                        print(f"    Bounding box: (x={detection['x']:.1f}, y={detection['y']:.1f}, w={detection['width']:.1f}, h={detection['height']:.1f})")
                        if recognized_text:
                            print(f"    ✅ OCR Result: '{recognized_text}' (in {ocr_inference_time:.3f}s)")
                        else:
                            print(f"    ⚠️ OCR: No text found.")
                            
                except Exception as e:
                    print(f"❌ Failed during OCR processing: {e}")

            # Simpan hasil (sekarang JSON akan berisi teks OCR)
            base_name = Path(image_path).stem
            output_path = Path(output_dir) if output_dir else Path(".")
            os.makedirs(output_path, exist_ok=True)
            
            image_out = str(output_path / f"{base_name}_result.jpg")
            json_out = str(output_path / f"{base_name}_result.json")
            
            result.save(image_out)
            with open(json_out, 'w') as jf:
                json.dump(detections, jf, indent=2)
            print(f"💾 Saved annotated image: {image_out}")
            print(f"💾 Saved enhanced JSON (with OCR): {json_out}")
            
            return {
                'inference_time': inference_time,
                'detections': len(detections.get('predictions', [])),
                'raw_result': detections,
                'image_output': image_out,
                'json_output': json_out
            }
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return None
            
    # Sisa fungsi lainnya (performance_benchmark, create_demo_report, dll) tidak saya ubah
    # karena fokusnya adalah pada test_single_image.
    # Anda bisa mengadaptasi logika OCR ke fungsi benchmark jika diperlukan.

# ... (Sisa kode dari `parse_args` dan `main` bisa tetap sama) ...
# ... (Salin semua fungsi dari script asli Anda dari sini ke bawah) ...

from typing import Optional

def resolve_input_dir(cli_input: Optional[str], repo_root: Path) -> str:
    """Resolve input directory from CLI/env/defaults with sensible fallbacks."""
    # 1) CLI takes precedence
    if cli_input:
        return cli_input
    # 2) Environment variable
    env_dir = os.getenv('IMAGE_INPUT_DIR')
    if env_dir:
        return env_dir
    # 3) Repo-local default (requested: data/images)
    repo_default = repo_root / "data" / "images"
    if repo_default.exists():
        return str(repo_default)
    # 4) External dataset path (common on this machine)
    external_default = Path(os.path.expanduser("~/COMPANY/ieko-media/dataset/images/in"))
    return str(external_default)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Roboflow license plate detection and OCR on a batch of images.")
    parser.add_argument("--input", dest="input_dir", help="Input images directory")
    parser.add_argument("--output", dest="output_dir", help="Output base directory (defaults to outputs/roboflow_batch_<ts>)")
    parser.add_argument("--max", dest="max_images", type=int, default=1, help="Max number of images to process (default: 1)") # Diubah ke 1 untuk tes
    parser.add_argument("--confidence", type=int, default=40, help="Confidence threshold (default: 40)")
    parser.add_argument("--overlap", type=int, default=30, help="Overlap threshold (default: 30)")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively in input directory")
    parser.add_argument("--workspace", dest="workspace", help="Roboflow workspace slug (default: roboflow-universe-projects or ROBOFLOW_WORKSPACE)")
    parser.add_argument("--project", dest="project", help="Roboflow project slug (default: license-plate-recognition-rxg4e or ROBOFLOW_PROJECT)")
    parser.add_argument("--version", dest="version", type=int, help="Roboflow version number (default: 4 or ROBOFLOW_VERSION)")
    return parser.parse_args()

def main():
    """Main test execution"""
    print("🚀 Roboflow License Plate Detection & OCR Test")
    print("=" * 50)
    args = parse_args()

    # Initialize tester
    tester = RoboflowTester(
        workspace=args.workspace,
        project_slug=args.project,
        version=args.version,
    )

    # Sisa logika main tetap sama, hanya pastikan direktori input dan output benar
    repo_root = Path(__file__).resolve().parent
    input_dir = resolve_input_dir(args.input_dir, repo_root)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = repo_root / "outputs" / f"roboflow_batch_{timestamp}"
    output_images_dir = output_base / "images"

    os.makedirs(output_images_dir, exist_ok=True)

    patterns = ("*.jpg", "*.jpeg", "*.png")
    test_images = []
    input_path = Path(input_dir)
    searcher = input_path.rglob if args.recursive else input_path.glob
    if input_path.is_dir():
        for pat in patterns:
            test_images.extend(sorted(str(p) for p in searcher(pat)))
    elif input_path.is_file(): # Tambahan: Handle jika input adalah file tunggal
        test_images = [str(input_path)]
    else:
        print(f"⚠️  Input path not found: {input_dir}")

    test_images = test_images[: max(0, args.max_images)]

    if not test_images:
        print("⚠️  No images found. Add images and retry.")
        print(f"   Looked at: {input_dir}")
        return

    print(f"📁 Using input path: {input_dir}")
    print(f"📁 Output directory: {output_base}")
    print(f"🖼️  Images to process: {len(test_images)} (max={args.max_images})")

    results = []
    
    for image_path in test_images:
        result = tester.test_single_image(
            image_path,
            output_dir=str(output_images_dir),
            confidence=args.confidence,
            overlap=args.overlap,
        )
        if result:
            results.append(result)
    
    # ... Sisa fungsi main bisa di-skip untuk tes tunggal ini ...
    if results:
        print(f"\n✅ Testing complete! Review results in {output_images_dir}")
    else:
        print("\n⚠️  No valid results were generated.")

if __name__ == "__main__":
    main()