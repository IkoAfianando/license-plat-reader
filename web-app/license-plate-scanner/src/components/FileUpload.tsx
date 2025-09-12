'use client';

import { useState } from 'react';
import axios from 'axios';

interface DetectionResponse {
  success: boolean;
  detections: Array<{
    bbox: [number, number, number, number];
    confidence: number;
    class: string;
    text?: string;
  }>;
  processing_time: number;
  image_size: [number, number];
  model_info: {
    detector: string;
    ocr_engine?: string;
  };
  message?: string;
}

export default function FileUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<'yolo' | 'roboflow'>('yolo');
  const [useNextJSAPI, setUseNextJSAPI] = useState(true);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [debugInfo, setDebugInfo] = useState<string>('');

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
      setResults(null);
      setDebugInfo('');
    }
  };

  const uploadAndDetect = async () => {
    if (!selectedFile) {
      setError('Please select an image file first');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    setDebugInfo('');

    const debugLog: string[] = [];
    debugLog.push(`🔍 UPLOAD DEBUG SESSION - ${new Date().toISOString()}`);
    debugLog.push(`📁 File: ${selectedFile.name} (${selectedFile.size} bytes, ${selectedFile.type})`);
    debugLog.push(`🤖 Model: ${selectedModel}`);
    debugLog.push(`🌐 API Mode: ${useNextJSAPI ? 'Next.js API Proxy' : 'Direct FastAPI'}`);
    debugLog.push(`🔗 API URL: ${apiUrl}`);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('confidence', '0.5');
      formData.append('use_roboflow', selectedModel === 'roboflow' ? 'true' : 'false');
      formData.append('extract_text', 'true');
      formData.append('return_image', 'false');

      debugLog.push('\n📦 FormData prepared:');
      for (const [key, value] of formData.entries()) {
        if (value instanceof File) {
          debugLog.push(`  ${key}: File(${value.name}, ${value.size} bytes, ${value.type})`);
        } else {
          debugLog.push(`  ${key}: ${value}`);
        }
      }

      const endpoint = useNextJSAPI ? '/api/detect' : `${apiUrl}/detect/image`;
      debugLog.push(`\n🎯 Target endpoint: ${endpoint}`);

      let apiResponse;
      
      try {
        debugLog.push('\n⏳ Sending request...');
        apiResponse = await axios.post<DetectionResponse>(
          endpoint,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            timeout: 30000,
          }
        );
        debugLog.push('✅ Primary API request successful');
        debugLog.push(`📊 Response status: ${apiResponse.status}`);
      } catch (primaryError) {
        debugLog.push('⚠️ Primary endpoint failed, trying fallback if available');
        
        if (useNextJSAPI) {
          debugLog.push('🔄 Trying direct FastAPI fallback...');
          try {
            apiResponse = await axios.post<DetectionResponse>(
              `${apiUrl}/detect/image`,
              formData,
              {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
                timeout: 30000,
              }
            );
            debugLog.push('✅ Fallback API request successful');
            debugLog.push(`📊 Response status: ${apiResponse.status}`);
          } catch (fallbackError) {
            debugLog.push('❌ Both primary and fallback failed');
            debugLog.push(`Primary error: ${primaryError}`);
            debugLog.push(`Fallback error: ${fallbackError}`);
            throw primaryError;
          }
        } else {
          throw primaryError;
        }
      }

      if (apiResponse?.data) {
        setResults(apiResponse.data);
        debugLog.push('\n🎉 Detection completed successfully');
        debugLog.push(`⏱️ Processing time: ${apiResponse.data.processing_time}s`);
        debugLog.push(`🔍 Detections found: ${apiResponse.data.detections.length}`);
        debugLog.push(`📏 Image size: ${apiResponse.data.image_size[0]}x${apiResponse.data.image_size[1]}`);
      }

    } catch (err: Error | unknown) {
      console.error('Upload detection error:', err);
      debugLog.push('\n❌ ERROR OCCURRED:');
      
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { status?: number; data?: { detail?: string; error?: string } } };
        debugLog.push(`HTTP Status: ${axiosErr.response?.status}`);
        debugLog.push(`Response data: ${JSON.stringify(axiosErr.response?.data, null, 2)}`);
        
        if (axiosErr.response?.status === 500) {
          setError(`Server Error (500): ${axiosErr.response?.data?.detail || axiosErr.response?.data?.error || 'Internal server error. Check server logs.'}`);
        } else if (axiosErr.response?.status === 404) {
          setError('API endpoint not found. Make sure the server is running correctly.');
        } else if (axiosErr.response?.data?.detail) {
          setError(`API Error: ${axiosErr.response.data.detail}`);
        } else {
          setError(`HTTP ${axiosErr.response?.status}: ${axiosErr.response?.data?.error || 'Unknown API error'}`);
        }
      } else if (err instanceof Error) {
        debugLog.push(`Error message: ${err.message}`);
        debugLog.push(`Error stack: ${err.stack}`);
        setError(`Network/Request Error: ${err.message}`);
      } else {
        debugLog.push(`Unknown error: ${err}`);
        setError('Unknown error occurred during upload');
      }
    } finally {
      setDebugInfo(debugLog.join('\n'));
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4">
        📁 File Upload Test (Debug Mode)
      </h2>
      
      {/* File Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Image File:
        </label>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-full file:border-0
                     file:text-sm file:font-semibold
                     file:bg-blue-50 file:text-blue-700
                     hover:file:bg-blue-100"
        />
        {selectedFile && (
          <p className="text-sm text-gray-600 mt-1">
            Selected: {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
          </p>
        )}
      </div>

      {/* Model Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Detection Model:
        </label>
        <div className="flex space-x-4">
          <label className="flex items-center">
            <input
              type="radio"
              value="yolo"
              checked={selectedModel === 'yolo'}
              onChange={(e) => setSelectedModel(e.target.value as 'yolo')}
              className="mr-2"
            />
            🔥 YOLO Local
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              value="roboflow"
              checked={selectedModel === 'roboflow'}
              onChange={(e) => setSelectedModel(e.target.value as 'roboflow')}
              className="mr-2"
            />
            ☁️ Roboflow API
          </label>
        </div>
      </div>

      {/* API Mode Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          API Route:
        </label>
        <div className="flex space-x-4">
          <label className="flex items-center">
            <input
              type="radio"
              checked={useNextJSAPI}
              onChange={() => setUseNextJSAPI(true)}
              className="mr-2"
            />
            🌐 Next.js API (/api/detect)
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              checked={!useNextJSAPI}
              onChange={() => setUseNextJSAPI(false)}
              className="mr-2"
            />
            🚀 Direct FastAPI
          </label>
        </div>
      </div>

      {/* Upload Button */}
      <button
        onClick={uploadAndDetect}
        disabled={!selectedFile || loading}
        className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 
                   text-white font-semibold py-3 px-4 rounded-lg transition-colors"
      >
        {loading ? '⏳ Processing...' : '🔍 Upload & Detect'}
      </button>

      {/* Debug Information */}
      {debugInfo && (
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-800 mb-2">🔧 Debug Information</h3>
          <pre className="bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto whitespace-pre-wrap">
            {debugInfo}
          </pre>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <h3 className="text-red-800 font-medium">❌ Error</h3>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <h3 className="text-green-800 font-medium mb-2">✅ Detection Results</h3>
          <div className="text-sm text-green-700 space-y-1">
            <p><strong>Status:</strong> {results.success ? 'Success' : 'Failed'}</p>
            <p><strong>Processing Time:</strong> {results.processing_time.toFixed(2)}s</p>
            <p><strong>Detections Found:</strong> {results.detections.length}</p>
            <p><strong>Image Size:</strong> {results.image_size[0]}x{results.image_size[1]}</p>
            <p><strong>Model:</strong> {results.model_info.detector}</p>
            {results.model_info.ocr_engine && (
              <p><strong>OCR Engine:</strong> {results.model_info.ocr_engine}</p>
            )}
          </div>
          
          {results.detections.length > 0 && (
            <div className="mt-3">
              <h4 className="font-medium text-green-800">Detected License Plates:</h4>
              <div className="space-y-2 mt-2">
                {results.detections.map((detection, index) => (
                  <div key={index} className="bg-white p-2 rounded border">
                    <p><strong>Confidence:</strong> {(detection.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Position:</strong> [{detection.bbox.map(n => n.toFixed(0)).join(', ')}]</p>
                    {detection.text && (
                      <p><strong>Text:</strong> <span className="font-mono bg-gray-100 px-1 rounded">{detection.text}</span></p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}