"use client";

import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

interface Detection {
  confidence: number;
  bbox: number[];
  text?: string;
  text_confidence?: number;
}

interface DetectionResponse {
  success: boolean;
  detections: Detection[];
  processing_time: number;
  image_size: number[];
  model_info: {
    detector: string;
    ocr_engine: string;
  };
  message?: string;
}

export default function CameraCapture() {
  const webcamRef = useRef<Webcam>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useNextJSAPI, setUseNextJSAPI] = useState(true);
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [selectedModel, setSelectedModel] = useState<'yolo' | 'roboflow'>('yolo');
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [liveResults, setLiveResults] = useState<DetectionResponse[]>([]);
  const [detectionInterval, setDetectionInterval] = useState<NodeJS.Timeout | null>(null);
  
  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: { exact: "environment" }, // Use back camera on mobile
  };

  const capture = useCallback(async (isLiveCapture = false) => {
    console.log('📸 Capture started, live mode:', isLiveCapture);
    
    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) {
      if (!isLiveCapture) {
        setError('Failed to capture image. Make sure camera permission is granted.');
      }
      console.error('❌ Failed to get screenshot from webcam');
      return null;
    }

    console.log('📷 Screenshot captured, size:', imageSrc.length, 'characters');

    if (!isLiveCapture) {
      setIsLoading(true);
      setError(null);
      setResults(null);
    }

    try {
      // Convert base64 to blob using more reliable method
      console.log('🔄 Converting base64 to blob...');
      console.log('📋 Base64 data preview:', imageSrc.substring(0, 50) + '...');
      
      // Extract base64 data (remove data:image/xxx;base64, prefix)
      const base64Data = imageSrc.split(',')[1];
      if (!base64Data) {
        throw new Error('Invalid base64 data from camera');
      }
      
      // Convert base64 to binary
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Create blob with explicit JPEG type
      const blob = new Blob([bytes], { type: 'image/jpeg' });
      console.log('📦 Blob created, size:', blob.size, 'bytes, type:', blob.type);
      
      // Verify blob is valid
      if (!blob || blob.size === 0) {
        throw new Error('Invalid blob: empty or null blob created from camera');
      }
      
      // Create form data (exactly like FileUpload)
      const formData = new FormData();
      formData.append('file', blob, 'capture.jpg');
      formData.append('confidence', '0.5');
      formData.append('use_roboflow', selectedModel === 'roboflow' ? 'true' : 'false');
      formData.append('extract_text', 'true');
      formData.append('return_image', 'false');
      
      // Debug: Log form data contents
      console.log('📝 FormData prepared:');
      for (const [key, value] of formData.entries()) {
        if (value instanceof File) {
          console.log(`  ${key}: File(${value.name}, ${value.size} bytes, ${value.type})`);
        } else {
          console.log(`  ${key}: ${value}`);
        }
      }

      // Send to API - use Next.js API route or direct FastAPI
      const endpoint = useNextJSAPI ? '/api/detect' : `${apiUrl}/detect/image`;
      console.log('🌐 Sending request to:', endpoint, 'with model:', selectedModel);

      let apiResponse;
      
      try {
        // Try primary endpoint first
        console.log('⏳ Sending request...');
        apiResponse = await axios.post<DetectionResponse>(
          endpoint,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            timeout: 30000, // 30 second timeout
          }
        );
        console.log('✅ Primary API request successful');
        console.log('📊 Response status:', apiResponse.status);
      } catch (primaryError) {
        console.warn('⚠️ Primary endpoint failed, trying fallback if available');
        console.error('Primary error details:', primaryError);
        
        // If Next.js API fails, try direct FastAPI as fallback
        if (useNextJSAPI && !isLiveCapture) {
          console.log('🔄 Trying direct FastAPI fallback...');
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
            console.log('✅ Fallback API request successful');
            console.log('📊 Fallback response status:', apiResponse.status);
            
            // Show warning that fallback was used
            if (!isLiveCapture) {
              setError('Warning: Using direct FastAPI connection (Next.js API failed)');
            }
          } catch (fallbackError) {
            console.error('❌ Both primary and fallback failed');
            console.error('Primary error:', primaryError);
            console.error('Fallback error:', fallbackError);
            throw primaryError; // Throw the original error
          }
        } else {
          throw primaryError;
        }
      }

      console.log('📊 Detection results:', {
        detections: apiResponse.data.detections.length,
        processingTime: apiResponse.data.processing_time,
        model: apiResponse.data.model_info.detector
      });

      if (isLiveCapture) {
        // Add to live results if there are detections
        if (apiResponse.data.detections.length > 0) {
          setLiveResults(prev => {
            const newResults = [apiResponse.data, ...prev.slice(0, 9)]; // Keep last 10 results
            return newResults;
          });
        }
      } else {
        setResults(apiResponse.data);
      }
      
      return apiResponse.data;
    } catch (err: Error | unknown) {
      console.error('Detection error:', err);
      if (!isLiveCapture) {
        if (err && typeof err === 'object' && 'response' in err) {
          const axiosErr = err as { response?: { status?: number; data?: { detail?: string; error?: string } } };
          if (axiosErr.response?.status === 500) {
            setError(`Server Error (500): ${axiosErr.response?.data?.detail || axiosErr.response?.data?.error || 'Internal server error. Check server logs.'}`);
          } else if (axiosErr.response?.status === 404) {
            setError('API endpoint not found. Make sure the server is running correctly.');
          } else if (axiosErr.response?.data?.detail) {
            setError(`API Error: ${axiosErr.response.data.detail}`);
          } else if (axiosErr.response?.data?.error) {
            setError(`Server Error: ${axiosErr.response.data.error}`);
          } else {
            setError(`HTTP ${axiosErr.response?.status || 'Unknown'}: Request failed. Check server connection.`);
          }
        } else if (err instanceof Error) {
          setError(`Network Error: ${err.message}`);
        } else {
          setError('Unknown error occurred. Check camera permissions and network connection.');
        }
      }
      return null;
    } finally {
      if (!isLiveCapture) {
        setIsLoading(false);
      }
    }
  }, [apiUrl, selectedModel, useNextJSAPI]);

  // Live detection function
  const toggleLiveMode = useCallback(() => {
    if (isLiveMode) {
      // Stop live mode
      if (detectionInterval) {
        clearInterval(detectionInterval);
        setDetectionInterval(null);
      }
      setIsLiveMode(false);
      setLiveResults([]);
    } else {
      // Start live mode
      setIsLiveMode(true);
      setLiveResults([]);
      setError(null);
      
      // Start continuous detection every 2 seconds
      const interval = setInterval(() => {
        capture(true);
      }, 2000);
      
      setDetectionInterval(interval);
    }
  }, [isLiveMode, detectionInterval, capture]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      if (detectionInterval) {
        clearInterval(detectionInterval);
      }
    };
  }, [detectionInterval]);

  return (
    <div className="max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      {/* Settings */}
      <div className="mb-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            API Connection
          </label>
          <div className="flex space-x-4 mb-3">
            <label className="flex items-center">
              <input
                type="radio"
                name="apiMode"
                checked={useNextJSAPI}
                onChange={() => setUseNextJSAPI(true)}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Use Next.js API (Recommended)</span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="apiMode"
                checked={!useNextJSAPI}
                onChange={() => setUseNextJSAPI(false)}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Direct FastAPI</span>
            </label>
          </div>
          {!useNextJSAPI && (
            <input
              type="url"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              placeholder="http://localhost:8000"
            />
          )}
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Detection Model
          </label>
          <div className="flex space-x-4">
            <label className="flex items-center">
              <input
                type="radio"
                name="model"
                value="yolo"
                checked={selectedModel === 'yolo'}
                onChange={() => setSelectedModel('yolo')}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">YOLO Local</span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="model"
                value="roboflow"
                checked={selectedModel === 'roboflow'}
                onChange={() => setSelectedModel('roboflow')}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Roboflow API</span>
            </label>
          </div>
        </div>
      </div>

      {/* Camera */}
      <div className="mb-6">
        <div className="relative bg-black rounded-lg overflow-hidden">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            className="w-full max-h-96 object-cover"
          />
          
          {/* Overlay for license plate frame and live mode indicator */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="w-full h-full flex items-center justify-center">
              <div className="border-2 border-white border-dashed rounded-md w-64 h-16 flex items-center justify-center">
                <span className="text-white text-sm opacity-75">
                  {isLiveMode ? "🔴 LIVE SCANNING..." : "License Plate Area"}
                </span>
              </div>
            </div>
            
            {/* Live mode status indicator */}
            {isLiveMode && (
              <div className="absolute top-2 left-2 bg-red-600 text-white px-2 py-1 rounded-md text-xs font-bold animate-pulse">
                🔴 LIVE CCTV MODE
              </div>
            )}
          </div>
        </div>
        
        {/* Control Buttons */}
        <div className="mt-4 flex flex-col sm:flex-row gap-3 justify-center">
          {/* Single Capture Button */}
          <button
            onClick={() => capture(false)}
            disabled={isLoading || isLiveMode}
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Detecting...
              </>
            ) : (
              <>📸 Capture & Scan</>
            )}
          </button>

          {/* Live CCTV Mode Button */}
          <button
            onClick={toggleLiveMode}
            disabled={isLoading}
            className={`inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed ${
              isLiveMode
                ? 'text-white bg-red-600 hover:bg-red-700 focus:ring-red-500'
                : 'text-white bg-green-600 hover:bg-green-700 focus:ring-green-500'
            }`}
          >
            {isLiveMode ? (
              <>
                <span className="animate-pulse mr-2">🔴</span>
                Stop CCTV Mode
              </>
            ) : (
              <>📹 Start CCTV Mode</>
            )}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-800 rounded-md p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800 dark:text-red-200">
                Detection Error
              </h3>
              <div className="mt-2 text-sm text-red-700 dark:text-red-300">
                <p>{error}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <div className="bg-green-50 dark:bg-green-900 border border-green-200 dark:border-green-800 rounded-md p-4">
          <h3 className="text-sm font-medium text-green-800 dark:text-green-200 mb-3">
            Detection Results
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-sm text-green-700 dark:text-green-300">
                <strong>Model:</strong> {results.model_info.detector}
              </p>
              <p className="text-sm text-green-700 dark:text-green-300">
                <strong>Processing Time:</strong> {results.processing_time.toFixed(3)}s
              </p>
              <p className="text-sm text-green-700 dark:text-green-300">
                <strong>Detections:</strong> {results.detections.length}
              </p>
            </div>
            <div>
              <p className="text-sm text-green-700 dark:text-green-300">
                <strong>Image Size:</strong> {results.image_size.join('x')}
              </p>
              <p className="text-sm text-green-700 dark:text-green-300">
                <strong>OCR Engine:</strong> {results.model_info.ocr_engine}
              </p>
            </div>
          </div>

          {/* License Plates Found */}
          {results.detections.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-green-800 dark:text-green-200">
                License Plates Found:
              </h4>
              {results.detections.map((detection, index) => (
                <div key={index} className="bg-white dark:bg-gray-800 rounded-md p-3 border border-green-200 dark:border-green-700">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    <div>
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        <strong>Confidence:</strong> {(detection.confidence * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        <strong>Position:</strong> [{detection.bbox.map(n => n.toFixed(0)).join(', ')}]
                      </p>
                    </div>
                    {detection.text && (
                      <div>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          <strong>License Plate Text:</strong>
                        </p>
                        <p className="text-lg font-mono font-bold text-blue-600 dark:text-blue-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                          {detection.text}
                        </p>
                        {detection.text_confidence && (
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            OCR Confidence: {(detection.text_confidence * 100).toFixed(1)}%
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {results.detections.length === 0 && (
            <p className="text-sm text-green-700 dark:text-green-300">
              No license plates detected in this image.
            </p>
          )}
        </div>
      )}

      {/* Live CCTV Results Display */}
      {isLiveMode && (
        <div className="mt-6 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-800 dark:text-gray-200">
              🔴 Live Detection Feed
            </h3>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Scanning every 2 seconds | Found: {liveResults.length} detections
            </div>
          </div>
          
          {liveResults.length === 0 ? (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              <div className="animate-pulse">🔍 Watching for license plates...</div>
            </div>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {liveResults.map((result, index) => (
                <div key={index} className="bg-white dark:bg-gray-800 rounded-md p-3 border border-gray-200 dark:border-gray-600">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Detection #{liveResults.length - index}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {new Date().toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className="space-y-2">
                    {result.detections.map((detection, detIndex) => (
                      <div key={detIndex} className="flex flex-col sm:flex-row sm:items-center gap-2 p-2 bg-blue-50 dark:bg-blue-900 rounded">
                        <div className="flex-1">
                          {detection.text ? (
                            <div>
                              <span className="text-lg font-mono font-bold text-blue-600 dark:text-blue-400">
                                {detection.text}
                              </span>
                              <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                                ({(detection.confidence * 100).toFixed(1)}%)
                              </span>
                            </div>
                          ) : (
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                              Plate detected ({(detection.confidence * 100).toFixed(1)}%)
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {result.processing_time.toFixed(2)}s
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
          
          <div className="mt-3 text-center">
            <button
              onClick={() => setLiveResults([])}
              className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              🗑️ Clear History
            </button>
          </div>
        </div>
      )}
    </div>
  );
}