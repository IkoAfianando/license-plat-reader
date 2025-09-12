"use client";

import { useState } from 'react';
import CameraCapture from '@/components/CameraCapture';
import FileUpload from '@/components/FileUpload';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'camera' | 'upload'>('camera');

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-2">
            License Plate Scanner
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Scan license plates using your camera or upload images
          </p>
        </header>
        
        {/* Tab Navigation */}
        <div className="flex justify-center mb-6">
          <div className="bg-white rounded-lg p-1 shadow-md">
            <button
              onClick={() => setActiveTab('camera')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'camera'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-blue-500'
              }`}
            >
              📸 Camera Capture
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'upload'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-blue-500'
              }`}
            >
              📁 File Upload (Debug)
            </button>
          </div>
        </div>
        
        <main>
          {activeTab === 'camera' ? <CameraCapture /> : <FileUpload />}
        </main>
      </div>
    </div>
  );
}
