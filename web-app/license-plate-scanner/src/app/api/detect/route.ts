import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.LPR_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  console.log('🔍 API Route: /api/detect called');
  
  try {
    // Get the form data from the request
    console.log('📥 Getting form data...');
    const formData = await request.formData();
    
    // Log form data for debugging
    console.log('📝 Form data entries:');
    for (const [key, value] of formData.entries()) {
      if (value instanceof File) {
        console.log(`  ${key}: File(${value.name}, ${value.size} bytes, ${value.type})`);
      } else {
        console.log(`  ${key}: ${value}`);
      }
    }
    
    console.log(`🌐 Forwarding request to: ${API_BASE_URL}/detect/image`);
    
    // Forward the request to the FastAPI server
    const response = await fetch(`${API_BASE_URL}/detect/image`, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header, let fetch handle it for FormData
    });

    console.log(`📡 FastAPI response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ FastAPI error response:', errorText);
      return NextResponse.json(
        { 
          error: `FastAPI Error (${response.status})`,
          detail: errorText,
          api_url: API_BASE_URL 
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log('✅ FastAPI response received, forwarding to client');
    return NextResponse.json(data);
  } catch (error: Error | unknown) {
    console.error('❌ API Route Error:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';
    const stack = error instanceof Error ? error.stack : undefined;
    
    return NextResponse.json(
      { 
        error: `Next.js API Route Error: ${message}`,
        details: {
          message,
          stack: stack?.split('\n').slice(0, 5), // First 5 lines of stack
          api_url: API_BASE_URL,
          timestamp: new Date().toISOString()
        }
      },
      { status: 500 }
    );
  }
}