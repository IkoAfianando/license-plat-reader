import { NextResponse } from 'next/server';

const API_BASE_URL = process.env.LPR_API_URL || 'http://localhost:8000';

export async function GET() {
  try {
    // Check if the FastAPI server is running
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { 
          status: 'unhealthy',
          message: `FastAPI server returned ${response.status}`,
          api_url: API_BASE_URL
        },
        { status: 503 }
      );
    }

    const data = await response.json();
    return NextResponse.json({
      status: 'healthy',
      fastapi: data,
      api_url: API_BASE_URL,
      nextjs: {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      }
    });
  } catch (error: Error | unknown) {
    console.error('Health Check Error:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { 
        status: 'unhealthy',
        message: `Cannot connect to FastAPI server: ${message}`,
        api_url: API_BASE_URL
      },
      { status: 503 }
    );
  }
}