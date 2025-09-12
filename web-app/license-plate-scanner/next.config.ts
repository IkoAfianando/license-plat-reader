import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable experimental features for better mobile support
  // experimental: {
  //   optimizeCss: true,
  // },
  
  // Configure headers for better mobile experience
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          // Allow camera access for PWA
          {
            key: 'Permissions-Policy',
            value: 'camera=*, microphone=*',
          },
        ],
      },
    ];
  },

  // Configure for development with ngrok
  async rewrites() {
    return [
      // Proxy API calls to FastAPI server when in development
      {
        source: '/api/lpr/:path*',
        destination: `${process.env.LPR_API_URL || 'http://localhost:8000'}/:path*`,
      },
    ];
  },

  // Optimize images and enable PWA features
  images: {
    formats: ['image/avif', 'image/webp'],
    minimumCacheTTL: 60,
  },

  // Enable compression
  compress: true,

  // Configure for production deployment
  trailingSlash: false,
  
  // Allow external API calls
  async redirects() {
    return [];
  },
};

export default nextConfig;
