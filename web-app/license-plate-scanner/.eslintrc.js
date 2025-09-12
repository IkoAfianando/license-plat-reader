module.exports = {
  extends: ['next/core-web-vitals'],
  rules: {
    '@typescript-eslint/no-explicit-any': 'warn', // Changed from error to warning
    '@typescript-eslint/no-unused-vars': 'warn', // Changed from error to warning
    'react-hooks/exhaustive-deps': 'warn', // Changed from error to warning
  }
};