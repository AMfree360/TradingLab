module.exports = {
  testEnvironment: 'jsdom',
  testTimeout: 10000,
  roots: ['<rootDir>/gui_launcher/static/js/__tests__'],
  testMatch: ['**/*.test.js'],
  testPathIgnorePatterns: [
    '/node_modules/',
    '/\.venv/',
    '/data/',
    '/reports/',
    '/validation_state/',
    '/\.validation_state/',
  ],
  transform: {},
};
