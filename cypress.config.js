const { defineConfig } = require('cypress')

module.exports = defineConfig({
  e2e: {
    baseUrl: 'http://localhost:8000',
    specPattern: 'cypress/e2e/**/*.cy.js',
    supportFile: false,
    video: true,
    screenshotsFolder: 'cypress/screenshots',
    videosFolder: 'cypress/videos',
    pageLoadTimeout: 300000,
    defaultCommandTimeout: 10000,
    requestTimeout: 15000
  }
})

