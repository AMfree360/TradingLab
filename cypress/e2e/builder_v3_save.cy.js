describe('Builder V3 save conflict flow', () => {
  it('handles 409 conflict and overwrites when user chooses overwrite', () => {
    // stub metadata and validate endpoints before visiting the page
      // DEPRECATED: Builder V3 tests disabled (feature removed).
      // See branch `remove/builder-v3` for context.
      // cy.intercept('GET', '/api/builder_v3/metadata', { ok: true, data: {} });
      // cy.intercept('POST', '/api/builder_v3/validate', { ok: true, valid: true, errors: [] });
    // stub common external CDNs to avoid page load hangs during CI
    cy.intercept('GET', 'https://cdn.plot.ly/**', { statusCode: 200, body: 'window.Plotly = { react: function(){} };' });
    cy.intercept('GET', 'https://cdn.jsdelivr.net/**', { statusCode: 200, body: '' });
    cy.intercept('GET', 'https://unpkg.com/**', { statusCode: 200, body: '' });
    cy.intercept('GET', 'https://fonts.googleapis.com/**', { statusCode: 200, body: '' });
    cy.intercept('GET', 'https://fonts.gstatic.com/**', { statusCode: 200, body: '' });
    cy.intercept('GET', 'https://www.googletagmanager.com/**', { statusCode: 200, body: '' });
    // As a last resort, intercept any absolute external URL (non-localhost) and return empty
    cy.intercept({ url: /^https?:\/\/(?!localhost|127\.0\.0\.1).*/ }, { statusCode: 200, body: '' });

    // First save responds 409 with suggested name, second call succeeds
    let first = true;
      // cy.intercept('POST', '/api/builder_v3/save', (req) => {
      //   if (first) {
      //     first = false;
      //     req.reply({ statusCode: 409, body: { ok: false, message: 'Conflict', suggested_name: 'name_v2' } });
      //   } else {
      //     req.reply({ statusCode: 200, body: { ok: true, data: { next: '/create-strategy-guided/review' } } });
      //   }
      // }).as('saveReq');

    // Instead of using `cy.visit` (which waits for the native `load` event and
    // can hang in this environment), fetch the minimal page HTML and write it
    // into a blank document. This avoids Cypress waiting on external resource
    // loads while still allowing scripts to execute.
    cy.request({ url: '/builder-v3?cypress=1', failOnStatusCode: false }).then((resp) => {
      expect(resp.status).to.equal(200);
      // Visit the root quickly, then replace the document with the fetched
      // minimal template. This avoids cy.visit waiting on the target page's
      // native `load` event while allowing scripts to execute from the
      // injected HTML.
      cy.visit('/');
      cy.window().then((win) => {
        try {
          win.document.open();
          win.document.write(resp.body);
          win.document.close();
        } catch (e) {}
      });
    });

    // Wait for the app to hydrate and show the Save button
    cy.get('#v3-save-btn', { timeout: 60000 }).should('be.visible').click();

    // Wait for modal to appear
    cy.get('#v3-modal-root').should('be.visible');
    // Click Overwrite
    cy.get('#v3-modal-overwrite').click();

    // Confirm second save was sent and status updated
    cy.wait('@saveReq');
    cy.get('#v3-save-status', { timeout: 5000 }).should('contain', 'Saved');
  });
});
