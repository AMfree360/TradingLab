describe('Builder V3 save conflict flow', () => {
  beforeEach(() => {
    cy.visit('/builder-v3');
  });

  it('handles 409 conflict and overwrites when user chooses overwrite', () => {
    // stub metadata and validate endpoints
    cy.intercept('GET', '/api/builder_v3/metadata', { ok: true, data: {} });
    cy.intercept('POST', '/api/builder_v3/validate', { ok: true, valid: true, errors: [] });

    // First save responds 409 with suggested name, second call succeeds
    let first = true;
    cy.intercept('POST', '/api/builder_v3/save', (req) => {
      if (first) {
        first = false;
        req.reply({ statusCode: 409, body: { ok: false, message: 'Conflict', suggested_name: 'name_v2' } });
      } else {
        req.reply({ statusCode: 200, body: { ok: true, data: { next: '/create-strategy-guided/review' } } });
      }
    }).as('saveReq');

    // Click save button — modal should appear
    cy.get('#v3-save-btn').click();

    // Wait for modal to appear
    cy.get('#v3-modal-root').should('be.visible');
    // Click Overwrite
    cy.get('#v3-modal-overwrite').click();

    // Confirm second save was sent and status updated
    cy.wait('@saveReq');
    cy.get('#v3-save-status', { timeout: 5000 }).should('contain', 'Saved');
  });
});
describe('Builder V3 save conflict flow', () => {
  beforeEach(() => {
    cy.visit('/builder-v3');
  });

  it('handles 409 conflict and overwrites when user chooses overwrite', () => {
    // stub metadata and validate endpoints
    cy.intercept('GET', '/api/builder_v3/metadata', { ok: true, data: {} });
    cy.intercept('POST', '/api/builder_v3/validate', { ok: true, valid: true, errors: [] });

    // First save responds 409 with suggested name
    let first = true;
    cy.intercept('POST', '/api/builder_v3/save', (req) => {
      if (first) {
        first = false;
        req.reply({ statusCode: 409, body: { ok: false, message: 'Conflict', suggested_name: 'name_v2' } });
      } else {
        req.reply({ statusCode: 200, body: { ok: true, data: { next: '/create-strategy-guided/review' } } });
      }
    }).as('saveReq');

    // Click save button — modal should appear
    cy.get('#v3-save-btn').click();

    // Wait for modal to appear
    cy.get('#v3-modal-root').should('be.visible');
    // Click Overwrite
    cy.get('#v3-modal-overwrite').click();

    // Confirm second save was sent and status updated
    cy.wait('@saveReq');
    cy.get('#v3-save-status', { timeout: 5000 }).should('contain', 'Saved');
  });
});
