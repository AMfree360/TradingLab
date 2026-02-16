describe('Builder V3 smoke', () => {
  it('loads the Builder V3 page and shows UI skeleton', () => {
    cy.visit('/builder-v3');
    cy.get('#builder-v3-root').should('exist');
    cy.get('#v3-preview-json').should('exist');
  });

  it('shows validation errors when saving empty payload', () => {
    cy.visit('/builder-v3');
    cy.get('#v3-save-btn').click();
    // should reveal validation error area
    cy.get('#v3-validation-errors').should('be.visible');
  });
});
