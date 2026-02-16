const { loadComponent } = require('./component_helper');
const path = require('path');

describe('ValidationEngine', () => {
  const dirname = path.resolve(__dirname, '..');
  let ValidationEngine;
  beforeAll(() => {
    ValidationEngine = loadComponent('modules/validation_engine.js', dirname);
  });

  test('string min/max and pattern validation', () => {
    const schema = {
      type: 'object',
      required: ['name'],
      properties: {
        name: { type: 'string', minLength: 3, maxLength: 5, pattern: '^[a-z]+$' }
      }
    };
    const ve = new ValidationEngine(schema);
    expect(ve.validate({}).valid).toBe(false);
    expect(ve.validate({ name: 'ab' }).errors.some(e => e.message.includes('too short') || e.message.includes('String too short'))).toBe(true);
    expect(ve.validate({ name: 'abcdef' }).errors.some(e => e.message.includes('too long') || e.message.includes('String too long'))).toBe(true);
    expect(ve.validate({ name: 'Abc' }).errors.some(e => e.message.includes('pattern') || e.message.includes('does not match'))).toBe(true);
    expect(ve.validate({ name: 'abc' }).valid).toBe(true);
  });

  test('numeric minimum/maximum checks', () => {
    const schema = { type: 'object', properties: { v: { type: 'number', minimum: 0, maximum: 10 } } };
    const ve = new ValidationEngine(schema);
    const resLow = ve.validate({ v: -1 });
    expect(resLow.valid).toBe(false);
    expect(resLow.errors.some(e => e.message.includes('< minimum'))).toBe(true);
    const resHigh = ve.validate({ v: 11 });
    expect(resHigh.valid).toBe(false);
    expect(resHigh.errors.some(e => e.message.includes('> maximum'))).toBe(true);
    expect(ve.validate({ v: 5 }).valid).toBe(true);
  });

  test('array minItems and nested item validation', () => {
    const schema = {
      type: 'object',
      properties: {
        arr: {
          type: 'array',
          minItems: 2,
          items: {
            type: 'object',
            required: ['x'],
            properties: { x: { type: 'number' } }
          }
        }
      }
    };
    const ve = new ValidationEngine(schema);
    const r1 = ve.validate({ arr: [{ x: 1 }] });
    expect(r1.valid).toBe(false);
    expect(r1.errors.some(e => e.message.includes('minItems'))).toBe(true);

    const r2 = ve.validate({ arr: [{ x: 1 }, {}] });
    expect(r2.valid).toBe(false);
    expect(r2.errors.some(e => e.path === 'arr.1.x' || e.message.includes('Missing required'))).toBe(true);

    const r3 = ve.validate({ arr: [{ x: 1 }, { x: 2 }] });
    expect(r3.valid).toBe(true);
  });

  test('object additionalProperties and nested props', () => {
    const schema = {
      type: 'object',
      properties: {
        obj: {
          type: 'object',
          properties: { a: { type: 'number' } },
          additionalProperties: false
        }
      }
    };
    const ve = new ValidationEngine(schema);
    const r = ve.validate({ obj: { a: 1, b: 2 } });
    expect(r.valid).toBe(false);
    expect(r.errors.some(e => e.path === 'obj.b' || e.message.includes('Additional'))).toBe(true);
  });
});
