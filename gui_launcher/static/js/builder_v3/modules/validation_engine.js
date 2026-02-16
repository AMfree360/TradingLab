export default class ValidationEngine {
  constructor(schema) {
    this.schema = schema || {};
  }

  _typeOf(val) {
    if (val === null) return 'null';
    if (Array.isArray(val)) return 'array';
    return typeof val;
  }

  _matchType(expected, val) {
    const actual = this._typeOf(val);
    if (Array.isArray(expected)) return expected.includes(actual);
    if (expected === 'integer') return actual === 'number' && Number.isInteger(val);
    if (expected === 'number') return actual === 'number';
    return expected === actual;
  }

  _pushError(errors, path, message) {
    errors.push({ path: path || '', message });
  }

  _validateSchema(schema, value, path, errors) {
    if (!schema) return;

    const t = schema.type;
    if (t && value !== undefined) {
      if (!this._matchType(t, value)) {
        this._pushError(errors, path, `Expected type ${Array.isArray(t) ? t.join('|') : t}, got ${this._typeOf(value)}`);
        // If type mismatches, stop further checks for this node
        return;
      }
    }

    // Strings: pattern, minLength, maxLength
    if ((t === 'string' || (Array.isArray(t) && t.includes('string'))) && typeof value === 'string') {
      if (schema.minLength !== undefined && value.length < schema.minLength) {
        this._pushError(errors, path, `String too short (len ${value.length} < ${schema.minLength})`);
      }
      if (schema.maxLength !== undefined && value.length > schema.maxLength) {
        this._pushError(errors, path, `String too long (len ${value.length} > ${schema.maxLength})`);
      }
      if (schema.pattern) {
        try {
          const re = new RegExp(schema.pattern);
          if (!re.test(value)) this._pushError(errors, path, `Value does not match pattern ${schema.pattern}`);
        } catch (e) {
          // invalid pattern - skip
        }
      }
    }

    // Numbers: minimum, maximum
    if ((t === 'number' || t === 'integer' || (Array.isArray(t) && (t.includes('number') || t.includes('integer')))) && typeof value === 'number') {
      if (schema.minimum !== undefined && value < schema.minimum) this._pushError(errors, path, `Value ${value} < minimum ${schema.minimum}`);
      if (schema.maximum !== undefined && value > schema.maximum) this._pushError(errors, path, `Value ${value} > maximum ${schema.maximum}`);
    }

    // Arrays: minItems, maxItems, items
    if ((t === 'array' || (Array.isArray(t) && t.includes('array'))) && Array.isArray(value)) {
      if (schema.minItems !== undefined && value.length < schema.minItems) this._pushError(errors, path, `Array has ${value.length} items < minItems ${schema.minItems}`);
      if (schema.maxItems !== undefined && value.length > schema.maxItems) this._pushError(errors, path, `Array has ${value.length} items > maxItems ${schema.maxItems}`);
      if (schema.items) {
        for (let i = 0; i < value.length; i++) {
          this._validateSchema(schema.items, value[i], path ? `${path}.${i}` : `${i}`, errors);
        }
      }
    }

    // Objects: properties, required, additionalProperties
    if ((t === 'object' || (!t && typeof value === 'object' && !Array.isArray(value) && value !== null)) && typeof value === 'object' && value !== null) {
      const props = schema.properties || {};
      const req = Array.isArray(schema.required) ? schema.required : [];
      for (const r of req) {
        if (!(r in value)) this._pushError(errors, path ? `${path}.${r}` : r, 'Missing required property');
      }
      for (const [k, v] of Object.entries(value)) {
        const propSchema = props[k];
        if (propSchema) {
          this._validateSchema(propSchema, v, path ? `${path}.${k}` : k, errors);
        } else if (schema.additionalProperties === false) {
          this._pushError(errors, path ? `${path}.${k}` : k, 'Additional property not allowed');
        }
      }
    }
  }

  validate(payload) {
    const errors = [];
    const schema = this.schema || {};

    if (!payload || typeof payload !== 'object') {
      return { valid: false, errors: [{ path: '', message: 'Payload must be an object' }] };
    }

    // top-level required
    const topReq = Array.isArray(schema.required) ? schema.required : [];
    for (const r of topReq) {
      if (!(r in payload)) errors.push({ path: r, message: 'Missing required field' });
    }

    // Validate each declared property
    const props = schema.properties || {};
    for (const [k, propSchema] of Object.entries(props)) {
      const val = payload[k];
      if (val === undefined) continue;
      this._validateSchema(propSchema, val, k, errors);
    }

    return { valid: errors.length === 0, errors };
  }
}
