/* The shared slash-command parser (§35).
 *
 * A command is described by a JSON Schema — exactly as a tool is. The palette,
 * the parameter hints, enum completion, validation and the final parse all read
 * that one schema, so adding a command is adding a schema and nothing here
 * changes.
 *
 * Grammar: `/name pos1 "pos two" key:value key:"value with spaces"`
 * Positional args fill required parameters in declared order; `key:value` wins
 * over position. This file is the reference implementation for the corpus that
 * the daemon-side parser must also satisfy. */

export function tokenize(rest) {
  const out = [];
  let i = 0;
  while (i < rest.length) {
    while (i < rest.length && /\s/.test(rest[i])) i++;
    if (i >= rest.length) break;
    let key = null;
    const km = /^([A-Za-z_]\w*):/.exec(rest.slice(i));
    if (km) { key = km[1]; i += km[0].length; }
    let val = '';
    if (rest[i] === '"' || rest[i] === "'") {
      const q = rest[i++];
      while (i < rest.length && rest[i] !== q) { val += rest[i++]; }
      i++; // closing quote (tolerated if missing — the line is still being typed)
    } else {
      while (i < rest.length && !/\s/.test(rest[i])) val += rest[i++];
    }
    out.push({ key, val });
  }
  return out;
}

function coerce(schema, raw) {
  const t = schema && schema.type;
  if (t === 'integer' || t === 'number') {
    if (raw === '') return { error: 'expected a number' };
    const n = Number(raw);
    if (!Number.isFinite(n)) return { error: `"${raw}" is not a number` };
    if (t === 'integer' && !Number.isInteger(n)) return { error: 'expected a whole number' };
    if (schema.minimum != null && n < schema.minimum) return { error: `min ${schema.minimum}` };
    if (schema.maximum != null && n > schema.maximum) return { error: `max ${schema.maximum}` };
    return { value: n };
  }
  if (t === 'boolean') return { value: !/^(false|no|0|off)$/i.test(raw) };
  if (Array.isArray(schema.enum) && raw !== '') {
    const hit = schema.enum.find((e) => String(e).toLowerCase() === raw.toLowerCase());
    if (!hit) return { error: `one of: ${schema.enum.join(', ')}` };
    return { value: hit };
  }
  return { value: raw };
}

/** Rank commands for the palette. Name prefix beats name substring beats summary. */
export function filterCommands(commands, term) {
  const t = term.toLowerCase();
  if (!t) return commands.slice();
  return commands
    .map((c) => {
      const n = c.name.toLowerCase();
      let s = -1;
      if (n === t) s = 0;
      else if (n.startsWith(t)) s = 1;
      else if ((c.aliases || []).some((a) => a.toLowerCase().startsWith(t))) s = 2;
      else if (n.includes(t)) s = 3;
      else if ((c.summary || '').toLowerCase().includes(t)) s = 4;
      return { c, s };
    })
    .filter((x) => x.s >= 0)
    .sort((a, b) => a.s - b.s || a.c.name.localeCompare(b.c.name))
    .map((x) => x.c);
}

/**
 * Parse a composer line against the command catalog.
 * Returns `{ isCommand, term, command, args, fields, errors, complete }`.
 * `fields` drives the parameter view; it is derived entirely from the schema.
 */
export function parseLine(line, commands) {
  if (!line.startsWith('/')) return { isCommand: false };

  const sp = line.indexOf(' ');
  const name = (sp < 0 ? line.slice(1) : line.slice(1, sp)).trim();
  const rest = sp < 0 ? '' : line.slice(sp + 1);

  const exact = commands.find(
    (c) => c.name === name || (c.aliases || []).includes(name)
  );
  if (!exact) return { isCommand: true, term: name, command: null, matches: filterCommands(commands, name) };

  const props = (exact.parameters && exact.parameters.properties) || {};
  const order = Object.keys(props);
  const required = exact.required || [];
  const toks = tokenize(rest);

  const args = {};
  const errors = [];
  const seen = new Set();

  // keyed first, so a later positional cannot silently overwrite an explicit key
  for (const tk of toks) {
    if (!tk.key) continue;
    if (!props[tk.key]) { errors.push({ field: tk.key, message: 'unknown parameter' }); continue; }
    const r = coerce(props[tk.key], tk.val);
    if (r.error) errors.push({ field: tk.key, message: r.error });
    else { args[tk.key] = r.value; seen.add(tk.key); }
  }
  // positionals fill required-then-declared order, skipping anything keyed
  const slots = [...required, ...order.filter((k) => !required.includes(k))].filter((k) => !seen.has(k));
  let si = 0;
  for (const tk of toks) {
    if (tk.key) continue;
    const k = slots[si++];
    if (!k) { errors.push({ field: '_', message: 'too many arguments' }); break; }
    const r = coerce(props[k], tk.val);
    if (r.error) errors.push({ field: k, message: r.error });
    else args[k] = r.value;
  }
  // defaults
  for (const k of order) if (args[k] === undefined && props[k].default !== undefined) args[k] = props[k].default;

  const missing = required.filter((k) => args[k] === undefined || args[k] === '');
  const fields = order.map((k) => ({
    name: k,
    schema: props[k],
    required: required.includes(k),
    value: args[k],
    state: errors.some((e) => e.field === k) ? 'error'
      : args[k] !== undefined && args[k] !== '' ? 'satisfied'
        : required.includes(k) ? 'missing' : 'optional',
    error: (errors.find((e) => e.field === k) || {}).message,
  }));

  return {
    isCommand: true, term: name, command: exact, args, fields, errors, missing,
    matches: [exact],
    complete: missing.length === 0 && errors.length === 0,
  };
}

/** Render a parsed command back to a canonical line (used by palette accept). */
export function toLine(command, args) {
  const parts = ['/' + command.name];
  for (const [k, v] of Object.entries(args || {})) {
    const s = String(v);
    parts.push(`${k}:${/\s/.test(s) ? JSON.stringify(s) : s}`);
  }
  return parts.join(' ');
}
