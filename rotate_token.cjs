// Rotate AUTH_TOKEN in server/.env, and drop the unused MISTRAL_API_KEY.
// Usage: node rotate_token.cjs <new-token>
const fs = require('fs');
const path = require('path');

const envPath = path.join(__dirname, '.env');
const newToken = process.argv[2];
if (!newToken || newToken.length < 32) {
  console.error('usage: node rotate_token.cjs <new-token>');
  process.exit(2);
}

const before = fs.readFileSync(envPath, 'utf8');
const out = [];
let sawToken = false;
for (const line of before.split(/\r?\n/)) {
  if (/^\s*AUTH_TOKEN\s*=/.test(line)) {
    out.push(`AUTH_TOKEN=${newToken}`);
    sawToken = true;
  } else if (/^\s*MISTRAL_API_KEY\s*=/.test(line)) {
    // The Mistral key is no longer used (OpenRouter is the provider). Keep the
    // line as an empty, commented placeholder so the env file still documents
    // the optional knob, but never ship the old value again.
    out.push('# MISTRAL_API_KEY removed - not used (OpenRouter is the provider). Set it only if you add a Mistral fallback.');
    out.push('MISTRAL_API_KEY=');
  } else {
    out.push(line);
  }
}
if (!sawToken) {
  console.error('AUTH_TOKEN line not found in .env - nothing changed');
  process.exit(1);
}
fs.writeFileSync(envPath, out.join('\n'), 'utf8');
console.log('AUTH_TOKEN rotated; MISTRAL_API_KEY cleared.');
