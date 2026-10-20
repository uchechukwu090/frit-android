const sharp = require("sharp");
(async () => {
  const svg = `<svg width="300" height="120" xmlns="http://www.w3.org/2000/svg"><rect width="300" height="120" fill="#000"/><text x="20" y="70" font-family="DejaVu Sans, sans-serif" font-size="48" font-weight="700" fill="#fff">HELLO</text></svg>`;
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  const { data, info } = await sharp(buf).raw().toBuffer({ resolveWithObject: true });
  let white = 0;
  for (let i = 0; i < data.length; i += info.channels) { if (data[i] > 200 && data[i + 1] > 200 && data[i + 2] > 200) white++; }
  console.log("channels", info.channels, "whitePixels", white, info.width, info.height);
})().catch(e => { console.error("ERR", e.message); process.exit(1); });
