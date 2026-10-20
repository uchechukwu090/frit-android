import sharp from "sharp";

const svg = Buffer.from(
  '<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">' +
  '<text x="20" y="100" font-family="Helvetica, Arial, sans-serif" font-size="48" font-weight="800" fill="#101010">Hello</text>' +
  '</svg>'
);

const base = sharp({ create: { width: 400, height: 200, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } } });
const out = await base.composite([{ input: svg, top: 0, left: 0 }]).png().toBuffer();
const stats = await sharp(out).greyscale().stats();
console.log("min", stats.channels[0].min, "max", stats.channels[0].max, "mean", stats.channels[0].mean.toFixed(2));
console.log("has dark pixels:", stats.channels[0].min < 100 ? "YES" : "NO (text did not render)");
