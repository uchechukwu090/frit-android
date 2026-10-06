let sharp;
try {
  sharp = (await import("sharp")).default;
} catch {
  sharp = null;
}

function requireSharp() {
  if (!sharp) throw new Error("sharp not installed — run: npm install sharp");
  return sharp;
}

export async function resize(imageBuffer, width, height, fit = "cover") {
  return requireSharp()(imageBuffer).resize(width, height, { fit }).toBuffer();
}

export async function crop(imageBuffer, left, top, width, height) {
  return requireSharp()(imageBuffer).extract({ left, top, width, height }).toBuffer();
}

export async function rotate(imageBuffer, angle) {
  return requireSharp()(imageBuffer).rotate(angle).toBuffer();
}

export async function flip(imageBuffer, direction = "horizontal") {
  return direction === "vertical"
    ? requireSharp()(imageBuffer).flip().toBuffer()
    : requireSharp()(imageBuffer).flop().toBuffer();
}

export async function adjustColor(imageBuffer, { brightness, contrast, saturation } = {}) {
  let pipeline = requireSharp()(imageBuffer);
  const modulate = {};
  if (brightness != null) modulate.brightness = brightness;
  if (saturation != null) modulate.saturation = saturation;
  if (Object.keys(modulate).length) pipeline = pipeline.modulate(modulate);
  if (contrast != null) pipeline = pipeline.linear(contrast, -(128 * contrast) + 128);
  return pipeline.toBuffer();
}

export async function addText(imageBuffer, text, { x = 0, y = 0, fontSize = 32, color = "#FFFFFF", fontFamily = "sans-serif" } = {}) {
  const svg = `<svg width="100%" height="100%">
    <text x="${x}" y="${y}" font-family="${fontFamily}" font-size="${fontSize}" fill="${color}">${text}</text>
  </svg>`;
  return requireSharp()(imageBuffer)
    .composite([{ input: Buffer.from(svg), top: y, left: x }])
    .toBuffer();
}

export async function applyFilter(imageBuffer, filterType) {
  let pipeline = requireSharp()(imageBuffer);
  switch (filterType) {
    case "grayscale": pipeline = pipeline.grayscale(); break;
    case "sepia": pipeline = pipeline.sepia(); break;
    case "blur": pipeline = pipeline.blur(5); break;
    case "sharpen": pipeline = pipeline.sharpen(); break;
    case "negate": pipeline = pipeline.negate(); break;
    case "vignette": {
      const meta = await requireSharp()(imageBuffer).metadata();
      const svg = `<svg width="${meta.width}" height="${meta.height}">
        <rect width="100%" height="100%" fill="url(#v)"/>
        <defs><radialGradient id="v" cx="50%" cy="50%" r="70%">
          <stop offset="60%" stop-color="black" stop-opacity="0"/>
          <stop offset="100%" stop-color="black" stop-opacity="0.5"/>
        </radialGradient></defs>
      </svg>`;
      return requireSharp()(imageBuffer).composite([{ input: Buffer.from(svg) }]).toBuffer();
    }
    default: break;
  }
  return pipeline.toBuffer();
}

export async function composite(layers) {
  const s = requireSharp();
  const base = s(layers[0].input);
  const composites = layers.slice(1).map(l => ({
    input: l.input,
    top: l.top || 0,
    left: l.left || 0,
    blend: l.blend || "over",
    opacity: l.opacity || 1,
  }));
  return base.composite(composites).toBuffer();
}

export async function applyMask(imageBuffer, maskBuffer) {
  const s = requireSharp();
  const meta = await s(imageBuffer).metadata();
  const maskResized = await s(maskBuffer)
    .resize(meta.width, meta.height)
    .greyscale()
    .toBuffer();
  return s(imageBuffer)
    .ensureAlpha()
    .composite([{ input: maskResized, blend: "dest-in" }])
    .png()
    .toBuffer();
}

export async function convertFormat(imageBuffer, format = "png", quality = 90) {
  let pipeline = requireSharp()(imageBuffer);
  switch (format) {
    case "jpeg": case "jpg": pipeline = pipeline.jpeg({ quality }); break;
    case "webp": pipeline = pipeline.webp({ quality }); break;
    case "png": default: pipeline = pipeline.png(); break;
  }
  return pipeline.toBuffer();
}

export async function createThumbnail(imageBuffer, size = 256) {
  return requireSharp()(imageBuffer)
    .resize(size, size, { fit: "cover" })
    .jpeg({ quality: 80 })
    .toBuffer();
}

export async function blendImages(baseBuffer, overlayBuffer, opacity = 0.5, position = { x: 0, y: 0 }) {
  const s = requireSharp();
  const meta = await s(baseBuffer).metadata();
  const overlayResized = await s(overlayBuffer)
    .resize(meta.width, meta.height, { fit: "cover" })
    .ensureAlpha(opacity)
    .toBuffer();
  return s(baseBuffer)
    .composite([{ input: overlayResized, top: position.y, left: position.x }])
    .toBuffer();
}

export async function getImageMetadata(imageBuffer) {
  const meta = await requireSharp()(imageBuffer).metadata();
  return { width: meta.width, height: meta.height, format: meta.format, size: meta.size, channels: meta.channels };
}

export async function applyOperations(imageBuffer, operations) {
  let buffer = imageBuffer;
  for (const op of operations) {
    switch (op.type) {
      case "resize": buffer = await resize(buffer, op.params.width, op.params.height, op.params.fit); break;
      case "crop": buffer = await crop(buffer, op.params.left, op.params.top, op.params.width, op.params.height); break;
      case "rotate": buffer = await rotate(buffer, op.params.angle); break;
      case "flip": buffer = await flip(buffer, op.params.direction); break;
      case "adjust_color": buffer = await adjustColor(buffer, op.params); break;
      case "filter": buffer = await applyFilter(buffer, op.params.filter); break;
      case "add_text": buffer = await addText(buffer, op.params.text, op.params); break;
      case "composite": buffer = await composite([{ input: buffer }, ...op.params.layers]); break;
      case "mask": buffer = await applyMask(buffer, op.params.mask); break;
      case "convert": buffer = await convertFormat(buffer, op.params.format, op.params.quality); break;
      case "blend": buffer = await blendImages(buffer, op.params.overlay, op.params.opacity, op.params.position); break;
      default: break;
    }
  }
  return buffer;
}
