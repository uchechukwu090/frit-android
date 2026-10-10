export const TEMPLATES = [
  { id: "tpl_tiktok_001", name: "TikTok Vertical Post", category: "social_post", platform: "tiktok", aspect_ratio: "9:16", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 80 }, { type: "text", x: 5, y: 82, w: 90, h: 15 }] } },
  { id: "tpl_ig_001", name: "Instagram Square Post", category: "social_post", platform: "instagram", aspect_ratio: "1:1", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 100 }] } },
  { id: "tpl_ig_story", name: "Instagram Story", category: "social_post", platform: "instagram", aspect_ratio: "9:16", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 100 }, { type: "text", x: 10, y: 85, w: 80, h: 10 }] } },
  { id: "tpl_yt_thumb", name: "YouTube Thumbnail", category: "thumbnail", platform: "youtube", aspect_ratio: "16:9", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 100 }, { type: "text", x: 5, y: 40, w: 90, h: 20 }] } },
  { id: "tpl_yt_banner", name: "YouTube Banner", category: "thumbnail", platform: "youtube", aspect_ratio: "16:9", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 100 }, { type: "text", x: 30, y: 30, w: 40, h: 40 }] } },
  { id: "tpl_twitter_post", name: "Twitter/X Post", category: "social_post", platform: "twitter", aspect_ratio: "16:9", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 60 }, { type: "text", x: 5, y: 65, w: 90, h: 30 }] } },
  { id: "tpl_fb_cover", name: "Facebook Cover", category: "social_post", platform: "facebook", aspect_ratio: "16:9", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 100 }] } },
  { id: "tpl_ad_banner", name: "Ad Banner", category: "ad", platform: "multi", aspect_ratio: "16:9", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 60, h: 100 }, { type: "text", x: 65, y: 20, w: 30, h: 60 }] } },
  { id: "tpl_reel_001", name: "Reel Cover", category: "reel", platform: "instagram", aspect_ratio: "9:16", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 100 }, { type: "text", x: 10, y: 70, w: 80, h: 20 }] } },
  { id: "tpl_poster_001", name: "Event Poster", category: "poster", platform: "multi", aspect_ratio: "9:16", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 60 }, { type: "text", x: 10, y: 65, w: 80, h: 30 }] } },
  { id: "tpl_youtube_short", name: "YouTube Short", category: "reel", platform: "youtube", aspect_ratio: "9:16", layout_data: { zones: [{ type: "image", x: 0, y: 0, w: 100, h: 100 }] } },
];

export function getTemplates(category = null, platform = null) {
  let filtered = TEMPLATES;
  if (category) filtered = filtered.filter(t => t.category === category);
  if (platform) filtered = filtered.filter(t => t.platform === platform || t.platform === "multi");
  return filtered;
}

export function getTemplate(id) {
  return TEMPLATES.find(t => t.id === id) || null;
}

export function applyTemplate(templateId, assets) {
  const template = getTemplate(templateId);
  if (!template) return { error: "Template not found" };
  return {
    template: template.id,
    name: template.name,
    aspect_ratio: template.aspect_ratio,
    zones: template.layout_data.zones.map((zone, i) => ({
      ...zone,
      asset: assets[i] || null,
    })),
  };
}

// Graphics Studio templates: print/social graphics with real text zones.
// Rendered by a legible-text design model (OR_GRAPHICS_MODEL), not the photo model.
export const GRAPHICS_TEMPLATES = [
  { id: "gfx_flyer", name: "Event Flyer", category: "print", aspect_ratio: "9:16", zones: ["headline", "date_block", "hero_image", "cta", "logo"], hint: "bold headline top, hero visual middle, date + CTA bottom" },
  { id: "gfx_poster", name: "Poster", category: "print", aspect_ratio: "9:16", zones: ["headline", "hero_image", "body", "logo"], hint: "cinematic hero visual, oversized headline, minimal body copy" },
  { id: "gfx_yt_thumb", name: "YouTube Thumbnail", category: "thumbnail", aspect_ratio: "16:9", zones: ["headline_3_words", "face_right", "contrast_bg"], hint: "3 big words max, expressive face right, high-contrast background" },
  { id: "gfx_business_card", name: "Business Card", category: "print", aspect_ratio: "16:9", zones: ["name", "title", "contact", "logo"], hint: "clean front layout, generous whitespace, logo mark" },
  { id: "gfx_album", name: "Album Cover", category: "music", aspect_ratio: "1:1", zones: ["artwork", "artist_name"], hint: "full-bleed artwork, small artist name, no clutter" },
  { id: "gfx_banner", name: "Web Banner", category: "web", aspect_ratio: "16:9", zones: ["headline_left", "cta_button", "visual_right"], hint: "headline left, CTA button, product visual right" },
  { id: "gfx_church", name: "Church Program", category: "print", aspect_ratio: "9:16", zones: ["theme_text", "date_venue", "hero_image", "pastor_line"], hint: "reverent theme typography, date and venue block, warm uplifting visual" },
  { id: "gfx_owambe", name: "Owambe Invite", category: "print", aspect_ratio: "9:16", zones: ["couple_names", "date_venue", "asoebi_note", "rsvp"], hint: "celebration glamour, gold accents, couple names hero, RSVP footer" },
  { id: "gfx_pricelist", name: "Price List", category: "vendor", aspect_ratio: "9:16", zones: ["brand_name", "item_rows", "cta_order", "logo"], hint: "scannable item rows with prices, order CTA, tidy vendor branding" },
  { id: "gfx_thrift", name: "Thrift Sale", category: "vendor", aspect_ratio: "1:1", zones: ["drop_headline", "price_burst", "items_grid", "dm_cta"], hint: "urgent drop energy, price burst badge, DM-to-order CTA" },
  { id: "gfx_estate", name: "Real Estate", category: "ad", aspect_ratio: "16:9", zones: ["property_hero", "headline_price", "features_row", "agent_cta"], hint: "aspirational property hero, price headline, feature ticks, agent CTA" },
  { id: "gfx_menu", name: "Food Menu", category: "vendor", aspect_ratio: "9:16", zones: ["brand_name", "dish_rows", "prices", "order_cta"], hint: "appetizing dish visuals, readable dish rows with prices, order CTA" },
  { id: "gfx_skincare", name: "Skincare Ad", category: "ad", aspect_ratio: "4:5", zones: ["product_hero", "benefit_lines", "price_cta"], hint: "clean beauty product hero, benefit bullets, price and shop CTA" },
  { id: "gfx_carousel", name: "Carousel Slide", category: "social_post", aspect_ratio: "1:1", zones: ["slide_number", "one_idea", "swipe_hint"], hint: "ONE idea per slide, big number, swipe cue, ultra-clean" },
];

export function getGraphicsTemplates() { return GRAPHICS_TEMPLATES; }
export function getGraphicsTemplate(id) { return GRAPHICS_TEMPLATES.find(t => t.id === id) || null; }

export const EXPORT_PRESETS = {
  tiktok: { resolution: "1080x1920", aspect_ratio: "9:16", format: "mp4", max_duration: 60, fps: 30 },
  instagram: { resolution: "1080x1080", aspect_ratio: "1:1", format: "mp4", max_duration: 60, fps: 30 },
  instagram_story: { resolution: "1080x1920", aspect_ratio: "9:16", format: "mp4", max_duration: 15, fps: 30 },
  youtube: { resolution: "1920x1080", aspect_ratio: "16:9", format: "mp4", max_duration: 600, fps: 30 },
  youtube_short: { resolution: "1080x1920", aspect_ratio: "9:16", format: "mp4", max_duration: 60, fps: 30 },
  twitter: { resolution: "1920x1080", aspect_ratio: "16:9", format: "mp4", max_duration: 140, fps: 30 },
  facebook: { resolution: "1920x1080", aspect_ratio: "16:9", format: "mp4", max_duration: 240, fps: 30 },
};
