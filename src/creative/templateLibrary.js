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

export const EXPORT_PRESETS = {
  tiktok: { resolution: "1080x1920", aspect_ratio: "9:16", format: "mp4", max_duration: 60, fps: 30 },
  instagram: { resolution: "1080x1080", aspect_ratio: "1:1", format: "mp4", max_duration: 60, fps: 30 },
  instagram_story: { resolution: "1080x1920", aspect_ratio: "9:16", format: "mp4", max_duration: 15, fps: 30 },
  youtube: { resolution: "1920x1080", aspect_ratio: "16:9", format: "mp4", max_duration: 600, fps: 30 },
  youtube_short: { resolution: "1080x1920", aspect_ratio: "9:16", format: "mp4", max_duration: 60, fps: 30 },
  twitter: { resolution: "1920x1080", aspect_ratio: "16:9", format: "mp4", max_duration: 140, fps: 30 },
  facebook: { resolution: "1920x1080", aspect_ratio: "16:9", format: "mp4", max_duration: 240, fps: 30 },
};
