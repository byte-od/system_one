// System 1 Engine HUD — Service Worker
const CACHE_NAME = 's1-hud-v1';

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  // Pass-through fetch para telemetria em tempo real, SSE e vídeo sem interferência de cache
  event.respondWith(fetch(event.request).catch((err) => {
    return caches.match(event.request);
  }));
});
