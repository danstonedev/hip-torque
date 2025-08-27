// Simple service worker to cache Pyodide runtime and app shell for faster subsequent loads
const CACHE_NAME = 'imu-hip-id-cache-v1';
const PYODIDE_BASE = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/';
const APP_SHELL = [
  '/',
  '/index.html',
  '/style.css',
  '/app.js?v=2025-08-27-2',
  '/py/pages_pipeline.py',
  '/py/hip_inverse_dynamics.py'
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    try {
      await cache.addAll(APP_SHELL);
      // Pre-cache key Pyodide assets
      await cache.addAll([
        PYODIDE_BASE + 'pyodide.js',
        PYODIDE_BASE + 'pyodide.asm.js',
        PYODIDE_BASE + 'pyodide.asm.wasm',
        PYODIDE_BASE + 'repodata.json',
      ]);
    } catch (e) {
      // Some CDNs may block opaque caching; ignore failures
    }
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle GET requests
  if (request.method !== 'GET') return;

  // Network-first for local app shell to pick up updates; fallback to cache
  if (url.origin === self.location.origin) {
    event.respondWith((async () => {
      try {
        const net = await fetch(request);
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, net.clone());
        return net;
      } catch (e) {
        const cached = await caches.match(request);
        return cached || Response.error();
      }
    })());
    return;
  }

  // Let Pyodide CDN assets bypass SW so module imports get proper CORS headers
  if (url.href.startsWith(PYODIDE_BASE)) {
    return; // don't intercept
  }
});
