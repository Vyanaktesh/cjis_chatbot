# Frontend image: build the Vite SPA, then serve it (and reverse-proxy the API)
# with nginx. Multi-stage so the final image is just nginx + static files.

# --- build stage ---
FROM node:20-alpine AS build
WORKDIR /app

# VITE_API_BASE_URL is baked in at build time. Empty string => the app uses
# relative paths (/chat, /auth, ...), which nginx proxies to the backend on the
# same origin (see nginx.conf). That keeps the login cookie and CORS a non-issue.
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# --- serve stage ---
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
