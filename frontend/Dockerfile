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
# Self-signed cert so the site can be served over https (needed for the mic to
# work on a phone over the LAN -- see nginx.conf). It's self-signed, so browsers
# warn once; that's expected for a LAN demo. For a real domain, replace these
# with a proper certificate (e.g. Let's Encrypt).
RUN apk add --no-cache openssl \
    && mkdir -p /etc/nginx/certs \
    && openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
        -keyout /etc/nginx/certs/selfsigned.key \
        -out /etc/nginx/certs/selfsigned.crt \
        -subj "/CN=consulate-dost-demo" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80 443
