FROM nginx:alpine

COPY . /usr/share/nginx/html

# Nginx config to proxy API calls and serve the SPA
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1
