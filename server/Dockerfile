FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY . .

# Create directories for uploads and reports
RUN mkdir -p uploads reports

EXPOSE 5000

CMD ["node", "src/server.js"]
