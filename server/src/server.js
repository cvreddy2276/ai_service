require('dotenv').config();
const http = require('http');
const app = require('./app');
const { connectDB } = require('./config/db');

const PORT = process.env.PORT || 5000;

const startServer = async () => {
  // Connect Database (MongoDB or In-Memory store fallback)
  await connectDB();

  const server = http.createServer(app);

  server.listen(PORT, () => {
    console.log(`===================================================`);
    console.log(`🚀 Breast Cancer Detection Backend Server Running!`);
    console.log(`📡 Port: ${PORT}`);
    console.log(`🏥 Mode: ${process.env.NODE_ENV || 'development'}`);
    console.log(`🔗 API Base: http://localhost:${PORT}/api`);
    console.log(`===================================================`);
  });

  // Handle unhandled rejections
  process.on('unhandledRejection', (err) => {
    console.error('Unhandled Rejection:', err.message);
  });
};

startServer();
