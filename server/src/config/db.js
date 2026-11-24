const mongoose = require('mongoose');

let isConnected = false;
let useMemoryStore = false;

const connectDB = async () => {
  const uri = process.env.MONGODB_URI || 'mongodb://localhost:27017/breast_cancer_db';
  
  try {
    const conn = await mongoose.connect(uri, {
      serverSelectionTimeoutMS: 3000,
    });
    isConnected = true;
    useMemoryStore = false;
    console.log(`[Database] MongoDB Connected successfully to host: ${conn.connection.host}`);
    try {
      const { seedDatabase } = require('./seedDB');
      await seedDatabase();
    } catch (seedErr) {
      console.warn(`[Database] Auto-seeding warning: ${seedErr.message}`);
    }
  } catch (error) {
    console.warn(`[Database] MongoDB connection failed (${error.message}).`);
    console.log('[Database] Activating High-Performance In-Memory Data Store fallback.');
    isConnected = false;
    useMemoryStore = true;
  }
};

const getDBStatus = () => ({
  connected: isConnected,
  useMemoryStore: useMemoryStore,
  mode: isConnected ? 'MongoDB Live' : 'In-Memory Store',
});

module.exports = { connectDB, getDBStatus, isUsingMemoryStore: () => useMemoryStore };
