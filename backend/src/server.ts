import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import dotenv from 'dotenv';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';

import { datasetRoutes } from './routes/dataset';
import { annotationRoutes } from './routes/annotation';
import { strategyRoutes } from './routes/strategy';
import { modelRoutes } from './routes/model';
import { rlRoutes } from './routes/rl';
import { MLService } from './services/ml-service';
import { WebSocketService } from './services/websocket-service';

// Load environment variables
dotenv.config();

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

// Initialize services
const mlService = new MLService();
const wsService = new WebSocketService(wss);

// Middleware
app.use(helmet());
app.use(compression());
app.use(morgan('combined'));
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Health check
app.get('/health', async (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {
      ml: await mlService.isHealthy(),
      websocket: wsService.isHealthy()
    }
  });
});

// API routes
app.use('/api/dataset', datasetRoutes(mlService, wsService));
app.use('/api/annotation', annotationRoutes(mlService, wsService));
app.use('/api/strategy', strategyRoutes(mlService, wsService));
app.use('/api/model', modelRoutes(mlService, wsService));
app.use('/api/rl', rlRoutes(mlService, wsService));

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'MedAL Backend API',
    version: '1.0.0',
    description: 'Backend for RL + Active Learning medical annotation',
    endpoints: {
      health: '/health',
      dataset: '/api/dataset',
      annotation: '/api/annotation',
      strategy: '/api/strategy',
      model: '/api/model',
      rl: '/api/rl'
    }
  });
});

// Error handling middleware
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    error: {
      message: err.message || 'Internal server error',
      status: err.status || 500
    }
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: {
      message: 'Endpoint not found',
      status: 404
    }
  });
});

const PORT = process.env.PORT || 3001;

server.listen(PORT, () => {
  console.log(`MedAL Backend running on port ${PORT}`);
  console.log(`ML Service: ${mlService.getBaseUrl()}`);
  console.log(`WebSocket server ready`);
});

export default app;
