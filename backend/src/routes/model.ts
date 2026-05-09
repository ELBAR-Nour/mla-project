import express from 'express';
import { MLService } from '../services/ml-service';
import { WebSocketService } from '../services/websocket-service';

export const modelRoutes = (mlService: MLService, wsService: WebSocketService) => {
  const router = express.Router();

  // Get model metrics
  router.get('/metrics', async (req, res) => {
    try {
      const metrics = await mlService.getModelMetrics();
      res.json(metrics);
    } catch (error: any) {
      console.error('Model metrics error:', error);
      res.status(500).json({
        error: 'Failed to get model metrics',
        details: error.message,
      });
    }
  });

  // Get confusion matrix
  router.get('/confusion-matrix', async (req, res) => {
    try {
      const cm = await mlService.getConfusionMatrix();
      res.json(cm);
    } catch (error: any) {
      console.error('Confusion matrix error:', error);
      res.status(500).json({
        error: 'Failed to get confusion matrix',
        details: error.message,
      });
    }
  });

  // Get ROC curve
  router.get('/roc', async (req, res) => {
    try {
      const roc = await mlService.getROCCurve();
      res.json(roc);
    } catch (error: any) {
      console.error('ROC curve error:', error);
      res.status(500).json({
        error: 'Failed to get ROC curve',
        details: error.message,
      });
    }
  });

  return router;
};