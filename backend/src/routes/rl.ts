import express from 'express';
import { body, validationResult } from 'express-validator';
import { MLService } from '../services/ml-service';
import { WebSocketService } from '../services/websocket-service';

export const rlRoutes = (mlService: MLService, wsService: WebSocketService) => {
  const router = express.Router();

  // Get RL agent state
  router.get('/state', async (req, res) => {
    try {
      const state = await mlService.getRLState();
      res.json(state);
    } catch (error: any) {
      console.error('RL state error:', error);
      res.status(500).json({
        error: 'Failed to get RL state',
        details: error.message,
      });
    }
  });

  // Get RL policy
  router.get('/policy', async (req, res) => {
    try {
      const policy = await mlService.getRLPolicy();
      res.json(policy);
    } catch (error: any) {
      console.error('RL policy error:', error);
      res.status(500).json({
        error: 'Failed to get RL policy',
        details: error.message,
      });
    }
  });

  // Train RL agent
  router.post('/train-step', [
    body('episodes').isInt({ min: 1, max: 1000 }),
    body('learning_rate').optional().isFloat({ min: 0.0001, max: 0.1 }),
    body('gamma').optional().isFloat({ min: 0, max: 1 }),
  ], async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { episodes, learning_rate = 0.001, gamma = 0.99 } = req.body;

      const result = await mlService.trainRLAgent(episodes, learning_rate, gamma);

      wsService.broadcast('rl', {
        action: 'training_completed',
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('RL training error:', error);
      res.status(500).json({
        error: 'Failed to train RL agent',
        details: error.message,
      });
    }
  });

  // Reset RL agent
  router.post('/reset', async (req, res) => {
    try {
      const result = await mlService.resetRLAgent();

      wsService.broadcast('rl', {
        action: 'reset',
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('RL reset error:', error);
      res.status(500).json({
        error: 'Failed to reset RL agent',
        details: error.message,
      });
    }
  });

  return router;
};