import express from 'express';
import { body, validationResult } from 'express-validator';
import { MLService } from '../services/ml-service';
import { WebSocketService } from '../services/websocket-service';

export const rlRoutes = (mlService: MLService, wsService: WebSocketService) => {
  const router = express.Router();

  // List available RL checkpoints
  router.get('/checkpoints', async (req: express.Request, res: express.Response) => {
    try {
      const checkpoints = await mlService.listCheckpoints();
      res.json(checkpoints);
    } catch (error: any) {
      console.error('RL checkpoint listing error:', error);
      res.status(500).json({
        error: 'Failed to list RL checkpoints',
        details: error.message,
      });
    }
  });

  // Load an RL checkpoint
  router.post('/load-checkpoint', [
    body('checkpoint_name').isString().notEmpty(),
  ], async (req: express.Request, res: express.Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const result = await mlService.loadCheckpoint(req.body.checkpoint_name);

      wsService.broadcast('rl', {
        action: 'checkpoint_loaded',
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('RL checkpoint loading error:', error);
      res.status(500).json({
        error: 'Failed to load RL checkpoint',
        details: error.message,
      });
    }
  });

  // Run RL policy inference for a sample
  router.post('/decision', [
    body('checkpoint_name').optional().isString(),
    body('model_name').optional().isString(),
    body('image_id').isInt({ min: 0 }),
    body('dataset_name').optional().isString(),
    body('split').optional().isString(),
    body('budget_remaining').optional().isInt({ min: 0 }),
    body('max_budget').optional().isInt({ min: 1 }),
  ], async (req: express.Request, res: express.Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const result = await mlService.runRLDecision(req.body);

      wsService.broadcast('rl', {
        action: 'decision_completed',
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('RL decision error:', error);
      res.status(500).json({
        error: 'Failed to run RL decision',
        details: error.message,
      });
    }
  });

  // Get RL agent state
  router.get('/state', async (req: express.Request, res: express.Response) => {
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
  router.get('/policy', async (req: express.Request, res: express.Response) => {
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
  ], async (req: express.Request, res: express.Response) => {
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
  router.post('/reset', async (req: express.Request, res: express.Response) => {
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
