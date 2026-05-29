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
      const response = mlService.formatError(error, 'Failed to list RL checkpoints');
      res.status(response.status).json(response.body);
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
      const response = mlService.formatError(error, 'Failed to load RL checkpoint');
      res.status(response.status).json(response.body);
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
      const response = mlService.formatError(error, 'Failed to run RL decision');
      res.status(response.status).json(response.body);
    }
  });

  // Get RL agent state
  router.get('/state', async (req: express.Request, res: express.Response) => {
    try {
      const state = await mlService.getRLState();
      res.json(state);
    } catch (error: any) {
      console.error('RL state error:', error);
      const response = mlService.formatError(error, 'Failed to get RL state');
      res.status(response.status).json(response.body);
    }
  });

  // Get RL policy
  router.get('/policy', async (req: express.Request, res: express.Response) => {
    try {
      const policy = await mlService.getRLPolicy();
      res.json(policy);
    } catch (error: any) {
      console.error('RL policy error:', error);
      const response = mlService.formatError(error, 'Failed to get RL policy');
      res.status(response.status).json(response.body);
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
      const response = mlService.formatError(error, 'Failed to train RL agent');
      res.status(response.status).json(response.body);
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
      const response = mlService.formatError(error, 'Failed to reset RL agent');
      res.status(response.status).json(response.body);
    }
  });

  return router;
};
