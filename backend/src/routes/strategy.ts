import express from 'express';
import { body, param, query, validationResult } from 'express-validator';
import { MLService } from '../services/ml-service';
import { WebSocketService } from '../services/websocket-service';

export const strategyRoutes = (mlService: MLService, wsService: WebSocketService) => {
  const router = express.Router();

  // Run single strategy
  router.get('/:strategy', [
    param('strategy').isIn(['random', 'entropy', 'rl']),
    query('queries').isInt({ min: 1, max: 1000 }),
  ], async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const strategy = req.params.strategy;
      const queries = parseInt(req.query.queries as string);

      const result = await mlService.runStrategy(strategy, queries);

      wsService.broadcast('strategy', {
        action: 'completed',
        strategy,
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('Strategy execution error:', error);
      res.status(500).json({
        error: 'Failed to run strategy',
        details: error.message,
      });
    }
  });

  // Compare strategies
  router.post('/compare', [
    body('query_budget').isInt({ min: 1, max: 1000 }),
    body('num_runs').isInt({ min: 1, max: 10 }),
    body('strategies').isArray({ min: 1 }),
    body('strategies.*').isIn(['random', 'entropy', 'rl']),
  ], async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { query_budget, num_runs, strategies } = req.body;

      const result = await mlService.compareStrategies(query_budget, num_runs, strategies);

      wsService.broadcast('strategy', {
        action: 'comparison_completed',
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('Strategy comparison error:', error);
      res.status(500).json({
        error: 'Failed to compare strategies',
        details: error.message,
      });
    }
  });

  return router;
};