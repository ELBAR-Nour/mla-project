import express from 'express';
import { body, validationResult } from 'express-validator';
import { MLService } from '../services/ml-service';
import { WebSocketService } from '../services/websocket-service';

export const annotationRoutes = (mlService: MLService, wsService: WebSocketService) => {
  const router = express.Router();

  // Process annotation
  router.post('/', [
    body('image_id').isInt({ min: 0 }),
    body('action').isIn(['predict', 'request_label']),
    body('budget_remaining').isInt({ min: 0 }),
    body('max_budget').optional().isInt({ min: 1, max: 200 }),
  ], async (req: express.Request, res: express.Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { image_id, action, budget_remaining, max_budget } = req.body;

      const result = await mlService.processAnnotation(image_id, action, budget_remaining, max_budget);

      // Broadcast annotation result
      wsService.broadcast('annotation', {
        action: 'processed',
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('Annotation processing error:', error);
      res.status(500).json({
        error: 'Failed to process annotation',
        details: error.message,
      });
    }
  });

  return router;
};
